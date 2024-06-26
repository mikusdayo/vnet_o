import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

import volseg as vs
import tqdm

from volseg.unet_3d.model import UNet3d
from volseg.example.brain_mri_dataset import BrainMRIDataset
from volseg.loss.dice import DiceLoss
from volseg.utils.image_dimension_wrapper import ImageDimensionsWrapper

val_set = '/Users/nicole/Documents/Anatomical_mag_echo5/img/'
train_set = '/Users/nicole/Documents/whole_liver_segmentation/'

def get_file_paths(image_directory, mask_directory):
    image_paths = [os.path.join(image_directory, filename) for filename in os.listdir(image_directory) if filename.endswith('.nii')]
    mask_paths = [os.path.join(mask_directory, filename) for filename in os.listdir(mask_directory) if filename.endswith('.nii')]
    return image_paths, mask_paths

image_paths, mask_paths = get_file_paths(val_set, train_set)


class NiftiDataset(Dataset):
    def __init__(self, image_files, mask_files):
        self.image_files = image_files
        self.mask_files = mask_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image and mask using nibabel
        image = nib.load(self.image_files[idx]).get_fdata()
        mask = nib.load(self.mask_files[idx]).get_fdata()

        # Pad depth to 40 if not already
        if image.shape[2] != 40:
            image = self.pad_to_depth(image, 40)
            mask = self.pad_to_depth(mask, 40)

        # Convert numpy arrays to PyTorch tensors with dtype float32
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)    # Add channel dimension

        return image_tensor, mask_tensor

    def pad_to_depth(self, volume, target_depth):
        current_depth = volume.shape[2]
        if current_depth < target_depth:
            pad_size = (target_depth - current_depth) // 2
            pad_extra = (target_depth - current_depth) % 2
            volume = np.pad(volume, ((0, 0), (0, 0), (pad_size, pad_size + pad_extra)), mode='constant', constant_values=0)
        return volume

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1, res_connect=False):
        super(ConvBlock, self).__init__()
        self.res_connect = res_connect
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout3d(dropout)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout3d(dropout)

        if res_connect:
            self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        if self.res_connect:
            identity = self.residual(identity)
            x += identity  # Element-wise addition for residual connection
        
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1, res_connect=False):
        super(EncoderBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels, dropout=dropout, res_connect=res_connect)
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

    def forward(self, x):
        x = self.conv_block(x)
        pooled = self.pool(x)
        return x, pooled

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.conv = ConvBlock(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # Ensure that the upsampled x has the same dimensions as skip before concatenating
        if x.size() != skip.size():
            # Optionally add a print statement here to debug sizes
            # print("Adjusting size from", x.size(), "to", skip.size())
            x = F.interpolate(x, size=skip.size()[2:], mode='trilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class VNet(nn.Module):
    def __init__(self):
        super(VNet, self).__init__()
        # Adjust the filters to start with 1 if the input images are single-channel
        filters = [1, 64, 128, 256, 256]  # Starting with 1 channel now
        self.encoders = nn.ModuleList([
            EncoderBlock(filters[i], filters[i+1], dropout=0.1, res_connect=True)
            for i in range(len(filters)-1)
        ])

        self.bottleneck = ConvBlock(filters[-1], filters[-1], dropout=0.1, res_connect=True)

        # Assuming symmetrical structure for the decoder
        self.decoders = nn.ModuleList([
            EncoderBlock(filters[i+1], filters[i], dropout=0.1, res_connect=True)
            for i in reversed(range(len(filters)-1))
        ])

        self.final_conv = nn.Conv3d(filters[0], 1, kernel_size=1)  # Output segmentation map

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            x, pooled = encoder(x)
            skips.append(x)
            x = pooled

        x = self.bottleneck(x)

        for decoder, skip in zip(self.decoders, reversed(skips)):
            x, _ = decoder(x)  # Ignore pooling in decoder
            x = F.interpolate(x, size=skip.size()[2:], mode='trilinear', align_corners=False)

        x = self.final_conv(x)
        return torch.sigmoid(x)  # Assuming binary classification

# Model instantiation
model = VNet()
print(model)

dataset = NiftiDataset(image_paths, mask_paths)
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Set up the optimizer, loss function, and model
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def dice_coefficient(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

best_dice = 0
'''
# Training loop
num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        outputs = UNet3d(num_classes=1, image_dimensions=image_dimensions).to(device)
        loss = criterion(outputs, masks)
'''
image_dimensions = ImageDimensionsWrapper((3, 96, 128, 128))

model = UNet3d(num_classes=1, image_dimensions=image_dimensions).to(device)

epochs = 10
batch_size = 2
terminate_after_no_improvement_epochs = 10
criterion = DiceLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

training_history = []
validation_history = []

best_loss = float("inf")
best_epoch = None
best_state_dict = None

for epoch in range(1, epochs + 1):
    print(f"Epoch {epoch}")
    total_training_loss = 0.0
    for batch_number, data in tqdm.tqdm(enumerate(train_loader, start=1), total=len(train_loader)):
        inputs, labels, _ = data

        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()

        outputs = model(inputs.float())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_training_loss += loss.item()
    training_loss = total_training_loss / len(train_set)
    training_history.append(training_loss)
    print(f'[Training loss: {training_loss:.3f}')

    total_validation_loss = 0.0
    with torch.no_grad():
        for batch_number, data in tqdm.tqdm(enumerate(val_loader, start=1), total=len(val_loader)):
            inputs, labels, _ = data
            inputs = (inputs - train_set.mean) / train_set.stdev

            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs.float())

            total_validation_loss += criterion(outputs, labels).item()
    validation_loss = total_validation_loss / len(val_set)
    validation_history.append(validation_loss)
    print(f'[Validation loss: {validation_loss:.3f}')

    if validation_loss < best_loss:
        best_epoch = epoch
        best_loss = validation_loss
        best_state_dict = model.state_dict().copy()
        torch.save(best_state_dict, "./best_weights.pth")
    elif terminate_after_no_improvement_epochs is not None:
        epochs_without_improvement = epoch - best_epoch
        if epochs_without_improvement >= terminate_after_no_improvement_epochs:
            print(f"Stopping training after {epoch} epochs")
            break
'''
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
'''
'''
    # Evaluation and checkpoint saving
    model.eval()
    with torch.no_grad():
        dice_scores = []
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            preds = (outputs > 0.5).float()
            dice_scores.append(dice_coefficient(preds, masks).item())
        
        average_dice = np.mean(dice_scores)
        if average_dice > best_dice:
            best_dice = average_dice
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Saved Best Model with Dice Coefficient: {best_dice}')
'''
    #print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader)}')

print("Training complete.")

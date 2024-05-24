import torch

# Based on: https://github.com/lyakaap/pytorch-template/blob/master/src/losses.py


class DiceLoss(torch.nn.Module):
    def forward(self, output, target):
        if torch.sum(target) == 0:
            output = 1.0 - output
            target = 1.0 - target

        dice_score = 2 * torch.sum(output * target) / torch.sum(output + target + 1e-7)
        return 1.0 - dice_score

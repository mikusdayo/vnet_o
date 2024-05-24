import math
import os
import random

import cv2
import numpy as np
import torch.utils.data
import tqdm
from skimage.transform import resize
from volseg.utils.io_utils import print_info_message

file_path = os.path.dirname(os.path.abspath(__file__))


class BrainMRIDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path_to_root,
        folders,
        reshape_dhw=None,
        cache_loaded_images=False,
        autoscale=False,
        autoscale_using_n_images=None,
        seed=None,
    ):
        self.path_to_root = path_to_root
        self.folders = folders
        self.reshape_dhw = reshape_dhw
        self.cache_loaded_images = cache_loaded_images
        self.cache = {}
        self.autoscale = autoscale
        if autoscale:
            self.mean, self.stdev = self.get_scale(
                use_n_images=autoscale_using_n_images, seed=seed
            )

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        if self.cache_loaded_images and idx in self.cache:
            image, mask = self.cache[idx]
            return image.copy(), mask.copy(), self.folders[idx]

        path = os.path.join(self.path_to_root, self.folders[idx])
        image = BrainMRIDataset.__load_image(path, is_mask=False)
        mask = BrainMRIDataset.__load_image(path, is_mask=True)

        if self.reshape_dhw is not None:
            image = (255 * resize(image, (image.shape[0], *self.reshape_dhw))).astype(
                int
            )
            mask = resize(mask, self.reshape_dhw)
        mask = (mask > 0).astype(int)

        if self.autoscale:
            image = (image - self.mean) / self.stdev
        if self.cache_loaded_images:
            self.cache[idx] = (image, mask)
        return image.copy(), mask.copy(), self.folders[idx]

    @staticmethod
    def __load_image(path_to_directory, is_mask):
        if is_mask:
            return np.array(
                [
                    cv2.imread(
                        os.path.join(path_to_directory, filename), cv2.IMREAD_GRAYSCALE
                    )
                    for filename in sorted(os.listdir(path_to_directory))
                    if "mask" in filename
                ]
            )
        else:
            image = np.array(
                [
                    cv2.cvtColor(
                        cv2.imread(os.path.join(path_to_directory, filename)),
                        cv2.COLOR_BGR2RGB,
                    )
                    for filename in sorted(os.listdir(path_to_directory))
                    if "mask" not in filename
                ]
            )
            image = np.moveaxis(image, -1, 0)
            return image

    def get_scale(self, use_n_images=None, seed=None):
        rnd = random.Random(seed)
        folders = (
            rnd.sample(self.folders, k=use_n_images)
            if use_n_images is not None
            else self.folders
        )
        print_info_message("Calculating mean")
        mean = self.get_mean(folders)
        print_info_message("Calculating standard deviation")
        stdev = self.get_stdev(folders, mean)
        return mean, stdev

    def get_mean(self, folders):
        accum = 0
        count = 0
        for folder in tqdm.tqdm(folders):
            path = os.path.join(self.path_to_root, folder)
            image = BrainMRIDataset.__load_image(path, is_mask=False)
            if self.reshape_dhw is not None:
                image = (
                    255 * resize(image, (image.shape[0], *self.reshape_dhw))
                ).astype(int)
            accum += image.sum()
            count += np.prod(image.shape)
        return accum / count

    def get_stdev(self, folders, mean):
        accum = 0
        count = 0
        for folder in tqdm.tqdm(folders):
            path = os.path.join(self.path_to_root, folder)
            image = BrainMRIDataset.__load_image(path, is_mask=False)
            if self.reshape_dhw is not None:
                image = (
                    255 * resize(image, (image.shape[0], *self.reshape_dhw))
                ).astype(int)
            stdev_parts = np.vectorize(lambda pixel: (pixel - mean) ** 2)
            accum += sum(stdev_parts(image.flatten()))
            count += np.prod(image.shape)
        return math.sqrt(accum / (count - 1))

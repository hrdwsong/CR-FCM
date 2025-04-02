import os.path as osp
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

import random
import math
import torch
from PIL import ImageFilter, ImageOps
import os

DATASET_DIR = r'G:\FSL_filelists\tieredIN\tiered_imagenet'


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class TieredImagenet(Dataset):
    """ Usage:
    """

    def __init__(self, setname, args):
        split_path = osp.join(DATASET_DIR, setname)
        cls_names = os.listdir(split_path)
        cls_names.sort()
        data = []
        label = []
        for idx, cls_name in enumerate(cls_names):
            cls_dir = os.path.join(split_path, cls_name)
            img_files = os.listdir(cls_dir)
            for img_file in img_files:
                data.append(os.path.join(cls_dir, img_file))
                label.append(idx)
        self.data = data
        self.label = label
        self.num_class = len(set(label))
        self.setname = setname

        image_size = 224  # 112, 128; 144, 168

        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.25, 1.0), interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(1.0),
            normalize,
        ])

        self.transform_val_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225]))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        if self.setname == 'train':
            image = self.transform_train(Image.open(path).convert('RGB'))
        else:
            image = self.transform_val_test(Image.open(path).convert('RGB'))
        return image, label


import os
import os.path as osp
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import ImageFilter, ImageOps
import random

data_path = r'G:\FSL_filelists\cifar_fs'


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


class CIFARFS(Dataset):
    def __init__(self, setname, args):
        dataset_dir = data_path
        if setname == 'train':
            path = osp.join(dataset_dir, 'meta-train')
            label_list = os.listdir(path)
        elif setname == 'test':
            path = osp.join(dataset_dir, 'meta-test')
            label_list = os.listdir(path)
        elif setname == 'val':
            path = osp.join(dataset_dir, 'meta-val')
            label_list = os.listdir(path)
        else:
            raise ValueError('Incorrect set name. Please check!')

        data = []
        label = []

        folders = [osp.join(path, label) for label in label_list if os.path.isdir(osp.join(path, label))]

        for idx, this_folder in enumerate(folders):
            this_folder_images = os.listdir(this_folder)
            for image_path in this_folder_images:
                data.append(osp.join(this_folder, image_path))
                label.append(idx)

        self.data = data
        self.label = label
        self.num_class = len(set(label))
        self.setname = setname

        # Transformation
        image_size = 224

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
            transforms.Normalize(np.array([0.5071, 0.4866, 0.4409]),
                                 np.array([0.2009, 0.1984, 0.2023])),
        ])
        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.25, 1.0), interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(1.0),
            normalize,
        ])

        self.transform_val_test = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(np.array([0.5071, 0.4866, 0.4409]),
                                 np.array([0.2009, 0.1984, 0.2023]))
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

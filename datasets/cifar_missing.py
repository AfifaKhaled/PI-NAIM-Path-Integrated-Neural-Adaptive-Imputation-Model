import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os


class CIFARMissingDataset(Dataset):
    def __init__(self, dataset_name='cifar10', train=True, missing_type='random', missing_rate=0.5,
                 data_dir='./data', download=True):
        """
        CIFAR dataset with synthetic missing pixels

        Args:
            dataset_name: 'cifar10' or 'cifar100'
            train: True for training, False for testing
            missing_type: 'random', 'block', 'column'
            missing_rate: proportion of missing pixels (0-1)
            data_dir: directory to store data
            download: whether to download the dataset
        """
        self.dataset_name = dataset_name
        self.missing_type = missing_type
        self.missing_rate = missing_rate

        # Load CIFAR dataset
        if dataset_name == 'cifar10':
            self.dataset = torchvision.datasets.CIFAR10(
                root=data_dir, train=train, download=download,
                transform=transforms.ToTensor()
            )
        elif dataset_name == 'cifar100':
            self.dataset = torchvision.datasets.CIFAR100(
                root=data_dir, train=train, download=download,
                transform=transforms.ToTensor()
            )
        else:
            raise ValueError("dataset_name must be 'cifar10' or 'cifar100'")

    def __len__(self):
        return len(self.dataset)

    def create_missing_mask(self, img):
        """
        Create missing pixel mask for the image
        """
        C, H, W = img.shape

        if self.missing_type == 'random':
            # Random missing pixels
            mask = torch.rand(C, H, W) > self.missing_rate

        elif self.missing_type == 'block':
            # Block missing (random rectangular regions)
            mask = torch.ones(C, H, W)
            block_size_h = int(H * 0.3)  # 30% of height
            block_size_w = int(W * 0.3)  # 30% of width

            # Create multiple random blocks
            num_blocks = max(1, int(self.missing_rate * 10))
            for _ in range(num_blocks):
                start_h = np.random.randint(0, H - block_size_h)
                start_w = np.random.randint(0, W - block_size_w)
                mask[:, start_h:start_h + block_size_h, start_w:start_w + block_size_w] = 0

        elif self.missing_type == 'column':
            # Missing columns
            mask = torch.ones(C, H, W)
            num_missing_cols = int(W * self.missing_rate)
            missing_cols = np.random.choice(W, num_missing_cols, replace=False)
            mask[:, :, missing_cols] = 0

        else:
            raise ValueError("missing_type must be 'random', 'block', or 'column'")

        return mask.float()

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        # Create missing mask
        mask = self.create_missing_mask(img)

        # Apply missing pixels (set missing pixels to 0)
        incomplete_img = img * mask

        return {
            'incomplete': incomplete_img,
            'mask': mask,
            'complete': img,
            'label': label
        }


def get_cifar_missing_loaders(dataset_name='cifar10', batch_size=128, missing_type='random',
                              missing_rate=0.5, num_workers=4):
    """
    Get DataLoaders for CIFAR with missing pixels
    """
    train_dataset = CIFARMissingDataset(
        dataset_name=dataset_name,
        train=True,
        missing_type=missing_type,
        missing_rate=missing_rate
    )

    test_dataset = CIFARMissingDataset(
        dataset_name=dataset_name,
        train=False,
        missing_type=missing_type,
        missing_rate=missing_rate
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader
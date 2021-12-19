import os
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms as transforms

import pytorch_lightning as pl
import datasets

class mnist_wrapper(Dataset):
    def __init__(self, mnist):
        self._dataset = mnist
    
    def __len__(self):
        return len(self._dataset)
    
    def __getitem__(self, idx):
        img, label = self._dataset[idx]
        return {
            'img': img,
            'label': label,
        }


@datasets.register('mnist')
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.root_dir = config.root_dir
        self.batch_size = config.batch_size
        self.transform = transforms.Compose([transforms.ToTensor()])
    
    def setup(self, stage=None):
        # ([1, 28, 28], {0, 1, 2, ..., 9})
        if stage == 'fit' or stage is None:
            mnist_full = torchvision.datasets.MNIST(self.root_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
        if stage == 'test' or stage is None:
            self.mnist_test = torchvision.datasets.MNIST(self.root_dir, train=False, transform=self.transform)
        if stage == 'predict' or stage is None:
            self.mnist_predict = torchvision.datasets.MNIST(self.root_dir, train=False, transform=self.transform)

    def prepare_data(self):
        torchvision.datasets.MNIST(self.root_dir, train=True, download=True)
        torchvision.datasets.MNIST(self.root_dir, train=False, download=True)
    
    def general_loader(self, dataset, split):
        return DataLoader(
            dataset, 
            shuffle=(split=='train'), 
            num_workers=os.cpu_count(), 
            batch_size=self.batch_size,
            pin_memory=True,
        )
    
    def train_dataloader(self):
        return self.general_loader(mnist_wrapper(self.mnist_train), 'train')

    def val_dataloader(self):
        return self.general_loader(mnist_wrapper(self.mnist_val), 'val')

    def test_dataloader(self):
        return self.general_loader(mnist_wrapper(self.mnist_test), 'test')
    
    def test_dataloader(self):
        return self.general_loader(mnist_wrapper(self.mnist_predict), 'predict')

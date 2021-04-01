# Jonas Braun
# jonas.braun@epfl.ch
# 22.02.2021

# copied from Semih Günel's repo https://github.com/NeLy-EPFL/Drosoph2PRegistration

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Compose, ToPILImage
import torchvision.transforms.functional as TF
import Augmentor as A
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
import torch


class Dataset2P(torch.utils.data.Dataset):
    def __init__(self, path, data=None):
        self.path = path
        p = A.Pipeline()
        p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)
        self.aug = Compose([ToPILImage(), p.torch_transform(), ToTensor()])

        if data is None:
            self.data = np.load(self.path)[:, np.newaxis, :, :]
        else:
            self.data = data[:, np.newaxis, :, :]

    def __getitem__(self, i):
        k_ind = torch.randint(low=0, high=len(self), size=[1])
        return (
            torch.from_numpy(self.data[i]).float(),
            torch.from_numpy(self.data[i + 1]).float(),
            torch.from_numpy(self.data[k_ind]).float(),
        )

    def __len__(self):
        return self.data.shape[0]  - 1 


class Dataset2PLight(pl.LightningDataModule):
    def __init__(self, path, shuffle=True, batch_size=32, data=None):
        super().__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.path = path
        self.data = data

    def train_dataloader(self):
        return DataLoader(
            Dataset2P(self.path, data=self.data),
            batch_size=self.batch_size,
            num_workers=20,
            shuffle=self.shuffle,
            pin_memory=True,
            drop_last=False,
        )


from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl


class CifarDataModule(pl.LightningDataModule):
    mean: float = 0.4734
    std: float = 0.2516

    def __init__(
        self,
        dataset_path: Path,
        batch_size: int,
        num_workers: int = 2,
        standardize: bool = False,
    ):
        super().__init__()
        self._dataset_path = dataset_path
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._transforms = transforms.ToTensor()
        if standardize:
            self._transforms = transforms.Compose(
                [self._transforms, transforms.Normalize(mean=self.mean, std=self.std)]
            )

    def setup(self, stage=None):
        self.dataset_train = datasets.CIFAR10(
            self._dataset_path,
            train=True,
            download=True,
            transform=self._transforms,
        )
        self.dataset_val = datasets.CIFAR10(
            self._dataset_path,
            train=False,
            download=True,
            transform=self._transforms,
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=True,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=False,
            collate_fn=self._collate_fn,
        )

    @staticmethod
    def _collate_fn(batch):
        return torch.stack([images for images, label in batch])

    def rescale(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.std + self.mean

import json
import tarfile
import pickle
import hashlib
from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import gdown


class MiniImagenet(Dataset):
    """Dataset for Mini-ImageNet.

    This dataset was introducedd in https://arxiv.org/abs/1606.04080, the actual files
    were downloaded and processed using code from
    https://github.com/tristandeleu/pytorch-meta. This Dataset loads the selected
    split's labels and images. The `__getitem__` method treats the dataset as a stack of
    all classes. For this reason, it should be used with a dataloader set to suffle
    samples during training.

    The `__getitem__` method selects an image from the dataset based on its index and
    normalize it by dividing each pixel by `255`. If `standardize` is set to `True`,
    returns an image normalized with the dataset's mean and std.

    Parameters
    ----------
    root: Path to the Mini ImageNet dataset folder.
    split: One of `"train"`, `"test"` or `"val"`. Default is `"train"`.
    download: If `True` and the `root` folder can't be found, attemps to download the
    dataset. Default is `False`.
    standardize: If `True`, normalize the images by the dataset's mean and std. Default
    is False.

    Attributes
    ----------
    mean: mean of the dataset, computed on the training split.
    std: std of the dataset, computed on the training split.
    """

    gdrive_id = "16V_ZlkW4SsnNDtnGmaBRq2OoPmUOc5mY"
    gz_filename = "mini-imagenet.tar.gz"
    gz_md5 = "b38f1eb4251fb9459ecc8e7febf9b2eb"
    std: float = 0.2848
    mean: float = 0.4410

    def __init__(
        self,
        root: Path,
        split: str = "train",
        download: bool = False,
        standardize: bool = False,
    ):
        # Select split
        _allowed_splits = ["train", "test", "val"]
        assert (
            split in _allowed_splits
        ), f"Invalid split {split}, must be in {_allowed_splits}."
        root = Path(root)
        if not root.is_dir():
            if download:
                self.download(root)
            else:
                raise FileNotFoundError("Dataset not found.")
        # Load labels
        with open(root / f"{split}_labels.json", "r") as json_file:
            self.labels = json.load(json_file)
        # Load images
        f = h5py.File(root / f"{split}_data.hdf5", "r")
        self.dset_images = f["datasets"]
        self._standardize = standardize

    def __len__(self):
        return self.n_classes * self.n_image_per_class

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Select image
        image = self.dset_images[self.labels[idx // self.n_image_per_class]][
            idx % self.n_image_per_class
        ]
        # Normalize
        image = image / 255
        # Swap axis to channel first
        image = image.transpose(2, 0, 1)
        image = torch.Tensor(image)
        if self._standardize:
            image = (image - self.mean) / self.std
        return image

    def download(self, root: Path):
        # Create directory
        root.mkdir()
        # Download archive from Google drive
        gdown.download(id=self.gdrive_id, output=str(root / self.gz_filename))
        # Check hash
        with open(root / self.gz_filename, "rb") as f:
            data = f.read()
            computed_md5 = hashlib.md5(data).hexdigest()
        assert computed_md5 == self.gz_md5
        # Extract file
        with tarfile.open(root / self.gz_filename) as f:
            for member in f.getmembers():
                f.extract(member=member, path=root)
        # Convert to hdf5 for data and json for labels
        for split in ["train", "val", "test"]:
            split_file = next(root.glob(f"*{split}.pkl"))
            # Read pkl file
            with open(split_file, "rb") as f:
                data = pickle.load(f)
                images, classes = data["image_data"], data["class_dict"]
            # Write hdf5 file
            with h5py.File(split_file.with_name(f"{split}_data.hdf5"), "w") as f:
                group = f.create_group("datasets")
                for name, indices in classes.items():
                    group.create_dataset(name, data=images[indices])
            # Write json file
            with open(split_file.with_name(f"{split}_labels.json"), "w") as f:
                labels = sorted(list(classes.keys()))
                json.dump(labels, f)

    @property
    def n_classes(self):
        return len(self.labels)

    @property
    def n_image_per_class(self):
        return self.dset_images[self.labels[0]].shape[0]

    def rescale(self, tensor: torch.Tensor) -> torch.Tensor:
        """Rescale a noramlized tensor.

        This function simply rescales a tensor from a properly noramlized (i.e. centered
        with unit variance) dataset.

        Parameters
        ----------
        tensor: input normalized tensor.

        Returns
        -------
        Rescaled tensor with values between `0` and `1`.
        """
        return tensor * self.std + self.mean


class MiniImagenetDataModule(pl.LightningDataModule):
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
        self._standardize = standardize

    def setup(self, stage=None):
        self.dataset_train = MiniImagenet(
            self._dataset_path,
            split="train",
            download=True,
            standardize=self._standardize,
        )
        self.dataset_val = MiniImagenet(
            self._dataset_path,
            split="val",
            download=True,
            standardize=self._standardize,
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=False,
        )

    def rescale(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.dataset_train.std + self.dataset_train.mean

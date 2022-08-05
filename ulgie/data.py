import json
import os
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class SimpleDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        max_num_images: Optional[int],
        transforms: Callable,
        max_n_objects: int = 10,
        split: str = "train",
    ):
        super().__init__()
        self.data_root = data_root
        self.transforms = transforms
        self.data_path = os.path.join(data_root, "images", split)
        self.split = split
        assert os.path.exists(self.data_root), f"Path {self.data_root} does not exist"
        assert self.split == "train" or self.split == "val" or self.split == "test"
        assert os.path.exists(self.data_path), f"Path {self.data_path} does not exist"

    def __getitem__(self, index: int):
        image_path = f"{self.data_root}/images/{self.split}/{str(index)+'.png'}"
        img = Image.open(image_path)
        img = img.convert("RGB")
        return self.transforms(img)

    def __len__(self):
        return 20000 if self.split == "train" else 5000


class SimpleDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        train_batch_size: int,
        val_batch_size: int,
        transforms: Callable,
        max_n_objects: int,
        num_workers: int,
        num_train_images: Optional[int] = None,
        num_val_images: Optional[int] = None,
    ):
        super().__init__()
        self.data_root = data_root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.transforms = transforms
        self.max_n_objects = max_n_objects
        self.num_workers = num_workers
        self.num_train_images = num_train_images
        self.num_val_images = num_val_images

        self.train_dataset = SimpleDataset(
            data_root=self.data_root,
            max_num_images=self.num_train_images,
            transforms=self.clevr_transforms,
            split="train",
            max_n_objects=self.max_n_objects,
        )
        self.val_dataset = SimpleDataset(
            data_root=self.data_root,
            max_num_images=self.num_val_images,
            transforms=self.clevr_transforms,
            split="val",
            max_n_objects=self.max_n_objects,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class SimpleTransforms(object):
    def __init__(self, resolution: Tuple[int, int]):
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda X: 2 * X - 1.0),  # rescale between -1 and 1
                transforms.Resize(resolution),
            ]
        )

    def __call__(self, input, *args, **kwargs):
        return self.transforms(input)

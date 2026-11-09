"""Image data pipeline for HiNet training (official datasets.py / train.py).

Official behaviour reproduced:
- one flat folder of images per split (DIV2K train_HR / valid_HR, PNG);
- train: random H/V flips + RandomCrop(224); val: CenterCrop(1024);
- pixels in [0, 1], no normalisation;
- each batch is split in half: first half = secret, second half = cover.

Deliberate differences:
- unreadable files raise an error naming them (official: silently skipped);
- images smaller than the crop raise by default, or are upscaled with
  small_image_policy="resize";
- num_workers / pin_memory default to 0 / False (official: 8 / True).
"""

from model.data.config import POLICIES, DataConfig, SmallImagePolicy
from model.data.dataset import ImageFolderDataset
from model.data.files import list_images, natural_sort_key
from model.data.loaders import (
    build_train_dataset,
    build_train_loader,
    build_val_dataset,
    build_val_loader,
    split_cover_secret,
)
from model.data.transforms import EnsureMinSize, train_transform, val_transform

__all__ = [
    "POLICIES",
    "DataConfig",
    "EnsureMinSize",
    "ImageFolderDataset",
    "SmallImagePolicy",
    "build_train_dataset",
    "build_train_loader",
    "build_val_dataset",
    "build_val_loader",
    "list_images",
    "natural_sort_key",
    "split_cover_secret",
    "train_transform",
    "val_transform",
]

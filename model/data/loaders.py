"""DataLoaders and the official cover/secret batch split.

Random augmentation draws from torch's RNG: the global RNG when num_workers=0,
and each worker's RNG (seeded from the DataLoader generator) otherwise. Seed
both for reproducible runs.
"""

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from model.data.config import DataConfig
from model.data.dataset import ImageFolderDataset
from model.data.transforms import train_transform, val_transform


def split_cover_secret(batch: Tensor) -> tuple[Tensor, Tensor]:
    """Official split: secret = first half, cover = second half. Returns (cover, secret)."""
    if batch.ndim != 4:
        raise ValueError(f"Expected (B, C, H, W) batch; got shape {tuple(batch.shape)}.")
    size = batch.shape[0]
    if size < 2 or size % 2:
        raise ValueError(f"Batch size must be even and >= 2; got {size}.")
    half = size // 2
    return batch[half:], batch[:half]


def build_train_dataset(config: DataConfig) -> ImageFolderDataset:
    if config.train_dir is None:
        raise ValueError("DataConfig.train_dir is not set.")
    return ImageFolderDataset(
        config.train_dir,
        train_transform(config.crop_size, config.small_image_policy),
        config.extensions,
        config.recursive,
    )


def build_val_dataset(config: DataConfig) -> ImageFolderDataset:
    if config.val_dir is None:
        raise ValueError("DataConfig.val_dir is not set.")
    return ImageFolderDataset(
        config.val_dir,
        val_transform(config.val_crop_size, config.small_image_policy),
        config.extensions,
        config.recursive,
    )


def _check_enough_images(dataset: ImageFolderDataset, batch_size: int) -> None:
    if len(dataset) < batch_size:
        raise ValueError(
            f"{dataset.root} has {len(dataset)} images, fewer than the batch size "
            f"{batch_size}; with drop_last=True no batch would be produced."
        )


def build_train_loader(
    config: DataConfig, generator: torch.Generator | None = None
) -> DataLoader:
    dataset = build_train_dataset(config)
    _check_enough_images(dataset, config.batch_size)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0,
        generator=generator,
    )


def build_val_loader(config: DataConfig) -> DataLoader:
    dataset = build_val_dataset(config)
    _check_enough_images(dataset, config.val_batch_size)
    return DataLoader(
        dataset,
        batch_size=config.val_batch_size,
        shuffle=False,
        drop_last=True,  # official
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0,
    )

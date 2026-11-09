"""Data pipeline settings. Defaults follow the official HiNet config.py / datasets.py."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SmallImagePolicy = Literal["error", "resize"]
POLICIES = ("error", "resize")


@dataclass(frozen=True)
class DataConfig:
    train_dir: Path | None = None
    val_dir: Path | None = None
    extensions: tuple[str, ...] = ("png",)  # official: format_train = format_val = 'png'
    recursive: bool = False
    crop_size: int = 224  # official: cropsize = 224
    val_crop_size: int = 1024  # official: cropsize_val = 1024
    batch_size: int = 16  # official: batch_size = 16 (8 cover + 8 secret)
    val_batch_size: int = 2  # official: batchsize_val = 2
    num_workers: int = 0  # official: 8 (train), 1 (val)
    pin_memory: bool = False  # official: True
    small_image_policy: SmallImagePolicy = "error"

    def __post_init__(self) -> None:
        for name in ("crop_size", "val_crop_size"):
            size = getattr(self, name)
            if size < 2 or size % 2:
                raise ValueError(f"{name} must be an even integer >= 2 (Haar DWT); got {size}")
        for name in ("batch_size", "val_batch_size"):
            size = getattr(self, name)
            if size < 2 or size % 2:
                raise ValueError(
                    f"{name} must be an even integer >= 2 (split into cover/secret); got {size}"
                )
        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0")
        if self.small_image_policy not in POLICIES:
            raise ValueError(f"small_image_policy must be one of {POLICIES}")
        if not self.extensions:
            raise ValueError("extensions must not be empty")
        normalized = tuple(ext.lower().lstrip(".") for ext in self.extensions)
        object.__setattr__(self, "extensions", normalized)
        for name in ("train_dir", "val_dir"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value))

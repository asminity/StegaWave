"""Device selection and seeding shared by training and evaluation."""

import random

import numpy as np
import torch


def resolve_device(name: str) -> torch.device:
    """'auto' -> cuda if available, else cpu; anything else is passed to torch.device."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

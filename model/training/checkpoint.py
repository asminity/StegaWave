"""Checkpoints for the from-scratch HiNet rebuild.

Format (a plain dict, loadable with torch.load(weights_only=True)):
    format      "stegawave-hinet-v1"
    epoch       last completed epoch
    net         model state_dict
    opt         optimizer state_dict      (official key)
    sched       LR scheduler state_dict
    config      ExperimentConfig.to_dict()
    rng         torch / cuda / python / numpy RNG state
    loader_rng  state of the training DataLoader's shuffle generator
    metrics     metrics of the last completed epoch
    aborted     True if written by the error handler mid-epoch

These checkpoints are unrelated to the pretrained checkpoints/model.pt and are
not loadable by src/stegawave (different parameter names).
"""

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

FORMAT = "stegawave-hinet-v1"


def capture_rng_state() -> dict[str, Any]:
    name, keys, pos, has_gauss, cached = np.random.get_state()
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": {
            "name": name,
            "keys": torch.from_numpy(keys.astype(np.int64)),
            "pos": int(pos),
            "has_gauss": int(has_gauss),
            "cached": float(cached),
        },
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    np_state = state["numpy"]
    np.random.set_state((
        np_state["name"],
        np_state["keys"].numpy().astype(np.uint32),
        np_state["pos"],
        np_state["has_gauss"],
        np_state["cached"],
    ))
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: Path | str,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    config: dict[str, Any],
    loader_generator: torch.Generator | None = None,
    metrics: dict[str, float] | None = None,
    aborted: bool = False,
) -> Path:
    """Write atomically (temp file + rename) so a crash never leaves a torn file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": FORMAT,
        "epoch": epoch,
        "net": model.state_dict(),
        "opt": optimizer.state_dict(),
        "sched": scheduler.state_dict(),
        "config": config,
        "rng": capture_rng_state(),
        "loader_rng": loader_generator.get_state() if loader_generator is not None else None,
        "metrics": dict(metrics or {}),
        "aborted": aborted,
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path: Path | str, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != FORMAT:
        raise ValueError(
            f"{path} is not a {FORMAT} checkpoint (the pretrained checkpoints/model.pt "
            "uses a different format and is intentionally not supported)."
        )
    return checkpoint


def load_trained_model(
    path: Path | str, map_location: str | torch.device = "cpu"
) -> nn.Module:
    """Rebuild a HiNetSteganography from a training checkpoint, in eval mode."""
    from model.architecture.steganography import HiNetSteganography
    from model.training.config import ExperimentConfig

    checkpoint = load_checkpoint(path, map_location)
    config = ExperimentConfig.from_dict(checkpoint["config"])
    model = HiNetSteganography(config.architecture, initialize=False)
    model.load_state_dict(checkpoint["net"])
    return model.to(map_location).eval()

"""Command-line entry point for from-scratch HiNet training.

Examples (from the repository root):
    python -m model.cli.train --train-dir data/DIV2K_train_HR --val-dir data/DIV2K_valid_HR
    python -m model.cli.train --config my_run.json
    python -m model.cli.train --resume outputs/hinet/latest.pt
    python -m model.cli.train --train-dir ... --epochs 1 --max-train-batches 2   # smoke test

Settings are resolved in order: defaults (official HiNet values) < --config JSON
< the checkpoint's saved config when resuming < command-line flags.
"""

import argparse
import dataclasses
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from model.training.checkpoint import load_checkpoint
from model.training.config import ExperimentConfig
from model.training.trainer import Trainer

# flag -> (config section, field)
_OVERRIDES: dict[str, tuple[str, str]] = {
    "train_dir": ("data", "train_dir"),
    "val_dir": ("data", "val_dir"),
    "crop_size": ("data", "crop_size"),
    "val_crop_size": ("data", "val_crop_size"),
    "batch_size": ("data", "batch_size"),
    "val_batch_size": ("data", "val_batch_size"),
    "num_workers": ("data", "num_workers"),
    "small_image_policy": ("data", "small_image_policy"),
    "epochs": ("train", "epochs"),
    "lr": ("train", "learning_rate"),
    "seed": ("train", "seed"),
    "device": ("train", "device"),
    "output_dir": ("train", "output_dir"),
    "val_every": ("train", "val_every"),
    "save_every": ("train", "save_every"),
    "grad_clip_norm": ("train", "grad_clip_norm"),
    "max_train_batches": ("train", "max_train_batches"),
    "max_val_batches": ("train", "max_val_batches"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train HiNet from random initialisation.")
    parser.add_argument("--config", type=Path, help="ExperimentConfig JSON file")
    parser.add_argument("--resume", type=Path, help="training checkpoint to resume from")
    parser.add_argument("--overwrite", action="store_true",
                        help="allow writing into an output dir that already has checkpoints")
    parser.add_argument("--train-dir", type=Path)
    parser.add_argument("--val-dir", type=Path)
    parser.add_argument("--crop-size", type=int)
    parser.add_argument("--val-crop-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--val-batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--small-image-policy", choices=("error", "resize"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--val-every", type=int)
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--grad-clip-norm", type=float)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    return parser


def resolve_config(args: argparse.Namespace) -> ExperimentConfig:
    if args.resume is not None:
        config = ExperimentConfig.from_dict(load_checkpoint(args.resume)["config"])
    elif args.config is not None:
        config = ExperimentConfig.load_json(args.config)
    else:
        config = ExperimentConfig()

    changes: dict[str, dict[str, Any]] = {}
    for flag, (section, name) in _OVERRIDES.items():
        value = getattr(args, flag)
        if value is not None:
            changes.setdefault(section, {})[name] = value
    for section, fields in changes.items():
        config = dataclasses.replace(
            config, **{section: dataclasses.replace(getattr(config, section), **fields)}
        )
    return config


def main(argv: Sequence[str] | None = None) -> Trainer:
    args = build_parser().parse_args(argv)
    config = resolve_config(args)
    if config.data.train_dir is None:
        raise SystemExit("error: --train-dir (or data.train_dir in --config) is required")

    trainer = Trainer(config)
    if args.resume is not None:
        trainer.resume(args.resume)
    trainer.fit(overwrite=args.overwrite)
    return trainer


if __name__ == "__main__":
    main()

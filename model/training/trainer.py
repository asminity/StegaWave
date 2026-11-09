"""HiNet training loop (official train.py), with resumable checkpoints.

Per epoch, as in the official code:
    for each batch: split into cover/secret, hide + reveal, weighted loss,
                    zero_grad, backward, optimizer step
    every val_every epochs: validation (official-style PSNR for cover and secret)
    every save_every epochs: checkpoint
    scheduler.step() once per epoch
On any exception (including Ctrl+C) a model_ABORT.pt checkpoint is written and
the exception re-raised (official checkpoint_on_error behaviour).

Additions over the official loop: seeded initialisation, RNG-exact resume,
latest.pt every epoch, metrics.jsonl, a non-finite loss guard, and optional
gradient clipping (off by default, as in the official code).

Reverse-pass z during training is drawn from torch's global RNG, like the
official gauss_noise; validation uses a fixed-seed generator so validation
numbers are comparable across epochs.
"""

import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from model.architecture.steganography import HiNetSteganography
from model.data import build_train_loader, build_val_loader, split_cover_secret
from model.evaluation.metrics import official_psnr
from model.training.checkpoint import (
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from model.training.config import ExperimentConfig
from model.training.losses import HiNetLoss
from model.training.runtime import resolve_device, seed_everything

REPO_ROOT = Path(__file__).resolve().parents[2]
PRETRAINED_DIR = REPO_ROOT / "checkpoints"

CONFIG_FILE = "config.json"
METRICS_FILE = "metrics.jsonl"
LATEST_FILE = "latest.pt"
FINAL_FILE = "model.pt"
ABORT_FILE = "model_ABORT.pt"


def periodic_checkpoint_name(epoch: int) -> str:
    return f"model_checkpoint_{epoch:05d}.pt"  # official naming


class Trainer:
    def __init__(
        self,
        config: ExperimentConfig,
        log: Callable[[str], None] = print,
    ):
        self.config = config
        self.log = log
        train = config.train
        self.device = resolve_device(train.device)
        self.output_dir = Path(train.output_dir)
        _reject_pretrained_dir(self.output_dir)

        seed_everything(train.seed)
        self.model = HiNetSteganography(
            config.architecture,
            initialize=True,
            generator=torch.Generator().manual_seed(train.seed),
        ).to(self.device)
        self.criterion = HiNetLoss(config.loss, config.architecture.in_channels)
        self.optimizer = Adam(
            self.model.parameters(),
            lr=train.learning_rate,
            betas=train.betas,
            eps=train.eps,
            weight_decay=train.weight_decay,
        )
        self.scheduler = StepLR(self.optimizer, step_size=train.lr_step, gamma=train.lr_gamma)

        self.loader_generator = torch.Generator().manual_seed(train.seed + 1)
        self.train_loader = build_train_loader(config.data, self.loader_generator)
        self.val_loader = build_val_loader(config.data) if config.data.val_dir else None

        self.epoch = 0  # last completed epoch
        self.history: list[dict[str, Any]] = []
        self._resumed = False

    # ------------------------------------------------------------------ resume

    def resume(self, path: Path | str) -> None:
        checkpoint = load_checkpoint(path, map_location=self.device)
        saved = ExperimentConfig.from_dict(checkpoint["config"])
        if saved.architecture != self.config.architecture:
            raise ValueError(
                f"Architecture in {path} ({saved.architecture}) does not match the "
                f"current config ({self.config.architecture})."
            )
        self.model.load_state_dict(checkpoint["net"])
        self.optimizer.load_state_dict(checkpoint["opt"])
        self.scheduler.load_state_dict(checkpoint["sched"])
        if checkpoint.get("loader_rng") is not None:
            self.loader_generator.set_state(checkpoint["loader_rng"])
        restore_rng_state(checkpoint["rng"])
        self.epoch = int(checkpoint["epoch"])
        self._resumed = True
        note = " (aborted mid-epoch)" if checkpoint.get("aborted") else ""
        self.log(f"Resumed from {path} at epoch {self.epoch}{note}.")

    # ------------------------------------------------------------- one epoch

    def train_epoch(self) -> dict[str, float]:
        self.model.train()
        limit = self.config.train.max_train_batches
        clip = self.config.train.grad_clip_norm
        sums: dict[str, float] = defaultdict(float)
        batches = 0

        for index, batch in enumerate(self.train_loader):
            if limit is not None and index >= limit:
                break
            batch = batch.to(self.device, non_blocking=True)
            cover, secret = split_cover_secret(batch)

            hidden, revealed = self.model(cover, secret)
            losses = self.criterion(hidden, revealed, cover, secret)
            if self.config.train.stop_on_nonfinite and not torch.isfinite(losses.total):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {self.epoch + 1}, batch {index}: "
                    f"{losses.as_dict()}"
                )

            self.optimizer.zero_grad(set_to_none=True)
            losses.total.backward()
            if clip is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.optimizer.step()

            for key, value in losses.as_dict().items():
                sums[key] += value
            batches += 1

        if batches == 0:
            raise RuntimeError("The training loader produced no batches.")
        return {f"train_{key}": value / batches for key, value in sums.items()}

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        if self.val_loader is None:
            return {}
        self.model.eval()
        limit = self.config.train.max_val_batches
        generator = torch.Generator().manual_seed(self.config.train.seed + 2)
        sums: dict[str, float] = defaultdict(float)
        batches = 0

        for index, batch in enumerate(self.val_loader):
            if limit is not None and index >= limit:
                break
            batch = batch.to(self.device, non_blocking=True)
            cover, secret = split_cover_secret(batch)
            hidden, revealed = self.model(cover, secret, generator=generator)
            losses = self.criterion(hidden, revealed, cover, secret)

            for key, value in losses.as_dict().items():
                sums[f"val_{key}"] += value
            sums["val_psnr_cover"] += official_psnr(hidden.stego, cover)
            sums["val_psnr_secret"] += official_psnr(revealed.recovered_secret, secret)
            batches += 1

        self.model.train()
        if batches == 0:
            raise RuntimeError("The validation loader produced no batches.")
        return {key: value / batches for key, value in sums.items()}

    # ------------------------------------------------------------- full run

    def fit(self, overwrite: bool = False) -> list[dict[str, Any]]:
        train = self.config.train
        self._prepare_output_dir(overwrite)
        self.config.save_json(self.output_dir / CONFIG_FILE)
        self.log(
            f"Training on {self.device} for epochs {self.epoch + 1}..{train.epochs}; "
            f"{len(self.train_loader.dataset)} training images; output: {self.output_dir}"
        )

        try:
            for epoch in range(self.epoch + 1, train.epochs + 1):
                metrics: dict[str, Any] = {
                    "epoch": epoch,
                    "lr": self.optimizer.param_groups[0]["lr"],
                }
                metrics.update(self.train_epoch())
                if epoch % train.val_every == 0:
                    metrics.update(self.validate())

                self.scheduler.step()
                self.epoch = epoch
                self.history.append(metrics)
                self._append_metrics(metrics)
                self.log(_format_metrics(metrics))

                if epoch % train.save_every == 0:
                    self._save(periodic_checkpoint_name(epoch), metrics)
                if train.save_latest:
                    self._save(LATEST_FILE, metrics)
        except BaseException:
            path = self._save(ABORT_FILE, {}, aborted=True)
            self.log(f"Training interrupted; saved {path}")
            raise

        self._save(FINAL_FILE, self.history[-1] if self.history else {})
        return self.history

    # ---------------------------------------------------------------- helpers

    def _save(self, name: str, metrics: dict[str, Any], aborted: bool = False) -> Path:
        return save_checkpoint(
            self.output_dir / name,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.epoch,
            config=self.config.to_dict(),
            loader_generator=self.loader_generator,
            metrics={k: v for k, v in metrics.items() if isinstance(v, (int, float))},
            aborted=aborted,
        )

    def _prepare_output_dir(self, overwrite: bool) -> None:
        existing = list(self.output_dir.glob("*.pt")) if self.output_dir.is_dir() else []
        if existing and not (self._resumed or overwrite):
            raise FileExistsError(
                f"{self.output_dir} already contains checkpoints; resume from one, "
                "choose another output_dir, or pass overwrite=True."
            )
        if overwrite and not self._resumed:
            (self.output_dir / METRICS_FILE).unlink(missing_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _append_metrics(self, metrics: dict[str, Any]) -> None:
        with (self.output_dir / METRICS_FILE).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics) + "\n")


def _reject_pretrained_dir(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    if resolved == PRETRAINED_DIR or PRETRAINED_DIR in resolved.parents:
        raise ValueError(
            f"Refusing to write training output into {PRETRAINED_DIR}, which holds "
            "the pretrained checkpoint."
        )


def _format_metrics(metrics: dict[str, Any]) -> str:
    parts = [f"epoch {metrics['epoch']}", f"lr {metrics['lr']:.3e}"]
    for key, value in metrics.items():
        if key in ("epoch", "lr"):
            continue
        parts.append(f"{key} {value:.4f}")
    return " | ".join(parts)

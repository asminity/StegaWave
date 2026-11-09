"""Training hyperparameters and the full experiment configuration.

TrainConfig defaults follow the official HiNet config.py / train.py:
Adam(lr=10**-4.5, betas=(0.5, 0.999), eps=1e-6, weight_decay=1e-5),
StepLR(step_size=1000, gamma=0.5) stepped once per epoch, 1000 epochs,
validation and checkpointing every 50 epochs, no gradient clipping.
"""

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from model.architecture.config import ArchitectureConfig
from model.data import DataConfig
from model.training.losses import LossConfig


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 1000  # official: epochs = 1000
    learning_rate: float = 10 ** -4.5  # official: log10_lr = -4.5
    betas: tuple[float, float] = (0.5, 0.999)  # official: betas
    eps: float = 1e-6  # official: hard-coded in train.py
    weight_decay: float = 1e-5  # official: weight_decay
    lr_step: int = 1000  # official: weight_step (epochs)
    lr_gamma: float = 0.5  # official: gamma
    val_every: int = 50  # official: val_freq
    save_every: int = 50  # official: SAVE_freq
    grad_clip_norm: float | None = None  # official: no clipping
    seed: int = 0
    device: str = "auto"  # "auto" -> cuda if available, else cpu
    output_dir: Path = Path("outputs/hinet")  # git-ignored by the repo's .gitignore
    max_train_batches: int | None = None  # cap per epoch (smoke tests / debugging)
    max_val_batches: int | None = None
    stop_on_nonfinite: bool = True
    save_latest: bool = True  # write latest.pt every epoch for resuming

    def __post_init__(self) -> None:
        object.__setattr__(self, "betas", tuple(self.betas))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if len(self.betas) != 2 or not all(0 <= b < 1 for b in self.betas):
            raise ValueError("betas must be two values in [0, 1)")
        if self.eps <= 0 or self.weight_decay < 0:
            raise ValueError("eps must be > 0 and weight_decay >= 0")
        if self.lr_step < 1 or self.lr_gamma <= 0:
            raise ValueError("lr_step must be >= 1 and lr_gamma > 0")
        if self.val_every < 1 or self.save_every < 1:
            raise ValueError("val_every and save_every must be >= 1")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be > 0 or None")
        for name in ("max_train_batches", "max_val_batches"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be >= 1 or None")


@dataclass(frozen=True)
class ExperimentConfig:
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(dataclasses.asdict(self))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExperimentConfig":
        sections = {
            "architecture": ArchitectureConfig,
            "loss": LossConfig,
            "data": DataConfig,
            "train": TrainConfig,
        }
        unknown = set(raw) - set(sections)
        if unknown:
            raise ValueError(f"Unknown config sections: {sorted(unknown)}")
        return cls(**{
            name: section_cls(**raw.get(name, {}))
            for name, section_cls in sections.items()
        })

    def save_json(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load_json(cls, path: Path | str) -> "ExperimentConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value

import json

import pytest

from model.architecture.config import ArchitectureConfig
from model.data import DataConfig
from model.training.config import ExperimentConfig, TrainConfig
from model.training.losses import LossConfig


def test_train_config_defaults_are_official():
    c = TrainConfig()
    assert c.learning_rate == pytest.approx(10 ** -4.5)
    assert (c.betas, c.eps, c.weight_decay) == ((0.5, 0.999), 1e-6, 1e-5)
    assert (c.lr_step, c.lr_gamma, c.epochs) == (1000, 0.5, 1000)
    assert (c.val_every, c.save_every, c.grad_clip_norm) == (50, 50, None)


@pytest.mark.parametrize("kwargs", [
    {"epochs": 0}, {"learning_rate": 0}, {"betas": (0.5,)}, {"betas": (1.0, 0.9)},
    {"eps": 0}, {"weight_decay": -1}, {"lr_step": 0}, {"lr_gamma": 0},
    {"val_every": 0}, {"save_every": 0}, {"grad_clip_norm": 0.0},
    {"max_train_batches": 0}, {"max_val_batches": 0},
])
def test_train_config_validation(kwargs):
    with pytest.raises(ValueError):
        TrainConfig(**kwargs)


def test_experiment_config_json_round_trip(tmp_path):
    config = ExperimentConfig(
        architecture=ArchitectureConfig(num_blocks=4, clamp=1.5),
        loss=LossConfig(reconstruction_weight=3.0, reduction="mean"),
        data=DataConfig(train_dir=tmp_path / "train", extensions=("png", "jpg"), crop_size=64),
        train=TrainConfig(epochs=7, betas=(0.9, 0.99), output_dir=tmp_path / "run"),
    )
    path = config.save_json(tmp_path / "config.json")
    assert json.loads(path.read_text())["train"]["output_dir"] == (tmp_path / "run").as_posix()
    assert ExperimentConfig.load_json(path) == config


def test_experiment_config_rejects_unknown_sections():
    with pytest.raises(ValueError, match="Unknown"):
        ExperimentConfig.from_dict({"optimizer": {}})

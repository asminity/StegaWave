import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from model.architecture.config import ArchitectureConfig
from model.cli.train import main
from model.data import DataConfig
from model.training.checkpoint import load_checkpoint, load_trained_model
from model.training.config import ExperimentConfig, TrainConfig
from model.training.losses import LossOutput
from model.training.trainer import PRETRAINED_DIR, Trainer


@pytest.fixture
def dataset(tmp_path) -> tuple[Path, Path]:
    rng = np.random.default_rng(0)
    dirs = []
    for split, count in (("train", 8), ("val", 4)):
        root = tmp_path / split
        root.mkdir()
        for i in range(count):
            array = rng.integers(0, 256, (24, 24, 3), dtype=np.uint8)
            Image.fromarray(array).save(root / f"{i}.png")
        dirs.append(root)
    return dirs[0], dirs[1]


def make_config(dataset, output_dir: Path, **train_kwargs) -> ExperimentConfig:
    train_dir, val_dir = dataset
    defaults = {
        "epochs": 2, "learning_rate": 1e-3, "val_every": 1, "save_every": 1,
        "output_dir": output_dir, "device": "cpu", "seed": 0,
    }
    defaults.update(train_kwargs)
    return ExperimentConfig(
        architecture=ArchitectureConfig(num_blocks=2),
        data=DataConfig(
            train_dir=train_dir, val_dir=val_dir,
            crop_size=16, val_crop_size=16, batch_size=4, val_batch_size=2,
        ),
        train=TrainConfig(**defaults),
    )


def quiet(_: str) -> None:
    pass


def test_fit_writes_outputs_and_learns(dataset, tmp_path):
    trainer = Trainer(make_config(dataset, tmp_path / "run"), log=quiet)
    initial = [p.detach().clone() for p in trainer.model.parameters()]
    history = trainer.fit()

    assert [m["epoch"] for m in history] == [1, 2]
    for metrics in history:
        assert np.isfinite(metrics["train_total"])
        assert {"val_psnr_cover", "val_psnr_secret", "val_total"} <= set(metrics)

    run = tmp_path / "run"
    for name in ("config.json", "metrics.jsonl", "latest.pt", "model.pt",
                 "model_checkpoint_00001.pt", "model_checkpoint_00002.pt"):
        assert (run / name).is_file(), name
    lines = (run / "metrics.jsonl").read_text().splitlines()
    assert [json.loads(line)["epoch"] for line in lines] == [1, 2]
    assert ExperimentConfig.load_json(run / "config.json") == trainer.config
    assert load_checkpoint(run / "model.pt")["epoch"] == 2

    changed = [not torch.equal(a, b) for a, b in zip(initial, trainer.model.parameters())]
    assert any(changed)


def test_learning_rate_schedule_steps_per_epoch(dataset, tmp_path):
    config = make_config(dataset, tmp_path / "run", epochs=3, lr_step=1, lr_gamma=0.5)
    history = Trainer(config, log=quiet).fit()
    assert [m["lr"] for m in history] == pytest.approx([1e-3, 5e-4, 2.5e-4])


def test_official_schedule_never_decays_within_default_run(dataset, tmp_path):
    trainer = Trainer(make_config(dataset, tmp_path / "run", epochs=2, lr_step=1000), log=quiet)
    trainer.fit()
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)


def test_resume_is_bit_exact(dataset, tmp_path):
    full = Trainer(make_config(dataset, tmp_path / "full", epochs=3), log=quiet)
    full.fit()

    part = Trainer(make_config(dataset, tmp_path / "part", epochs=1), log=quiet)
    part.fit()
    resumed = Trainer(make_config(dataset, tmp_path / "part", epochs=3), log=quiet)
    resumed.resume(tmp_path / "part" / "latest.pt")
    assert resumed.epoch == 1
    history = resumed.fit()

    assert [m["epoch"] for m in history] == [2, 3]
    for a, b in zip(full.model.parameters(), resumed.model.parameters(), strict=True):
        assert torch.equal(a, b)
    assert history[-1]["train_total"] == full.history[-1]["train_total"]
    assert history[-1]["val_psnr_secret"] == full.history[-1]["val_psnr_secret"]


def test_same_seed_same_result_different_seed_differs(dataset, tmp_path):
    def final_params(seed, name):
        trainer = Trainer(make_config(dataset, tmp_path / name, epochs=1, seed=seed), log=quiet)
        trainer.fit()
        return torch.cat([p.flatten() for p in trainer.model.parameters()])

    a, b, c = final_params(0, "a"), final_params(0, "b"), final_params(1, "c")
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_resume_rejects_architecture_mismatch(dataset, tmp_path):
    Trainer(make_config(dataset, tmp_path / "run", epochs=1), log=quiet).fit()
    config = make_config(dataset, tmp_path / "other", epochs=2)
    config = ExperimentConfig(
        architecture=ArchitectureConfig(num_blocks=3), loss=config.loss,
        data=config.data, train=config.train,
    )
    with pytest.raises(ValueError, match="Architecture"):
        Trainer(config, log=quiet).resume(tmp_path / "run" / "latest.pt")


def test_refuses_existing_run_without_overwrite(dataset, tmp_path):
    Trainer(make_config(dataset, tmp_path / "run", epochs=1), log=quiet).fit()
    with pytest.raises(FileExistsError):
        Trainer(make_config(dataset, tmp_path / "run", epochs=1), log=quiet).fit()
    Trainer(make_config(dataset, tmp_path / "run", epochs=1), log=quiet).fit(overwrite=True)
    lines = (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 1


def test_refuses_pretrained_checkpoint_dir(dataset):
    with pytest.raises(ValueError, match="pretrained"):
        Trainer(make_config(dataset, PRETRAINED_DIR), log=quiet)
    with pytest.raises(ValueError, match="pretrained"):
        Trainer(make_config(dataset, PRETRAINED_DIR / "sub"), log=quiet)


def test_nonfinite_loss_aborts_with_checkpoint(dataset, tmp_path):
    trainer = Trainer(make_config(dataset, tmp_path / "run"), log=quiet)
    nan = torch.tensor(float("nan"))
    trainer.criterion = lambda *args: LossOutput(nan, nan, nan, nan)
    with pytest.raises(FloatingPointError):
        trainer.fit()
    checkpoint = load_checkpoint(tmp_path / "run" / "model_ABORT.pt")
    assert checkpoint["aborted"] is True and checkpoint["epoch"] == 0


def test_max_batches_and_no_validation(dataset, tmp_path):
    train_dir, _ = dataset
    config = make_config(dataset, tmp_path / "run", epochs=1, max_train_batches=1)
    config = ExperimentConfig(
        architecture=config.architecture, loss=config.loss,
        data=DataConfig(train_dir=train_dir, crop_size=16, batch_size=2),
        train=config.train,
    )
    trainer = Trainer(config, log=quiet)
    criterion = trainer.criterion
    calls = []

    def counting(*args):
        calls.append(1)
        return criterion(*args)

    trainer.criterion = counting
    history = trainer.fit()
    assert len(trainer.train_loader) == 4  # 8 images / batch 2
    assert len(calls) == 1  # capped by max_train_batches
    assert "val_psnr_cover" not in history[0]
    assert trainer.val_loader is None


def test_load_trained_model_for_inference(dataset, tmp_path):
    trainer = Trainer(make_config(dataset, tmp_path / "run", epochs=1), log=quiet)
    trainer.fit()
    model = load_trained_model(tmp_path / "run" / "model.pt")
    assert not model.training
    cover, secret = torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16)
    with torch.no_grad():
        expected = trainer.model.eval().hide(cover, secret).stego
        torch.testing.assert_close(model.hide(cover, secret).stego, expected)


def test_cli_train_then_resume(dataset, tmp_path):
    train_dir, val_dir = dataset
    config = make_config(dataset, tmp_path / "unused", epochs=1)
    config_path = config.save_json(tmp_path / "config.json")
    out = tmp_path / "cli_run"
    trainer = main([
        "--config", str(config_path), "--output-dir", str(out),
        "--train-dir", str(train_dir), "--val-dir", str(val_dir),
    ])
    assert trainer.epoch == 1 and (out / "latest.pt").is_file()
    assert trainer.config.train.output_dir == out

    resumed = main(["--resume", str(out / "latest.pt"), "--epochs", "2"])
    assert resumed.epoch == 2
    assert resumed.config.data.crop_size == 16  # restored from the checkpoint's config


def test_cli_requires_train_dir(tmp_path):
    with pytest.raises(SystemExit):
        main(["--output-dir", str(tmp_path / "x")])

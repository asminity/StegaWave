import random

import numpy as np
import pytest
import torch

from model.architecture.config import ArchitectureConfig
from model.architecture.steganography import HiNetSteganography
from model.training.checkpoint import (
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from model.training.config import ExperimentConfig


def test_rng_capture_and_restore():
    torch.manual_seed(3)
    random.seed(3)
    np.random.seed(3)
    state = capture_rng_state()
    expected = (torch.rand(4), random.random(), np.random.rand(3))
    torch.rand(10)
    random.random()
    np.random.rand(5)
    restore_rng_state(state)
    assert torch.equal(torch.rand(4), expected[0])
    assert random.random() == expected[1]
    assert np.array_equal(np.random.rand(3), expected[2])


def _parts():
    model = HiNetSteganography(ArchitectureConfig(num_blocks=1))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    return model, optimizer, scheduler


def test_checkpoint_round_trip_with_weights_only(tmp_path):
    model, optimizer, scheduler = _parts()
    generator = torch.Generator().manual_seed(5)
    path = save_checkpoint(
        tmp_path / "ckpt.pt", model=model, optimizer=optimizer, scheduler=scheduler,
        epoch=3, config=ExperimentConfig().to_dict(), loader_generator=generator,
        metrics={"train_total": 1.5},
    )
    assert not (tmp_path / "ckpt.pt.tmp").exists()
    checkpoint = load_checkpoint(path)  # torch.load(weights_only=True) inside
    assert checkpoint["epoch"] == 3 and checkpoint["aborted"] is False
    assert checkpoint["metrics"] == {"train_total": 1.5}
    assert torch.equal(checkpoint["loader_rng"], generator.get_state())
    fresh, *_ = _parts()
    fresh.load_state_dict(checkpoint["net"])
    for a, b in zip(model.parameters(), fresh.parameters(), strict=True):
        assert torch.equal(a, b)


def test_load_checkpoint_rejects_foreign_files(tmp_path):
    torch.save({"net": {}}, tmp_path / "foreign.pt")
    with pytest.raises(ValueError, match="not a stegawave-hinet-v1"):
        load_checkpoint(tmp_path / "foreign.pt")
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing.pt")

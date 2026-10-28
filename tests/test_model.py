from pathlib import Path

import pytest
import torch
from torch import nn

from stegawave.models.model import Model


@pytest.fixture
def model():
    net = Model()
    net.eval()
    return net


def test_model_forward_cover_secret(model):
    cover = torch.rand(1, 3, 64, 64)
    secret = torch.rand(1, 3, 64, 64)

    with torch.no_grad():
        stego, recovered = model(cover, secret)

    assert stego.shape == cover.shape
    assert recovered.shape == secret.shape


def test_model_single_input_wavelet(model):
    wavelet_input = torch.randn(1, 24, 16, 16)

    with torch.no_grad():
        hidden = model(wavelet_input)
        recovered = model(hidden, rev=True)

    assert hidden.shape == wavelet_input.shape
    assert recovered.shape == wavelet_input.shape


def test_model_conceal_and_reveal(model):
    cover = torch.rand(1, 3, 64, 64)
    secret = torch.rand(1, 3, 64, 64)

    with torch.no_grad():
        stego, lost_secret = model.conceal(cover, secret)
        recovered = model.reveal(stego)

    assert stego.shape == cover.shape
    assert lost_secret.shape == (1, 12, 32, 32)
    assert recovered.shape == secret.shape


def test_model_dataparallel_checkpoint_compatibility(model):
    checkpoint_path = Path(__file__).resolve().parents[1] / "checkpoints" / "model.pt"
    if not checkpoint_path.is_file():
        pytest.skip("Checkpoint file not found.")

    dp_model = nn.DataParallel(model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = {
        key: value
        for key, value in checkpoint["net"].items()
        if "tmp_var" not in key
    }
    dp_model.load_state_dict(state_dict)

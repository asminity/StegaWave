from pathlib import Path

import torch
from torch import nn

from stegawave.models.model import Model


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "model.pt"
DEVICE = torch.device("cpu")


def main():
    # Keep the "module.model..." parameter names used by the checkpoint.
    net = nn.DataParallel(Model()).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )

    state_dict = {
        key: value
        for key, value in checkpoint["net"].items()
        if "tmp_var" not in key
    }

    net.load_state_dict(state_dict)
    net.eval()

    # A single pair of RGB images after DWT produces 24 channels.
    wavelet_input = torch.randn(1, 24, 32, 32, device=DEVICE)

    with torch.inference_mode():
        hidden = net(wavelet_input)
        recovered = net(hidden, rev=True)

    print(f"Device: {DEVICE}")
    print(f"Input shape: {tuple(wavelet_input.shape)}")
    print(f"Forward output shape: {tuple(hidden.shape)}")
    print(f"Reverse output shape: {tuple(recovered.shape)}")
    print("CPU forward and reverse passes completed.")


if __name__ == "__main__":
    main()

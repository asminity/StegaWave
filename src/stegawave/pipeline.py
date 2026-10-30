"""Core pipeline functions for StegaWAVE inference and model verification."""

from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision.transforms import ToPILImage

from stegawave.data.preprocessing import (
    DATA_DIR,
    TARGET_SIZE,
    TRANSFORMED_DIR,
    load_and_preprocess_image,
)
from stegawave.models.model import Model

to_image = ToPILImage()
DEFAULT_DEVICE = torch.device("cpu")
CHECKPOINT_PATH = Path(__file__).resolve().parents[2] / "checkpoints" / "model.pt"


def load_image_tensor(
    path: Path | str,
    save_path: Path | str | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Load an RGB image, resize it to TARGET_SIZE, optionally save transformed copy, and return tensor."""
    if device is None:
        device = DEFAULT_DEVICE

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Image not found: {path}\n"
            "Place the image in the data/samples folder or provide a valid path."
        )

    with Image.open(path) as raw_img:
        original_size = raw_img.size

    tensor = load_and_preprocess_image(path, save_path=save_path).to(device)

    print(
        f"Loaded {path.name}: "
        f"{original_size[0]}x{original_size[1]} -> "
        f"{TARGET_SIZE[0]}x{TARGET_SIZE[1]}"
    )
    if save_path:
        print(f"Saved transformed image to: {save_path}")

    return tensor


def save_image_tensor(tensor: torch.Tensor, path: Path | str) -> None:
    """Save a single RGB image tensor (1, 3, H, W)."""
    path = Path(path)
    tensor = tensor.detach().cpu()

    if tensor.ndim != 4 or tensor.shape[0] != 1:
        raise ValueError(
            f"Expected tensor shape (1, C, H, W); received {tuple(tensor.shape)}"
        )

    if tensor.shape[1] != 3:
        raise ValueError(
            f"Expected an RGB tensor with 3 channels; received {tensor.shape[1]} channels."
        )

    tensor = tensor.squeeze(0).clamp(0, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    to_image(tensor).save(path, quality=95)


def load_model_checkpoint(
    checkpoint_path: Path | str | None = None,
    device: torch.device | None = None,
) -> Model:
    """Initialize Model and load its pretrained weights."""
    if device is None:
        device = DEFAULT_DEVICE
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_PATH

    chk_path = Path(checkpoint_path)
    if not chk_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {chk_path}")

    checkpoint = torch.load(
        chk_path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict) or "net" not in checkpoint:
        raise ValueError("Expected a checkpoint dictionary containing the 'net' key.")

    state_dict = checkpoint["net"]
    model = Model().to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print(f"Loaded checkpoint: {chk_path}")
    return model


def run_inference(
    cover_path: Path | str,
    secret_path: Path | str,
    checkpoint_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    save_transformed: bool = True,
    device: str = "cpu",
) -> tuple[Path, Path]:
    """Execute end-to-end steganography concealing and revelation."""
    cover_path = Path(cover_path)
    secret_path = Path(secret_path)
    output_dir = Path(output_dir) if output_dir is not None else DATA_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)

    transformed_cover = TRANSFORMED_DIR / "cover" / cover_path.name if save_transformed else None
    transformed_secret = TRANSFORMED_DIR / "secret" / secret_path.name if save_transformed else None

    print("Loading and preprocessing input images...")
    cover_tensor = load_image_tensor(cover_path, save_path=transformed_cover, device=dev)
    secret_tensor = load_image_tensor(secret_path, save_path=transformed_secret, device=dev)

    if cover_tensor.shape != secret_tensor.shape:
        raise ValueError(
            f"Dimension mismatch after preprocessing: {tuple(cover_tensor.shape)} vs {tuple(secret_tensor.shape)}"
        )

    print("Loading pretrained HiNet model...")
    model = load_model_checkpoint(checkpoint_path, device=dev)

    print("Running concealment and recovery...")
    with torch.inference_mode():
        stego, recovered_secret = model(cover_tensor, secret_tensor)

    stego_out = output_dir / "stego.jpeg"
    recovered_out = output_dir / "recovered_secret.jpeg"

    save_image_tensor(stego, stego_out)
    save_image_tensor(recovered_secret, recovered_out)

    print("\nInference completed successfully.")
    print("Stego image:", stego_out)
    print("Recovered secret:", recovered_out)
    return stego_out, recovered_out


def verify_checkpoint(
    checkpoint_path: Path | str | None = None,
    device: str = "cpu",
) -> bool:
    """Validate checkpoint loading and run dry forward/reverse passes."""
    dev = torch.device(device)
    chk_path = Path(checkpoint_path) if checkpoint_path else CHECKPOINT_PATH

    if not chk_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {chk_path}")

    print(f"Loading checkpoint from: {chk_path}")
    net = nn.DataParallel(Model()).to(dev)

    checkpoint = torch.load(
        chk_path,
        map_location=dev,
        weights_only=False,
    )

    state_dict = {
        key: value
        for key, value in checkpoint["net"].items()
        if "tmp_var" not in key
    }

    net.load_state_dict(state_dict)
    net.eval()

    wavelet_input = torch.randn(1, 24, 32, 32, device=dev)

    with torch.inference_mode():
        hidden = net(wavelet_input)
        recovered = net(hidden, rev=True)

    print(f"Device: {dev}")
    print(f"Input shape: {tuple(wavelet_input.shape)}")
    print(f"Forward output shape: {tuple(hidden.shape)}")
    print(f"Reverse output shape: {tuple(recovered.shape)}")
    print("Checkpoint verified successfully: CPU forward and reverse passes completed.")
    return True

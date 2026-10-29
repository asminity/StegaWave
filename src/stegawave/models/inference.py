import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import ToPILImage

# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stegawave.data.preprocessing import (
    DATA_DIR,
    TARGET_SIZE,
    TRANSFORMED_DIR,
    load_and_preprocess_image,
)
from stegawave.models.model import Model

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "model.pt"


def get_image_path(folder: Path, stem: str, default_ext: str = ".jpg") -> Path:
    """Find an image matching stem with common image extensions."""
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = folder / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return folder / f"{stem}{default_ext}"


# Data subfolders merged under data/
COVER1_PATH = get_image_path(DATA_DIR / "samples" / "cover", "cover1", ".jpg")
SECRET1_PATH = get_image_path(DATA_DIR / "samples" / "secret", "secret1", ".jpeg")

TRANSFORMED_COVER1_PATH = TRANSFORMED_DIR / "cover" / COVER1_PATH.name
TRANSFORMED_SECRET1_PATH = TRANSFORMED_DIR / "secret" / SECRET1_PATH.name

OUTPUT_DIR = DATA_DIR / "outputs"

DEVICE = torch.device("cpu")


# ==================================================
# 2. IMAGE PREPROCESSING & UTILITIES
# ==================================================

to_image = ToPILImage()


def load_image(path: Path, save_path: Path | None = None) -> torch.Tensor:
    """Load an RGB image, resize it to TARGET_SIZE, optionally save it, and return a tensor."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Image not found: {path}\n"
            "Place the image in the data/samples folder or "
            "update its path in inference.py."
        )

    with Image.open(path) as raw_img:
        original_size = raw_img.size

    tensor = load_and_preprocess_image(path, save_path=save_path).to(DEVICE)

    print(
        f"Loaded {path.name}: "
        f"{original_size[0]}x{original_size[1]} -> "
        f"{TARGET_SIZE[0]}x{TARGET_SIZE[1]}"
    )
    if save_path:
        print(f"Saved transformed image to: {save_path}")

    return tensor


def save_image(tensor: torch.Tensor, path: Path) -> None:
    """Save a single RGB image tensor."""
    tensor = tensor.detach().cpu()

    if tensor.ndim != 4 or tensor.shape[0] != 1:
        raise ValueError(
            f"Expected tensor shape (1, C, H, W); "
            f"received {tuple(tensor.shape)}"
        )

    if tensor.shape[1] != 3:
        raise ValueError(
            f"Expected an RGB tensor with 3 channels; "
            f"received {tensor.shape[1]} channels."
        )

    tensor = tensor.squeeze(0).clamp(0, 1)

    path.parent.mkdir(parents=True, exist_ok=True)
    to_image(tensor).save(path, quality=95)


# ==================================================
# 3. LOAD PRETRAINED HINET
# ==================================================

def load_model() -> torch.nn.Module:
    """Initialize HiNet and load its pretrained parameters."""
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict) or "net" not in checkpoint:
        raise ValueError(
            "Expected a checkpoint dictionary containing the 'net' key."
        )

    state_dict = checkpoint["net"]

    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint['net'] must be a state dictionary.")

    # Remove wrapper prefixes from the checkpoint parameter names.
    cleaned_state_dict = {}

    for key, value in state_dict.items():
        cleaned_key = key

        while True:
            if cleaned_key.startswith("module."):
                cleaned_key = cleaned_key[len("module."):]
            elif cleaned_key.startswith("model."):
                cleaned_key = cleaned_key[len("model."):]
            else:
                break

        if cleaned_key in cleaned_state_dict:
            raise ValueError(
                f"Duplicate parameter after prefix removal: {cleaned_key}"
            )

        cleaned_state_dict[cleaned_key] = value

    model = Model().to(DEVICE)

    # Strict loading ensures every expected parameter matches.
    model.load_state_dict(cleaned_state_dict, strict=True)
    model.eval()

    print(f"Loaded checkpoint: {CHECKPOINT_PATH}")
    print(f"Loaded {len(cleaned_state_dict)} parameter entries successfully.")

    return model


# ==================================================
# 4. CONCEAL AND REVEAL
# ==================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading and preprocessing input images...")

    cover1 = load_image(COVER1_PATH, save_path=TRANSFORMED_COVER1_PATH)
    secret1 = load_image(SECRET1_PATH, save_path=TRANSFORMED_SECRET1_PATH)

    print("Cover1 shape:", tuple(cover1.shape))
    print("Secret1 shape:", tuple(secret1.shape))

    # Both inputs must have identical tensor dimensions.
    if cover1.shape != secret1.shape:
        raise ValueError(
            "Cover1 and secret1 must have matching dimensions after "
            "preprocessing. "
            f"Cover1: {tuple(cover1.shape)}, "
            f"secret1: {tuple(secret1.shape)}"
        )

    print("Input dimensions verified.")
    print("Loading pretrained HiNet model...")

    model = load_model()

    print("Running concealment and recovery...")

    with torch.inference_mode():
        stego, recovered_secret = model(cover1, secret1)

    # Save generated images.
    save_image(
        stego,
        OUTPUT_DIR / "stego.jpeg",
    )

    save_image(
        recovered_secret,
        OUTPUT_DIR / "recovered_secret.jpeg",
    )

    print("\nInference completed successfully.")
    print("Stego image:", OUTPUT_DIR / "stego.jpeg")
    print("Recovered secret:", OUTPUT_DIR / "recovered_secret.jpeg")


if __name__ == "__main__":
    main()

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stegawave.data.preprocessing import (
    DATA_DIR,
    TRANSFORMED_DIR,
    load_and_preprocess_image,
)


def get_image_path(folder: Path, stem: str, default_ext: str = ".jpg") -> Path:
    """Find an image matching stem with common image extensions."""
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = folder / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return folder / f"{stem}{default_ext}"


# Reference images are taken from data/transformed to match the 256x256 output dimensions
COVER1_SAMPLE = get_image_path(DATA_DIR / "samples" / "cover", "cover1", ".jpg")
SECRET1_SAMPLE = get_image_path(DATA_DIR / "samples" / "secret", "secret1", ".jpeg")

COVER1 = TRANSFORMED_DIR / "cover" / COVER1_SAMPLE.name
SECRET1 = TRANSFORMED_DIR / "secret" / SECRET1_SAMPLE.name
STEGO = DATA_DIR / "outputs" / "stego.jpeg"
RECOVERED = DATA_DIR / "outputs" / "recovered_secret.jpeg"


def ensure_reference_image(transformed_path: Path, sample_path: Path, name: str) -> Path:
    """Ensure the 256x256 transformed reference image exists; generate it from samples if missing."""
    if transformed_path.is_file():
        return transformed_path
    if sample_path.is_file():
        print(f"Transformed {name} reference not found. Generating from {sample_path}...")
        load_and_preprocess_image(sample_path, save_path=transformed_path)
        return transformed_path
    raise FileNotFoundError(
        f"Neither transformed reference ({transformed_path}) nor sample image ({sample_path}) was found."
    )


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def evaluate_pair(
    reference_path: Path,
    result_path: Path,
    name: str,
) -> None:
    if not result_path.is_file():
        raise FileNotFoundError(
            f"Result image not found: {result_path}. Please run inference first."
        )

    reference = load_image(reference_path)
    result = load_image(result_path)

    if reference.shape != result.shape:
        raise ValueError(
            f"Shape mismatch: {reference.shape} vs {result.shape}. "
            "Ensure you are comparing against the transformed (256x256) reference image."
        )

    psnr = peak_signal_noise_ratio(
        reference, result, data_range=255
    )

    ssim = structural_similarity(
        reference,
        result,
        channel_axis=2,
        data_range=255,
    )

    print(f"\n{name}")
    print(f"  PSNR: {psnr:.4f} dB")
    print(f"  SSIM: {ssim:.6f}")


def main() -> None:
    cover1_ref = ensure_reference_image(COVER1, COVER1_SAMPLE, "cover1")
    secret1_ref = ensure_reference_image(SECRET1, SECRET1_SAMPLE, "secret1")

    evaluate_pair(cover1_ref, STEGO, "Cover1 vs Stego")
    evaluate_pair(secret1_ref, RECOVERED, "Secret1 vs Recovered")


if __name__ == "__main__":
    main()

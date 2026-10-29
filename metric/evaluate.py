import sys
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stegawave.data.preprocessing import (
    DATA_DIR,
    load_and_preprocess_image,
)

# Reference images are taken from data/transformed to match the 256x256 output dimensions
COVER_SAMPLE = DATA_DIR / "samples" / "cover" / "cover.jpeg"
SECRET_SAMPLE = DATA_DIR / "samples" / "secret" / "secret.jpeg"

COVER = DATA_DIR / "transformed" / "cover" / "cover.jpeg"
SECRET = DATA_DIR / "transformed" / "secret" / "secret.jpeg"
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
    cover_ref = ensure_reference_image(COVER, COVER_SAMPLE, "cover")
    secret_ref = ensure_reference_image(SECRET, SECRET_SAMPLE, "secret")

    evaluate_pair(cover_ref, STEGO, "Cover vs Stego")
    evaluate_pair(secret_ref, RECOVERED, "Secret vs Recovered")


if __name__ == "__main__":
    main()

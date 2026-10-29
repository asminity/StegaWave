from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


ROOT = Path(__file__).resolve().parents[1]

COVER = ROOT / "samples" / "cover" / "cover.jpeg"
SECRET = ROOT / "samples" / "secret" /"secret.jpeg"
STEGO = ROOT / "outputs" / "stego.jpeg"
RECOVERED = ROOT / "outputs" / "recovered_secret.jpeg"


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def evaluate_pair(
    reference_path: Path,
    result_path: Path,
    name: str,
) -> None:
    reference = load_image(reference_path)
    result = load_image(result_path)

    if reference.shape != result.shape:
        raise ValueError(
            f"Shape mismatch: {reference.shape} vs {result.shape}"
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
    evaluate_pair(COVER, STEGO, "Cover vs Stego")
    evaluate_pair(SECRET, RECOVERED, "Secret vs Recovered")


if __name__ == "__main__":
    main()

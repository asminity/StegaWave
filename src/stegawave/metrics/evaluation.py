"""Evaluation metrics (PSNR and SSIM) and image comparison helpers."""

from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from stegawave.data.preprocessing import (
    load_and_preprocess_image,
)


def find_sample_image(folder: Path, stem: str, default_ext: str = ".jpg") -> Path:
    """Find an image matching stem with common image extensions."""
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = folder / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return folder / f"{stem}{default_ext}"


def ensure_reference_image(transformed_path: Path, sample_path: Path, name: str = "image") -> Path:
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


def load_image_array(path: Path | str) -> np.ndarray:
    """Load an image file into an RGB uint8 numpy array."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def calculate_psnr(reference: np.ndarray, result: np.ndarray) -> float:
    """Calculate Peak Signal-to-Noise Ratio (PSNR) in dB."""
    if reference.shape != result.shape:
        raise ValueError(
            f"Shape mismatch: {reference.shape} vs {result.shape}."
        )
    return float(peak_signal_noise_ratio(reference, result, data_range=255))


def calculate_ssim(reference: np.ndarray, result: np.ndarray) -> float:
    """Calculate Structural Similarity Index (SSIM)."""
    if reference.shape != result.shape:
        raise ValueError(
            f"Shape mismatch: {reference.shape} vs {result.shape}."
        )
    return float(
        structural_similarity(
            reference,
            result,
            channel_axis=2,
            data_range=255,
        )
    )


def evaluate_pair(
    reference_path: Path | str,
    result_path: Path | str,
    name: str = "",
) -> dict[str, float]:
    """
    Evaluate PSNR and SSIM between reference and result image files.

    Returns:
        dict: {'psnr': float, 'ssim': float}
    """
    ref_path = Path(reference_path)
    res_path = Path(result_path)

    if not res_path.is_file():
        raise FileNotFoundError(
            f"Result image not found: {res_path}. Please run inference first."
        )

    reference = load_image_array(ref_path)
    result = load_image_array(res_path)

    psnr = calculate_psnr(reference, result)
    ssim = calculate_ssim(reference, result)

    if name:
        print(f"\n{name}")
        print(f"  PSNR: {psnr:.4f} dB")
        print(f"  SSIM: {ssim:.6f}")

    return {"psnr": psnr, "ssim": ssim}


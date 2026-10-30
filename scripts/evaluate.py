"""
StegaWAVE Evaluation Script.

Evaluates reconstruction fidelity using Peak Signal-to-Noise Ratio (PSNR)
and Structural Similarity Index (SSIM).
"""

import argparse
import sys
from pathlib import Path

# Ensure src is on sys.path if run directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stegawave.data.preprocessing import (
    DATA_DIR,
    TRANSFORMED_DIR,
)
from stegawave.metrics.evaluation import (
    ensure_reference_image,
    evaluate_pair,
    find_sample_image,
)


def parse_args():
    cover1_sample = find_sample_image(DATA_DIR / "samples" / "cover", "cover1", ".jpg")
    secret1_sample = find_sample_image(DATA_DIR / "samples" / "secret", "secret1", ".jpeg")

    default_cover_ref = TRANSFORMED_DIR / "cover" / cover1_sample.name
    default_secret_ref = TRANSFORMED_DIR / "secret" / secret1_sample.name
    default_stego = DATA_DIR / "outputs" / "stego.jpeg"
    default_recovered = DATA_DIR / "outputs" / "recovered_secret.jpeg"

    parser = argparse.ArgumentParser(description="Evaluate StegaWAVE steganography quality (PSNR & SSIM)")
    parser.add_argument(
        "--cover-ref",
        type=Path,
        default=default_cover_ref,
        help="Path to transformed cover reference image",
    )
    parser.add_argument(
        "--cover-sample",
        type=Path,
        default=cover1_sample,
        help="Fallback raw cover sample to generate reference if missing",
    )
    parser.add_argument(
        "--secret-ref",
        type=Path,
        default=default_secret_ref,
        help="Path to transformed secret reference image",
    )
    parser.add_argument(
        "--secret-sample",
        type=Path,
        default=secret1_sample,
        help="Fallback raw secret sample to generate reference if missing",
    )
    parser.add_argument(
        "--stego",
        type=Path,
        default=default_stego,
        help="Path to output stego image",
    )
    parser.add_argument(
        "--recovered",
        type=Path,
        default=default_recovered,
        help="Path to output recovered secret image",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cover_ref = ensure_reference_image(args.cover_ref, args.cover_sample, "cover")
    secret_ref = ensure_reference_image(args.secret_ref, args.secret_sample, "secret")

    evaluate_pair(cover_ref, args.stego, "Cover vs Stego")
    evaluate_pair(secret_ref, args.recovered, "Secret vs Recovered")


if __name__ == "__main__":
    main()


"""
StegaWAVE Inference Script.

Conceals a secret image into a cover image using the pretrained HiNet model
and extracts (reveals) the secret image from the generated stego image.
"""

import argparse
import sys
from pathlib import Path

# Ensure src is on sys.path if run directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stegawave.data.preprocessing import DATA_DIR
from stegawave.metrics.evaluation import find_sample_image
from stegawave.pipeline import run_inference


def parse_args():
    default_cover = find_sample_image(DATA_DIR / "samples" / "cover", "cover1", ".jpg")
    default_secret = find_sample_image(DATA_DIR / "samples" / "secret", "secret1", ".jpeg")
    default_checkpoint = PROJECT_ROOT / "checkpoints" / "model.pt"
    default_output_dir = DATA_DIR / "outputs"

    parser = argparse.ArgumentParser(description="StegaWAVE Image Steganography Inference")
    parser.add_argument(
        "--cover",
        type=Path,
        default=default_cover,
        help="Path to the cover image (default: data/samples/cover/cover1.jpg)",
    )
    parser.add_argument(
        "--secret",
        type=Path,
        default=default_secret,
        help="Path to the secret image (default: data/samples/secret/secret1.jpeg)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=default_checkpoint,
        help="Path to model weights checkpoint (default: checkpoints/model.pt)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory to save output images (default: data/outputs)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run inference on (default: cpu)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_inference(
        cover_path=args.cover,
        secret_path=args.secret,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()

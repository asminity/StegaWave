"""StegaWAVE Command Line Interface."""

import argparse
import sys
from pathlib import Path

from stegawave.data.preprocessing import DATA_DIR, TRANSFORMED_DIR
from stegawave.metrics.evaluation import (
    ensure_reference_image,
    evaluate_pair,
    find_sample_image,
)
from stegawave.pipeline import run_inference, verify_checkpoint


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="stegawave",
        description="StegaWAVE: Invertible Neural Network (HiNet) Wavelet-based Image Steganography",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Subcommand: infer
    infer_parser = subparsers.add_parser("infer", help="Run steganographic inference")
    infer_parser.add_argument("--cover", type=Path, help="Cover image path")
    infer_parser.add_argument("--secret", type=Path, help="Secret image path")
    infer_parser.add_argument("--checkpoint", type=Path, help="Model checkpoint path")
    infer_parser.add_argument("--output-dir", type=Path, help="Output directory")
    infer_parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")

    # Subcommand: evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate PSNR and SSIM metrics")
    eval_parser.add_argument("--cover-ref", type=Path, help="Reference cover image")
    eval_parser.add_argument("--cover-sample", type=Path, help="Raw cover sample if ref missing")
    eval_parser.add_argument("--secret-ref", type=Path, help="Reference secret image")
    eval_parser.add_argument("--secret-sample", type=Path, help="Raw secret sample if ref missing")
    eval_parser.add_argument("--stego", type=Path, help="Output stego image")
    eval_parser.add_argument("--recovered", type=Path, help="Output recovered secret image")

    # Subcommand: check
    check_parser = subparsers.add_parser("check", help="Verify model checkpoint and forward/reverse passes")
    check_parser.add_argument("--checkpoint", type=Path, help="Path to checkpoint model.pt")

    args = parser.parse_args(argv)

    if args.command == "infer":
        default_cover = find_sample_image(DATA_DIR / "samples" / "cover", "cover1", ".jpg")
        default_secret = find_sample_image(DATA_DIR / "samples" / "secret", "secret1", ".jpeg")
        cover = args.cover if args.cover else default_cover
        secret = args.secret if args.secret else default_secret
        run_inference(
            cover_path=cover,
            secret_path=secret,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            device=args.device,
        )
        return 0
    elif args.command == "evaluate":
        default_cover_sample = find_sample_image(DATA_DIR / "samples" / "cover", "cover1", ".jpg")
        default_secret_sample = find_sample_image(DATA_DIR / "samples" / "secret", "secret1", ".jpeg")
        cover_sample = args.cover_sample or default_cover_sample
        secret_sample = args.secret_sample or default_secret_sample

        cover_ref = args.cover_ref or (TRANSFORMED_DIR / "cover" / cover_sample.name)
        secret_ref = args.secret_ref or (TRANSFORMED_DIR / "secret" / secret_sample.name)
        stego = args.stego or (DATA_DIR / "outputs" / "stego.jpeg")
        recovered = args.recovered or (DATA_DIR / "outputs" / "recovered_secret.jpeg")

        c_ref = ensure_reference_image(cover_ref, cover_sample, "cover")
        s_ref = ensure_reference_image(secret_ref, secret_sample, "secret")
        evaluate_pair(c_ref, stego, "Cover vs Stego")
        evaluate_pair(s_ref, recovered, "Secret vs Recovered")
        return 0
    elif args.command == "check":
        verify_checkpoint(args.checkpoint)
        return 0
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())

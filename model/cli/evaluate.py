"""Command-line entry point for evaluating a from-scratch HiNet checkpoint.

Examples (from the repository root):
    python -m model.cli.evaluate --checkpoint outputs/hinet/model.pt --images data/DIV2K_valid_HR
    python -m model.cli.evaluate --checkpoint outputs/hinet/model.pt \
        --cover-dir covers/ --secret-dir secrets/ --crop-size 256 --jpeg-quality 95 75
    python -m model.cli.evaluate --checkpoint ... --images ... --full-image --save-images

Writes results.csv (one row per pair and setting) and summary.json to
--output-dir (default: outputs/hinet_eval, git-ignored).
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

from model.evaluation import (
    BASE_SETTINGS,
    EvalConfig,
    EvalResult,
    evaluate,
    format_summary,
    pairs_from_dirs,
    pairs_from_folder,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained HiNet checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--images", type=Path,
                        help="one folder; consecutive sorted images form (secret, cover) pairs")
    source.add_argument("--cover-dir", type=Path, help="cover folder (use with --secret-dir)")
    parser.add_argument("--secret-dir", type=Path)
    parser.add_argument("--extensions", nargs="+", default=["png"])
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--crop-size", type=int, default=1024, help="centre crop (default 1024)")
    size.add_argument("--full-image", action="store_true",
                      help="no crop; trim to even size (cover and secret must match)")
    parser.add_argument("--small-image-policy", choices=("error", "resize"), default="error")
    parser.add_argument("--settings", nargs="*", choices=BASE_SETTINGS, default=list(BASE_SETTINGS))
    parser.add_argument("--jpeg-quality", type=int, nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/hinet_eval"))
    return parser


def main(argv: Sequence[str] | None = None) -> EvalResult:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.cover_dir is None) != (args.secret_dir is None):
        parser.error("--cover-dir and --secret-dir must be given together")

    if args.images is not None:
        pairs = pairs_from_folder(args.images, args.extensions)
    else:
        pairs = pairs_from_dirs(args.cover_dir, args.secret_dir, args.extensions)

    config = EvalConfig(
        crop_size=None if args.full_image else args.crop_size,
        small_image_policy=args.small_image_policy,
        settings=tuple(args.settings),
        jpeg_qualities=tuple(args.jpeg_quality),
        extensions=tuple(args.extensions),
        seed=args.seed,
        device=args.device,
        save_images=args.save_images,
        limit=args.limit,
    )
    result = evaluate(args.checkpoint, pairs, config, output_dir=args.output_dir)
    print(format_summary(result.summary))
    print(f"\nWrote {', '.join(str(p) for p in result.files.values())}")
    return result


if __name__ == "__main__":
    main()

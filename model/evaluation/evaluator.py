"""Evaluating a trained (from-scratch) HiNet checkpoint on image pairs.

Each pair is hidden once and revealed under every setting with the same seeded
z, so settings differ only in what happens to the stego image. Per setting:
    official_psnr_cover / official_psnr_secret   float, unclipped (official style)
    cover_* / secret_*   per-image uint8 PSNR, SSIM, MAE, RMSE
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from model.architecture.steganography import HiNetSteganography
from model.evaluation.channels import apply_channel, load_image
from model.evaluation.config import EvalConfig
from model.evaluation.metrics import official_psnr, uint8_metrics
from model.evaluation.pairs import ImagePair
from model.evaluation.report import (
    RESULTS_FILE,
    SUMMARY_FILE,
    jsonable,
    save_pair_images,
    summarize,
    write_csv,
)
from model.training.checkpoint import load_checkpoint, load_trained_model
from model.training.runtime import resolve_device


@dataclass
class EvalResult:
    rows: list[dict[str, Any]]
    summary: dict[str, Any]
    output_dir: Path | None = None
    files: dict[str, Path] = field(default_factory=dict)


@torch.no_grad()
def evaluate_pair(
    model: HiNetSteganography,
    cover: Tensor,
    secret: Tensor,
    settings: tuple[str, ...],
    generator: torch.Generator,
) -> dict[str, dict[str, Any]]:
    """Metrics and images for one pair under each setting."""
    if cover.shape != secret.shape:
        raise ValueError(
            f"cover {tuple(cover.shape)} and secret {tuple(secret.shape)} differ in size; "
            "use a crop_size so both are cropped to the same size."
        )
    hidden = model.hide(cover, secret)
    z = model.sample_z(hidden.stego_wavelet, generator)

    results: dict[str, dict[str, Any]] = {}
    for setting in settings:
        if setting == "float":
            stego = hidden.stego
            recovered = model.reveal(hidden.stego_wavelet, z=z).recovered_secret
        else:
            stego = apply_channel(hidden.stego, setting)
            recovered = model.reveal_image(stego, z=z).recovered_secret

        metrics: dict[str, Any] = {
            "official_psnr_cover": official_psnr(stego, cover),
            "official_psnr_secret": official_psnr(recovered, secret),
        }
        metrics.update({f"cover_{k}": v for k, v in uint8_metrics(stego, cover).items()})
        metrics.update({f"secret_{k}": v for k, v in uint8_metrics(recovered, secret).items()})
        results[setting] = {"metrics": metrics, "stego": stego, "recovered": recovered}
    return results


def evaluate(
    checkpoint: Path | str,
    pairs: list[ImagePair],
    config: EvalConfig | None = None,
    output_dir: Path | str | None = None,
) -> EvalResult:
    config = config or EvalConfig()
    if not pairs:
        raise ValueError("No image pairs to evaluate.")
    if config.limit is not None:
        pairs = pairs[: config.limit]

    device = resolve_device(config.device)
    meta = load_checkpoint(checkpoint)
    model = load_trained_model(checkpoint, map_location=device)
    generator = torch.Generator().manual_seed(config.seed)
    settings = config.all_settings
    out = Path(output_dir) if output_dir is not None else None

    rows: list[dict[str, Any]] = []
    for pair in pairs:
        cover = load_image(pair.cover, config.crop_size, config.small_image_policy).to(device)
        secret = load_image(pair.secret, config.crop_size, config.small_image_policy).to(device)
        results = evaluate_pair(model, cover, secret, settings, generator)
        for setting, result in results.items():
            rows.append({
                "pair": pair.name, "setting": setting,
                "cover": pair.cover.as_posix(), "secret": pair.secret.as_posix(),
                **result["metrics"],
            })
        if out is not None and config.save_images:
            save_pair_images(out / "images" / pair.name, cover, secret, results)

    summary = {
        "checkpoint": Path(checkpoint).as_posix(),
        "checkpoint_epoch": int(meta["epoch"]),
        "num_pairs": len(pairs),
        "config": jsonable(asdict(config)),
        "settings": summarize(rows, settings),
    }

    files: dict[str, Path] = {}
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        files["results"] = write_csv(out / RESULTS_FILE, rows)
        files["summary"] = out / SUMMARY_FILE
        files["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return EvalResult(rows=rows, summary=summary, output_dir=out, files=files)

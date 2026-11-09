"""Summaries and files written by an evaluation run."""

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from torch import Tensor

from model.evaluation.metrics import to_uint8

RESULTS_FILE = "results.csv"
SUMMARY_FILE = "summary.json"
_ROW_INFO = ("pair", "setting", "cover", "secret")


def summarize(rows: list[dict[str, Any]], settings: tuple[str, ...]) -> dict[str, Any]:
    """Mean and standard deviation of every metric, per setting."""
    summary: dict[str, Any] = {}
    for setting in settings:
        selected = [row for row in rows if row["setting"] == setting]
        keys = [k for k in selected[0] if k not in _ROW_INFO]
        summary[setting] = {
            key: {
                "mean": float(np.mean([row[key] for row in selected])),
                "std": float(np.std([row[key] for row in selected])),
            }
            for key in keys
        }
    return summary


def save_pair_images(
    folder: Path, cover: Tensor, secret: Tensor, results: dict[str, dict[str, Any]]
) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(cover)).save(folder / "cover.png")
    Image.fromarray(to_uint8(secret)).save(folder / "secret.png")
    for setting, result in results.items():
        Image.fromarray(to_uint8(result["stego"])).save(folder / f"stego_{setting}.png")
        Image.fromarray(to_uint8(result["recovered"])).save(folder / f"recovered_{setting}.png")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def format_summary(summary: dict[str, Any]) -> str:
    columns = ("official_psnr_cover", "official_psnr_secret", "cover_psnr", "cover_ssim",
               "secret_psnr", "secret_ssim")
    header = f"{'setting':<12}" + "".join(f"{c:>22}" for c in columns)
    lines = [
        f"{summary['num_pairs']} pairs, checkpoint epoch {summary['checkpoint_epoch']}",
        header,
    ]
    for setting, stats in summary["settings"].items():
        lines.append(f"{setting:<12}" + "".join(
            f"{stats[c]['mean']:>14.4f} ±{stats[c]['std']:>6.3f}" for c in columns
        ))
    return "\n".join(lines)

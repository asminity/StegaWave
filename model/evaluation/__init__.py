"""Research evaluation of trained (from-scratch) HiNet checkpoints.

Each cover/secret pair is hidden once and revealed under several settings
(float / quantized / jpeg<q>, see channels.py). Images are only JPEG-encoded
when a jpeg setting asks for it; saved images are PNG.
"""

from model.evaluation.channels import (
    apply_channel,
    jpeg_round_trip,
    load_image,
    quantize_8bit,
)
from model.evaluation.config import BASE_SETTINGS, EvalConfig
from model.evaluation.evaluator import EvalResult, evaluate, evaluate_pair
from model.evaluation.metrics import PSNR_CAP, official_psnr, to_uint8, uint8_metrics
from model.evaluation.pairs import ImagePair, pairs_from_dirs, pairs_from_folder
from model.evaluation.report import (
    RESULTS_FILE,
    SUMMARY_FILE,
    format_summary,
    summarize,
)

__all__ = [
    "BASE_SETTINGS",
    "PSNR_CAP",
    "RESULTS_FILE",
    "SUMMARY_FILE",
    "EvalConfig",
    "EvalResult",
    "ImagePair",
    "apply_channel",
    "evaluate",
    "evaluate_pair",
    "format_summary",
    "jpeg_round_trip",
    "load_image",
    "official_psnr",
    "pairs_from_dirs",
    "pairs_from_folder",
    "quantize_8bit",
    "summarize",
    "to_uint8",
    "uint8_metrics",
]

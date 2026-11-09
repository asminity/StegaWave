"""From-scratch HiNet (invertible wavelet image steganography) research rebuild.

Kept separate from the inference package in src/stegawave: it shares no code
with it and never loads the pretrained checkpoints/model.pt.

Subpackages:
    architecture  wavelets, subnet, coupling blocks, HiNet, hide/reveal model
    data          image folders, transforms, loaders, cover/secret split
    training      losses, experiment config, checkpoints, trainer
    evaluation    metrics, channel settings, evaluator, reports
    cli           `python -m model.cli.train` / `python -m model.cli.evaluate`
"""

from model.architecture import (
    AffineCoupling,
    ArchitectureConfig,
    DenseSubnet,
    HaarDWT,
    HaarIWT,
    HidingOutput,
    HiNet,
    HiNetSteganography,
    RevealOutput,
    haar_dwt,
    haar_iwt,
    init_hinet_,
    low_frequency,
)
from model.data import (
    DataConfig,
    ImageFolderDataset,
    build_train_loader,
    build_val_loader,
    split_cover_secret,
    train_transform,
    val_transform,
)
from model.evaluation import (
    EvalConfig,
    EvalResult,
    evaluate,
    official_psnr,
    uint8_metrics,
)
from model.training import (
    ExperimentConfig,
    HiNetLoss,
    LossConfig,
    LossOutput,
    TrainConfig,
    Trainer,
    guide_loss,
    hinet_losses,
    load_checkpoint,
    load_trained_model,
    low_frequency_loss,
    reconstruction_loss,
    save_checkpoint,
)

__all__ = [
    "AffineCoupling",
    "ArchitectureConfig",
    "DataConfig",
    "DenseSubnet",
    "EvalConfig",
    "EvalResult",
    "ExperimentConfig",
    "HaarDWT",
    "HaarIWT",
    "HiNet",
    "HiNetLoss",
    "HiNetSteganography",
    "HidingOutput",
    "ImageFolderDataset",
    "LossConfig",
    "LossOutput",
    "RevealOutput",
    "TrainConfig",
    "Trainer",
    "build_train_loader",
    "build_val_loader",
    "evaluate",
    "guide_loss",
    "haar_dwt",
    "haar_iwt",
    "hinet_losses",
    "init_hinet_",
    "load_checkpoint",
    "load_trained_model",
    "low_frequency",
    "low_frequency_loss",
    "official_psnr",
    "reconstruction_loss",
    "save_checkpoint",
    "split_cover_secret",
    "train_transform",
    "uint8_metrics",
    "val_transform",
]

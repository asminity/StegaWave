---
title: StegaWave
emoji: 🌊
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# StegaWAVE

**StegaWAVE** is an Invertible Neural Network (HiNet) image steganography project leveraging Discrete Wavelet Transforms (DWT) to conceal full-resolution secret images inside cover images and faithfully recover them with minimal perceptual distortion.

The repository has two independent parts:

| Part | Folder | Weights | Purpose |
| :--- | :--- | :--- | :--- |
| **Inference package** | `src/stegawave/`, `scripts/`, `app.py` | Pretrained `checkpoints/model.pt` | Hide and reveal images today; powers the Gradio demo. |
| **From-scratch rebuild** | `model/` | Trained by you (none shipped) | Research reimplementation of HiNet that trains from random initialization. |

The two parts share no code. The inference package and the Gradio app keep using the pretrained `checkpoints/model.pt`, and `model/` never loads it. Checkpoints produced by `model/` use a different format and are not yet usable by `src/stegawave` or the app.

---

## Directory Structure

```text
StegaWAVE/
├── checkpoints/              # Pretrained model weights
│   └── model.pt              # Trained HiNet checkpoint weights
├── data/                     # Data assets (inputs, cached transformations, and outputs)
│   ├── samples/              # Original sample images (tracked in git)
│   │   ├── cover/            # Cover image samples (cover1.jpg, ...)
│   │   └── secret/           # Secret image samples (secret1.jpeg, ...)
│   ├── transformed/          # Preprocessed/resized (256x256) cached images (git-ignored)
│   │   ├── cover/
│   │   └── secret/
│   └── outputs/              # Generated stego and recovered secret images (git-ignored)
│       ├── stego.jpeg
│       └── recovered_secret.jpeg
├── scripts/                  # Standalone executable scripts & pipelines
│   ├── check_model.py        # Validates model checkpoint & runs forward/reverse passes
│   ├── evaluate.py           # Computes PSNR and SSIM fidelity metrics on outputs
│   └── run_inference.py      # End-to-end steganography concealing and revelation pipeline
├── model/                    # From-scratch HiNet rebuild (independent of src/)
│   ├── architecture/         # Haar DWT/IWT, dense subnet, coupling block, HiNet, hide/reveal model
│   ├── data/                 # Image folders, train/val transforms, loaders, cover/secret split
│   ├── training/             # Losses, experiment config, checkpoints, trainer
│   ├── evaluation/           # Metrics, channel settings (float/quantized/JPEG), evaluator, reports
│   ├── cli/                  # `python -m model.cli.train` / `python -m model.cli.evaluate`
│   └── tests/                # Pytest suite, one folder per subpackage
├── src/
│   └── stegawave/            # Main Python package (inference with the pretrained checkpoint)
│       ├── __init__.py       # Package exports (Model, Hinet, DWT, IWT, etc.)
│       ├── pipeline.py       # High-level pipeline functions for inference & checks
│       ├── data/             # Image dataset loading & preprocessing utilities
│       │   ├── __init__.py
│       │   └── preprocessing.py
│       ├── metrics/          # Quantitative image evaluation metrics (PSNR, SSIM)
│       │   ├── __init__.py
│       │   └── evaluation.py
│       ├── models/           # HiNet neural network architecture definitions
│       │   ├── __init__.py
│       │   ├── dense_block.py# Residual Dense Blocks
│       │   ├── hinet.py      # HiNet invertible backbone
│       │   ├── invblock.py   # Invertible coupling blocks
│       │   └── model.py      # Full DWT + HiNet + IWT steganography model wrapper
│       └── transforms/       # Wavelet transforms
│           ├── __init__.py
│           └── wavelet.py    # 2D Haar DWT and Inverse Wavelet Transform (IWT)
├── tests/                    # Pytest test suite
│   ├── test_metrics.py       # Unit tests for PSNR & SSIM metrics
│   ├── test_model.py         # Unit tests for HiNet model architecture & checkpoint loader
│   └── test_wavelet.py       # Unit tests for DWT / IWT mathematical invertibility
├── .gitignore                # Git ignore rules
├── .python-version           # Python version pin (3.12)
├── pyproject.toml            # Project dependencies & build configuration
├── README.md                 # Project documentation
└── uv.lock                   # Lockfile for reproducible dependencies
```

---

## Where to Find What

| Purpose | Directory / File | Description |
| :--- | :--- | :--- |
| **Model Weights** | `checkpoints/model.pt` | Pretrained HiNet PyTorch checkpoint. |
| **Input Images** | `data/samples/` | Sample cover and secret images. |
| **Output Images** | `data/outputs/` | Generated stego and extracted secret images. |
| **Gradio Web UI** | `app.py` | Interactive demo for Hugging Face Spaces deployment. |
| **Run Inference** | `scripts/run_inference.py` | Conceals secret in cover and saves stego + recovered secret. |
| **Evaluate Quality** | `scripts/evaluate.py` | Computes PSNR and SSIM between reference and generated images. |
| **Check Checkpoint** | `scripts/check_model.py` | Verifies weights loading and tensor dimensions. |
| **Core Architecture**| `src/stegawave/models/` | Neural network modules (`Model`, `Hinet`, `INVBlock`, `ResidualDenseBlockOut`). |
| **Wavelets** | `src/stegawave/transforms/` | Haar Discrete Wavelet Transform (`DWT`) & Inverse (`IWT`). |
| **Metrics Library** | `src/stegawave/metrics/` | Reusable PSNR & SSIM functions and image comparison tools. |
| **Automated Tests** | `tests/` | Pytest suite covering model, metrics, and wavelet operations. |
| **Rebuild: Architecture** | `model/architecture/` | `HiNetSteganography`, `HiNet`, `AffineCoupling`, `DenseSubnet`, Haar wavelets, official initialization. |
| **Rebuild: Training** | `model/training/`, `model/cli/train.py` | Losses, `TrainConfig`/`ExperimentConfig`, resumable checkpoints, `Trainer`. |
| **Rebuild: Evaluation** | `model/evaluation/`, `model/cli/evaluate.py` | Official-style and per-image metrics under float, quantized and JPEG settings. |
| **Rebuild: Tests** | `model/tests/` | Invertibility, initialization, losses, data, training and evaluation tests. |

---

## Quick Start

### 1. Run Inference
Conceal a secret image inside a cover image and extract it:
```bash
uv run python scripts/run_inference.py
```
Or with custom images:
```bash
uv run python scripts/run_inference.py --cover path/to/cover.jpg --secret path/to/secret.jpg
```

### 2. Evaluate Results
Calculate PSNR and SSIM metrics on the outputs:
```bash
uv run python scripts/evaluate.py
```

### 3. Verify Model Checkpoint
Test loading the pretrained checkpoint and running a dry forward/reverse pass:
```bash
uv run python scripts/check_model.py
```

### 4. Run Test Suite
Run all unit tests (both `tests/` and `model/tests/`) using pytest:
```bash
uv run pytest
```
Run only one part:
```bash
uv run pytest tests          # inference package
uv run pytest model/tests    # from-scratch rebuild
```

### 5. Launch Gradio Web UI
Launch the interactive web UI locally (or deploy to Hugging Face Spaces):
```bash
uv run python app.py
```
Open `http://localhost:7860` in your browser.

---

## From-Scratch HiNet Rebuild (`model/`)

`model/` reimplements HiNet so it can be trained from random initialization. The defaults follow the official HiNet code ([TomTomTommi/HiNet](https://github.com/TomTomTommi/HiNet)), not the paper; each value's source is noted in the config classes.

| Setting | Default |
| :--- | :--- |
| Architecture | 16 affine coupling blocks, clamp 2.0, dense subnets (growth 32), 4,050,240 parameters |
| Initialization | Weights and biases ~ N(0, 0.01²); each subnet's final conv set to zero, so every block starts as the identity |
| Losses | 5 × reconstruction + 1 × guide + 1 × low-frequency (LL band), summed MSE |
| Optimizer | Adam, lr 10⁻⁴·⁵, betas (0.5, 0.999), eps 1e-6, weight decay 1e-5; StepLR(1000, 0.5) per epoch |
| Data | PNG folders; train: random flips + 224px random crop, batch 16 (first half secret, second half cover); val: 1024px center crop, batch 2 |
| Schedule | 1000 epochs; validate and checkpoint every 50 epochs |

### Train
Training needs image folders (the official setup uses DIV2K `train_HR` / `valid_HR`, not included here):
```bash
uv run python -m model.cli.train --train-dir path/to/DIV2K_train_HR --val-dir path/to/DIV2K_valid_HR
```
Resume an interrupted run, or run a quick smoke test:
```bash
uv run python -m model.cli.train --resume outputs/hinet/latest.pt
uv run python -m model.cli.train --train-dir path/to/images --epochs 1 --max-train-batches 2
```
Settings are resolved as: defaults < `--config run.json` < the checkpoint's saved config (when resuming) < command-line flags. Run `uv run python -m model.cli.train --help` for all flags.

Each run writes to `outputs/hinet/` (git-ignored):
- `config.json` and `metrics.jsonl` (one line per epoch);
- `model_checkpoint_00050.pt`, … every 50 epochs, `latest.pt` every epoch, and `model.pt` at the end;
- `model_ABORT.pt` if training crashes or is interrupted.

Resuming restores the optimizer, schedule, data order and random-number state, giving bit-identical results to an uninterrupted run. The trainer refuses to write into `checkpoints/` or into a folder that already holds a run (use `--resume` or `--overwrite`).

### Evaluate
```bash
uv run python -m model.cli.evaluate --checkpoint outputs/hinet/model.pt --images path/to/DIV2K_valid_HR
uv run python -m model.cli.evaluate --checkpoint outputs/hinet/model.pt \
    --cover-dir covers/ --secret-dir secrets/ --crop-size 256 --jpeg-quality 95 75 --save-images
```
With `--images`, consecutive sorted images form a (secret, cover) pair. Each pair is hidden once and revealed under each setting with the same seeded noise:

| Setting | Stego image passed to the reveal step |
| :--- | :--- |
| `float` | Raw float output, as in the official `test.py` |
| `quantized` | Rounded to 8 bits, exactly what a PNG file holds |
| `jpeg<q>` | Quantized, then JPEG-compressed at quality `q` (optional robustness check) |

For each setting it reports official-style PSNR (float, unclipped, comparable with official training logs) and per-image 8-bit PSNR, SSIM, MAE and RMSE. Results go to `outputs/hinet_eval/results.csv` and `summary.json`; `--save-images` also writes PNGs.

The evaluator accepts only checkpoints produced by `model/`; the pretrained `checkpoints/model.pt` is rejected.

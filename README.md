# StegaWAVE

**StegaWAVE** is an Invertible Neural Network (HiNet) image steganography project leveraging Discrete Wavelet Transforms (DWT) to conceal full-resolution secret images inside cover images and faithfully recover them with minimal perceptual distortion.

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
├── src/
│   └── stegawave/            # Main Python package
│       ├── __init__.py       # Package exports (Model, Hinet, DWT, IWT, etc.)
│       ├── cli.py            # Command-line interface router (`stegawave` CLI)
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
| **Run Inference** | `scripts/run_inference.py` | Conceals secret in cover and saves stego + recovered secret. |
| **Evaluate Quality** | `scripts/evaluate.py` | Computes PSNR and SSIM between reference and generated images. |
| **Check Checkpoint** | `scripts/check_model.py` | Verifies weights loading and tensor dimensions. |
| **Core Architecture**| `src/stegawave/models/` | Neural network modules (`Model`, `Hinet`, `INVBlock`, `ResidualDenseBlockOut`). |
| **Wavelets** | `src/stegawave/transforms/` | Haar Discrete Wavelet Transform (`DWT`) & Inverse (`IWT`). |
| **Metrics Library** | `src/stegawave/metrics/` | Reusable PSNR & SSIM functions and image comparison tools. |
| **Automated Tests** | `tests/` | Pytest suite covering model, metrics, and wavelet operations. |

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
Run all unit tests using pytest:
```bash
uv run pytest
```

### 5. CLI Interface
You can also use the unified CLI tool:
```bash
uv run stegawave infer
uv run stegawave evaluate
uv run stegawave check
```

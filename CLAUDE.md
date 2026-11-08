# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

StegaWAVE hides a full-resolution secret image inside a cover image using a pretrained HiNet invertible neural network operating in the Haar wavelet domain. Inference only — there is no training code; `checkpoints/model.pt` (Git LFS) is the only source of weights.

## Commands

Uses `uv` with Python 3.12 (pinned `>=3.12,<3.13`).

```bash
uv sync                                         # install deps (incl. dev group: pytest, ruff)
uv run pytest                                   # all tests
uv run pytest tests/test_model.py::test_model_forward_cover_secret   # single test
uv run ruff check .                             # lint
uv run python scripts/run_inference.py [--cover PATH --secret PATH]  # writes data/outputs/{stego,recovered_secret}.jpeg
uv run python scripts/evaluate.py               # PSNR/SSIM on outputs vs. references
uv run python scripts/check_model.py            # checkpoint load + dry forward/reverse pass
uv run python app.py                            # Gradio UI on http://localhost:7860
```

`pyproject.toml` sets `pythonpath = ["src"]` for pytest; scripts and `app.py` insert `src/` into `sys.path` themselves, so they run without installing the package.

## Architecture

Data flow in `Model.forward(cover, secret)` (`src/stegawave/models/model.py`):

1. Cover and secret (each `(B,3,H,W)`, values in `[0,1]`, H/W even) → `DWT` → `(B,12,H/2,W/2)` each (4 Haar subbands × 3 RGB channels, concatenated LL/HL/LH/HH along channels).
2. Concatenate → 24 channels → `Hinet` forward (16 `INVBlock`s, `inv1`..`inv16`).
3. First 12 output channels → `IWT` → stego image. Last 12 channels ("lost secret") are discarded.
4. Recovery: `DWT(stego)` + Gaussian noise in place of the lost channels → `Hinet` reverse (blocks run 16→1) → last 12 channels → `IWT` → recovered secret.

Key invariants when editing models:
- `INVBlock` splits channels 12/12 and is an affine coupling layer (`f`, `r`, `y` are `ResidualDenseBlockOut` subnets; scale is `exp(clamp*2*(sigmoid(s)-0.5))`). Forward and `rev=True` must remain exact inverses — `tests/test_wavelet.py` and `tests/test_model.py` check invertibility.
- Parameter names (`model.inv{1..16}.{f,r,y}...`) must match the checkpoint. `Model.load_state_dict` is overridden to strip `module.` (DataParallel) prefixes, add the `model.` prefix, and drop `tmp_var` keys — keep this when renaming anything.
- Checkpoints are dicts with a `"net"` key and are loaded with `weights_only=False`.
- `Model.forward(x)` with `secret=None` passes a raw 24-channel wavelet tensor straight to `Hinet` (used by `verify_checkpoint`).

Module roles:
- `src/stegawave/pipeline.py` — high-level functions used by scripts (`run_inference`, `verify_checkpoint`, `load_model_checkpoint`), CPU by default.
- `src/stegawave/data/preprocessing.py` — resizes all inputs to `TARGET_SIZE = (256, 256)` and by default caches resized copies under `data/transformed/{cover,secret,misc}/`. `PROJECT_ROOT`/`DATA_DIR` are derived from this file's location.
- `src/stegawave/metrics/evaluation.py` — PSNR/SSIM (scikit-image) plus sample-image lookup helpers used by scripts.
- `app.py` — standalone Gradio app; it duplicates preprocessing/model loading rather than importing `pipeline.py`, so changes to preprocessing or checkpoint loading usually need to be made in both. It loads the model at import time and uses `spaces.GPU` (Hugging Face ZeroGPU) with a local no-op fallback.

## Deployment

Every push to `main` syncs the repo to the Hugging Face Space `asminity12/StegaWave` via `.github/workflows/sync-to-hf.yml`. The YAML front matter at the top of `README.md` is the Space config (must stay valid), and `requirements.txt` is what the Space installs — keep it consistent with `pyproject.toml` dependencies.

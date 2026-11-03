"""StegaWAVE: Hugging Face Spaces Gradio Application with ZeroGPU Support.

Provides an interactive web interface for invertible neural network
wavelet-based image steganography (concealment and extraction).
Compatible with Hugging Face ZeroGPU and local CPU/CUDA environments.
"""

import functools
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# --- ZeroGPU Integration with Local Fallback ---
try:
    import spaces  # type: ignore
except ImportError:
    # Graceful fallback when running locally or in standard CPU environments
    class _SpacesMock:
        @staticmethod
        def GPU(func=None, duration=None):
            if func is not None:
                return func

            def decorator(f):
                return f

            return decorator

    spaces = _SpacesMock()

# Ensure src is accessible in Python path
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stegawave.metrics.evaluation import calculate_psnr, calculate_ssim
from stegawave.models.model import Model

TARGET_SIZE = (256, 256)
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "model.pt"

# Under ZeroGPU, torch.cuda.is_available() is True during startup via CUDA emulation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()


@functools.lru_cache(maxsize=1)
def get_model() -> Model:
    """Load and cache the pretrained HiNet steganography model once at startup."""
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {CHECKPOINT_PATH}")

    model = Model().to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint["net"] if isinstance(checkpoint, dict) and "net" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


# Preload and initialize model at module level per ZeroGPU best practice
MODEL = get_model()


def preprocess_pil(image: Image.Image, device: torch.device | None = None) -> tuple[torch.Tensor, np.ndarray]:
    """Convert input PIL Image to RGB (256x256), a normalized Tensor on active device, and a uint8 array."""
    if device is None:
        device = next(MODEL.parameters()).device

    image = image.convert("RGB")
    if image.size != TARGET_SIZE:
        image = image.resize(TARGET_SIZE, Image.Resampling.BILINEAR)
    np_arr = np.asarray(image, dtype=np.uint8)
    tensor = to_tensor(image).unsqueeze(0).to(device)
    return tensor, np_arr


def tensor_to_outputs(tensor: torch.Tensor) -> tuple[Image.Image, np.ndarray]:
    """Convert a PyTorch tensor (1, 3, H, W) to a PIL Image and a uint8 array."""
    clamped = tensor.detach().cpu().squeeze(0).clamp(0, 1)
    pil_img = to_pil(clamped)
    np_arr = np.asarray(pil_img, dtype=np.uint8)
    return pil_img, np_arr


@spaces.GPU(duration=30)
def full_cycle(
    cover_image: Image.Image | None,
    secret_image: Image.Image | None,
) -> tuple[Image.Image | None, Image.Image | None, str]:
    """Conceal secret in cover, extract secret, and compute evaluation metrics on GPU."""
    if cover_image is None or secret_image is None:
        raise gr.Error("Please provide both a Cover image and a Secret image.")

    try:
        device = next(MODEL.parameters()).device
        cover_tensor, cover_arr = preprocess_pil(cover_image, device=device)
        secret_tensor, secret_arr = preprocess_pil(secret_image, device=device)

        with torch.inference_mode():
            stego_tensor, recovered_tensor = MODEL(cover_tensor, secret_tensor)

        stego_pil, stego_arr = tensor_to_outputs(stego_tensor)
        recovered_pil, recovered_arr = tensor_to_outputs(recovered_tensor)

        cover_psnr = calculate_psnr(cover_arr, stego_arr)
        cover_ssim = calculate_ssim(cover_arr, stego_arr)

        secret_psnr = calculate_psnr(secret_arr, recovered_arr)
        secret_ssim = calculate_ssim(secret_arr, recovered_arr)

        metrics_md = (
            "### 📊 Fidelity & Steganography Quality Report\n\n"
            f"| Comparison | PSNR (dB) | SSIM | Assessment |\n"
            f"| :--- | :--- | :--- | :--- |\n"
            f"| **Cover $\\to$ Stego** | `{cover_psnr:.2f}` | `{cover_ssim:.4f}` | "
            f"{'🟢 Imperceptible distortion' if cover_psnr >= 40 else '🟡 Minor distortion'} |\n"
            f"| **Secret $\\to$ Recovered** | `{secret_psnr:.2f}` | `{secret_ssim:.4f}` | "
            f"{'🟢 Near-lossless recovery' if secret_psnr >= 40 else '🟡 Acceptable recovery'} |\n"
        )

        return stego_pil, recovered_pil, metrics_md

    except Exception as exc:
        raise gr.Error(f"Inference error during full cycle execution: {exc}") from exc


@spaces.GPU(duration=20)
def conceal_only(
    cover_image: Image.Image | None,
    secret_image: Image.Image | None,
) -> tuple[Image.Image | None, str]:
    """Conceal secret inside cover image to generate stego image on GPU."""
    if cover_image is None or secret_image is None:
        raise gr.Error("Please provide both a Cover image and a Secret image.")

    try:
        device = next(MODEL.parameters()).device
        cover_tensor, cover_arr = preprocess_pil(cover_image, device=device)
        secret_tensor, _ = preprocess_pil(secret_image, device=device)

        with torch.inference_mode():
            stego_tensor, _ = MODEL.conceal(cover_tensor, secret_tensor)

        stego_pil, stego_arr = tensor_to_outputs(stego_tensor)
        psnr_val = calculate_psnr(cover_arr, stego_arr)
        ssim_val = calculate_ssim(cover_arr, stego_arr)

        metrics_md = (
            "### 🛡️ Stego Quality (Imperceptibility)\n\n"
            f"- **PSNR**: `{psnr_val:.2f} dB`\n"
            f"- **SSIM**: `{ssim_val:.4f}`\n\n"
            "*(Higher PSNR indicates the stego image looks virtually identical to the original cover)*"
        )
        return stego_pil, metrics_md

    except Exception as exc:
        raise gr.Error(f"Inference error during concealment: {exc}") from exc


@spaces.GPU(duration=20)
def reveal_only(
    stego_image: Image.Image | None,
) -> Image.Image | None:
    """Extract hidden secret image from a stego image on GPU."""
    if stego_image is None:
        raise gr.Error("Please provide a Stego image.")

    try:
        device = next(MODEL.parameters()).device
        stego_tensor, _ = preprocess_pil(stego_image, device=device)

        with torch.inference_mode():
            recovered_tensor = MODEL.reveal(stego_tensor)

        recovered_pil, _ = tensor_to_outputs(recovered_tensor)
        return recovered_pil

    except Exception as exc:
        raise gr.Error(f"Inference error during secret extraction: {exc}") from exc


# Prepare example pairs
sample_cover_dir = PROJECT_ROOT / "data" / "samples" / "cover"
sample_secret_dir = PROJECT_ROOT / "data" / "samples" / "secret"

example_pairs = []
if sample_cover_dir.exists() and sample_secret_dir.exists():
    for i in range(1, 6):
        c_path = sample_cover_dir / f"cover{i}.jpg"
        s_path = sample_secret_dir / f"secret{i}.jpg"
        if not s_path.exists():
            s_path = sample_secret_dir / f"secret{i}.jpeg"
        if c_path.exists() and s_path.exists():
            example_pairs.append([str(c_path), str(s_path)])


# Build Gradio UI
with gr.Blocks(title="StegaWAVE | Invertible Wavelet Steganography") as demo:
    gr.Markdown(
        """
        # 🌊 StegaWAVE: Deep Invertible Wavelet Image Steganography
        **StegaWAVE** utilizes an Invertible Neural Network (**HiNet**) combined with 2D Haar **Discrete Wavelet Transforms (DWT)**
        to conceal full-resolution secret images inside cover images with imperceptible visual changes and near-lossless recovery.
        """
    )

    with gr.Tabs():
        # Tab 1: Full Cycle
        with gr.TabItem("🔄 Full Cycle (Conceal & Recover)"):
            gr.Markdown("Upload both images (or select an example below) to conceal the secret and automatically test recovery.")
            with gr.Row():
                with gr.Column():
                    in_cover = gr.Image(type="pil", label="Cover Image (Host)")
                    in_secret = gr.Image(type="pil", label="Secret Image (Hidden Payload)")
                    btn_cycle = gr.Button("🚀 Conceal & Recover Secret", variant="primary")
                with gr.Column():
                    out_stego = gr.Image(type="pil", label="Generated Stego Image (Looks like Cover)")
                    out_recovered = gr.Image(type="pil", label="Extracted Secret Image")
                    out_metrics = gr.Markdown("Run the model to see PSNR & SSIM metrics.")

            btn_cycle.click(
                fn=full_cycle,
                inputs=[in_cover, in_secret],
                outputs=[out_stego, out_recovered, out_metrics],
            )

            if example_pairs:
                gr.Markdown("### 💡 Click an Example Pair to Test Instantly")
                gr.Examples(
                    examples=example_pairs,
                    inputs=[in_cover, in_secret],
                    outputs=[out_stego, out_recovered, out_metrics],
                    fn=full_cycle,
                    cache_examples=False,
                )

        # Tab 2: Conceal Only
        with gr.TabItem("🔒 Conceal Only (Host + Secret → Stego)"):
            gr.Markdown("Conceal a secret image inside a cover image to create an innocent-looking stego image for download.")
            with gr.Row():
                with gr.Column():
                    c_cover = gr.Image(type="pil", label="Cover Image")
                    c_secret = gr.Image(type="pil", label="Secret Image")
                    btn_conceal = gr.Button("🔒 Generate Stego Image", variant="primary")
                with gr.Column():
                    c_out_stego = gr.Image(type="pil", label="Stego Image")
                    c_out_metrics = gr.Markdown()

            btn_conceal.click(
                fn=conceal_only,
                inputs=[c_cover, c_secret],
                outputs=[c_out_stego, c_out_metrics],
            )

        # Tab 3: Reveal Only
        with gr.TabItem("🔓 Reveal Only (Stego → Recovered Secret)"):
            gr.Markdown("Extract a hidden secret image from a stego image created by StegaWAVE.")
            with gr.Row():
                with gr.Column():
                    r_stego = gr.Image(type="pil", label="Stego Image")
                    btn_reveal = gr.Button("🔓 Extract Hidden Secret", variant="primary")
                with gr.Column():
                    r_out_secret = gr.Image(type="pil", label="Extracted Secret Image")

            btn_reveal.click(
                fn=reveal_only,
                inputs=[r_stego],
                outputs=[r_out_secret],
            )

    gr.Markdown(
        """
        ---
        **StegaWAVE Architecture**:
        - Input images are decomposed using **2D Haar DWT** into 4 frequency subbands (12 channels each).
        - Cover and secret subbands are concatenated (24 channels) and processed through **Invertible Blocks** with Residual Dense Networks (RDN).
        - Inverse pass uses Gaussian noise to substitute residual frequencies and reconstructs the hidden secret via **IWT**.
        """
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())

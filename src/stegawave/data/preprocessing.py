from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

TARGET_SIZE = (256, 256)

# Default data root locations
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
TRANSFORMED_DIR = DATA_DIR / "transformed"

transform = transforms.Compose([
    transforms.Resize(TARGET_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
])


def get_default_transformed_path(image_path: Path, category: str | None = None) -> Path:
    """Determine the default destination path under data/transformed/."""
    if category is None:
        path_str = str(image_path).lower()
        if "cover" in path_str:
            category = "cover"
        elif "secret" in path_str:
            category = "secret"
        else:
            category = "misc"

    return TRANSFORMED_DIR / category / image_path.name


def load_and_preprocess_image(
    image_path: str | Path,
    save_path: str | Path | None = None,
    category: str | None = None,
    save_transformed: bool = True,
) -> torch.Tensor:
    """
    Load an image, convert to RGB, resize to TARGET_SIZE (256, 256).
    Saves the transformed image .
    or to an explicitly provided save_path.

    Returns:
        torch.Tensor: RGB tensor of shape (1, 3, 256, 256) in [0, 1].
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != TARGET_SIZE:
            image = image.resize(TARGET_SIZE, Image.Resampling.BILINEAR)

        if save_transformed:
            if save_path is not None:
                out_path = Path(save_path)
            else:
                out_path = get_default_transformed_path(path, category=category)

            out_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(out_path, quality=95)

        tensor = transforms.functional.to_tensor(image)

    return tensor.unsqueeze(0)

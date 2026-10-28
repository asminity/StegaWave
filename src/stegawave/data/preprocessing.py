from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


TARGET_SIZE = (256, 256)

transform = transforms.Compose([
    transforms.Resize(TARGET_SIZE),
    transforms.ToTensor(),
])


def load_and_preprocess_image(
    image_path: str | Path,
) -> torch.Tensor:
    """Load an image and return an RGB tensor of shape (1, 3, 256, 256)."""

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        tensor = transform(image)

    return tensor.unsqueeze(0)
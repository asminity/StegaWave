"""Train / validation image transforms (official datasets.py).

train: RandomHorizontalFlip, RandomVerticalFlip, RandomCrop(224), ToTensor
val:   CenterCrop(1024), ToTensor
Pixels end up in [0, 1] with no normalisation. EnsureMinSize is an addition:
it makes too-small images fail clearly, or upscales them on request.
"""

from PIL import Image
from torchvision import transforms as T

from model.data.config import POLICIES, SmallImagePolicy


class EnsureMinSize:
    """Guarantee both sides are >= size before cropping."""

    def __init__(self, size: int, policy: SmallImagePolicy = "error"):
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        self.size = size
        self.policy = policy

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if width >= self.size and height >= self.size:
            return image
        if self.policy == "error":
            raise ValueError(
                f"Image is {width}x{height}, smaller than the {self.size}px crop "
                "(use small_image_policy='resize' to upscale)."
            )
        scale = self.size / min(width, height)
        new_size = (max(self.size, round(width * scale)), max(self.size, round(height * scale)))
        return image.resize(new_size, Image.Resampling.BICUBIC)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(size={self.size}, policy={self.policy!r})"


def train_transform(crop_size: int = 224, policy: SmallImagePolicy = "error") -> T.Compose:
    return T.Compose([
        EnsureMinSize(crop_size, policy),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomCrop(crop_size),
        T.ToTensor(),
    ])


def val_transform(crop_size: int = 1024, policy: SmallImagePolicy = "error") -> T.Compose:
    return T.Compose([
        EnsureMinSize(crop_size, policy),
        T.CenterCrop(crop_size),
        T.ToTensor(),
    ])

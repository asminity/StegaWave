"""Folder-of-images dataset."""

from collections.abc import Callable, Sequence
from pathlib import Path

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms as T

from model.data.files import list_images


class ImageFolderDataset(Dataset[Tensor]):
    """RGB images from one folder, returned as (3, H, W) float tensors in [0, 1].

    Unlike the official dataset, a file that fails to load raises an error
    naming it instead of silently moving on to the next index.
    """

    def __init__(
        self,
        root: Path | str,
        transform: Callable[[Image.Image], Tensor] | None = None,
        extensions: Sequence[str] = ("png",),
        recursive: bool = False,
    ):
        self.root = Path(root)
        self.files = list_images(self.root, extensions, recursive)
        self.transform = transform or T.ToTensor()

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tensor:
        path = self.files[index]
        try:
            with Image.open(path) as image:
                return self.transform(image.convert("RGB"))
        except Exception as exc:
            raise RuntimeError(f"Failed to load training image {path}: {exc}") from exc

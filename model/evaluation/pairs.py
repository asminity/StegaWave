"""Building (cover, secret) image pairs for evaluation.

pairs_from_folder: consecutive sorted images, first = secret, second = cover
                   (the official half-batch split at batch size 2).
pairs_from_dirs:   separate cover and secret folders matched by sorted order.
"""

from dataclasses import dataclass
from pathlib import Path

from model.data import list_images


@dataclass(frozen=True)
class ImagePair:
    name: str
    cover: Path
    secret: Path


def pairs_from_folder(root: Path | str, extensions=("png",)) -> list[ImagePair]:
    files = list_images(root, extensions)
    if len(files) < 2:
        raise ValueError(f"{root} needs at least two images to form a pair.")
    return [
        ImagePair(name=f"{i:05d}", secret=files[2 * i], cover=files[2 * i + 1])
        for i in range(len(files) // 2)
    ]


def pairs_from_dirs(
    cover_dir: Path | str, secret_dir: Path | str, extensions=("png",)
) -> list[ImagePair]:
    covers = list_images(cover_dir, extensions)
    secrets = list_images(secret_dir, extensions)
    if len(covers) != len(secrets):
        raise ValueError(
            f"{cover_dir} has {len(covers)} images but {secret_dir} has {len(secrets)}; "
            "pairs are matched by sorted order, so the counts must be equal."
        )
    return [
        ImagePair(name=f"{i:05d}", cover=c, secret=s)
        for i, (c, s) in enumerate(zip(covers, secrets, strict=True))
    ]

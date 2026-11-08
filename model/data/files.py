"""Finding image files on disk, in natural sort order."""

import re
from collections.abc import Sequence
from pathlib import Path


def natural_sort_key(path: Path) -> list[int | str]:
    """Sort key so that 'img2.png' comes before 'img10.png'."""
    parts = re.split(r"(\d+)", path.as_posix())
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def list_images(
    root: Path | str,
    extensions: Sequence[str] = ("png",),
    recursive: bool = False,
) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")

    wanted = {ext.lower().lstrip(".") for ext in extensions}
    candidates = root.rglob("*") if recursive else root.iterdir()
    files = [
        path for path in candidates
        if path.is_file() and path.suffix.lower().lstrip(".") in wanted
    ]
    if not files:
        raise ValueError(f"No images with extensions {sorted(wanted)} found in {root}")
    return sorted(files, key=lambda p: natural_sort_key(p.relative_to(root)))

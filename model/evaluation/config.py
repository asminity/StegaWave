"""Evaluation settings."""

from dataclasses import dataclass

from model.data import SmallImagePolicy

BASE_SETTINGS = ("float", "quantized")


@dataclass(frozen=True)
class EvalConfig:
    crop_size: int | None = 1024  # official test: CenterCrop(1024); None = full image
    small_image_policy: SmallImagePolicy = "error"
    settings: tuple[str, ...] = BASE_SETTINGS
    jpeg_qualities: tuple[int, ...] = ()
    extensions: tuple[str, ...] = ("png",)
    seed: int = 0
    device: str = "auto"
    save_images: bool = False
    limit: int | None = None  # evaluate at most this many pairs

    def __post_init__(self) -> None:
        object.__setattr__(self, "settings", tuple(self.settings))
        object.__setattr__(self, "jpeg_qualities", tuple(self.jpeg_qualities))
        object.__setattr__(self, "extensions", tuple(self.extensions))
        unknown = set(self.settings) - set(BASE_SETTINGS)
        if unknown:
            raise ValueError(f"Unknown settings {sorted(unknown)}; choose from {BASE_SETTINGS}")
        if not self.settings and not self.jpeg_qualities:
            raise ValueError("Nothing to evaluate: no settings and no jpeg_qualities.")
        if any(not 1 <= q <= 100 for q in self.jpeg_qualities):
            raise ValueError("jpeg_qualities must be in 1..100")
        if self.crop_size is not None and (self.crop_size < 2 or self.crop_size % 2):
            raise ValueError("crop_size must be an even integer >= 2, or None")
        if self.limit is not None and self.limit < 1:
            raise ValueError("limit must be >= 1 or None")

    @property
    def all_settings(self) -> tuple[str, ...]:
        return self.settings + tuple(f"jpeg{q}" for q in self.jpeg_qualities)

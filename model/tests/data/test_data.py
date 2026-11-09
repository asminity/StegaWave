from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from model.architecture.config import ArchitectureConfig
from model.architecture.steganography import HiNetSteganography
from model.data import (
    DataConfig,
    EnsureMinSize,
    ImageFolderDataset,
    build_train_loader,
    build_val_loader,
    list_images,
    natural_sort_key,
    split_cover_secret,
    train_transform,
    val_transform,
)
from model.training.losses import hinet_losses


def write_image(path: Path, size=(40, 32), seed=0, mode="RGB") -> np.ndarray:
    """Write a random uint8 image; the array shape selects L / RGB / RGBA."""
    rng = np.random.default_rng(seed)
    channels = {"L": (), "RGB": (3,), "RGBA": (4,)}[mode]
    array = rng.integers(0, 256, (size[1], size[0], *channels), dtype=np.uint8)
    Image.fromarray(array).save(path)
    return array


@pytest.fixture
def image_dir(tmp_path):
    root = tmp_path / "train"
    root.mkdir()
    for i in range(6):
        write_image(root / f"img{i}.png", seed=i)
    return root


def test_natural_sort_key():
    names = ["img10.png", "img2.png", "img1.png", "Img3.png"]
    ordered = sorted((Path(n) for n in names), key=natural_sort_key)
    assert [p.name for p in ordered] == ["img1.png", "img2.png", "Img3.png", "img10.png"]


def test_list_images_filters_extensions(tmp_path):
    write_image(tmp_path / "a.png")
    write_image(tmp_path / "b.PNG")
    write_image(tmp_path / "c.jpg")
    (tmp_path / "notes.txt").write_text("x")
    assert [p.name for p in list_images(tmp_path, ("png",))] == ["a.png", "b.PNG"]
    assert len(list_images(tmp_path, (".png", "JPG"))) == 3


def test_list_images_recursive(tmp_path):
    (tmp_path / "sub").mkdir()
    write_image(tmp_path / "a.png")
    write_image(tmp_path / "sub" / "b.png")
    assert len(list_images(tmp_path)) == 1
    assert len(list_images(tmp_path, recursive=True)) == 2


def test_list_images_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_images(tmp_path / "missing")
    with pytest.raises(ValueError, match="No images"):
        list_images(tmp_path)


def test_dataset_returns_rgb_tensors_in_unit_range(tmp_path):
    write_image(tmp_path / "rgb.png")
    write_image(tmp_path / "gray.png", mode="L")
    write_image(tmp_path / "alpha.png", mode="RGBA")
    dataset = ImageFolderDataset(tmp_path)
    assert len(dataset) == 3
    for item in dataset:
        assert item.shape == (3, 32, 40)
        assert item.dtype == torch.float32
        assert 0.0 <= item.min() and item.max() <= 1.0


def test_train_transform_crops_and_stays_in_source(tmp_path):
    array = write_image(tmp_path / "a.png", size=(20, 20))
    torch.manual_seed(0)
    out = train_transform(crop_size=8)(Image.open(tmp_path / "a.png").convert("RGB"))
    assert out.shape == (3, 8, 8)
    source = {tuple(px) for px in (array.reshape(-1, 3) / 255.0).round(5).tolist()}
    cropped = {tuple(px) for px in out.permute(1, 2, 0).reshape(-1, 3).double().numpy().round(5).tolist()}
    assert cropped <= source


def test_train_transform_is_reproducible_with_seed(image_dir):
    image = Image.open(image_dir / "img0.png").convert("RGB")
    transform = train_transform(crop_size=16)
    torch.manual_seed(42)
    a = transform(image)
    torch.manual_seed(42)
    b = transform(image)
    assert torch.equal(a, b)


def test_train_transform_flips_occur():
    image = Image.fromarray(np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3))
    transform = train_transform(crop_size=16)  # full-size crop: only flips vary
    torch.manual_seed(0)
    outputs = {transform(image).sum(dim=0)[0, 0].item() for _ in range(40)}
    assert len(outputs) == 4  # identity, H flip, V flip, both


def test_val_transform_is_deterministic_center_crop(tmp_path):
    array = write_image(tmp_path / "a.png", size=(20, 16))
    out = val_transform(crop_size=8)(Image.open(tmp_path / "a.png").convert("RGB"))
    expected = torch.from_numpy(array[4:12, 6:14]).permute(2, 0, 1).float() / 255
    torch.testing.assert_close(out, expected)


def test_small_image_error_names_the_file(tmp_path):
    write_image(tmp_path / "tiny.png", size=(10, 10))
    dataset = ImageFolderDataset(tmp_path, train_transform(crop_size=16))
    with pytest.raises(RuntimeError, match="tiny.png"):
        dataset[0]


def test_small_image_resize_policy():
    out = EnsureMinSize(16, "resize")(Image.new("RGB", (10, 20)))
    assert min(out.size) == 16 and out.size == (16, 32)
    assert EnsureMinSize(16, "resize")(Image.new("RGB", (30, 20))).size == (30, 20)


def test_corrupt_file_raises_instead_of_skipping(tmp_path):
    write_image(tmp_path / "a.png")
    (tmp_path / "b.png").write_bytes(b"not an image")
    dataset = ImageFolderDataset(tmp_path)
    dataset[0]
    with pytest.raises(RuntimeError, match="b.png"):
        dataset[1]


def test_split_cover_secret_matches_official():
    batch = torch.arange(4.0).view(4, 1, 1, 1).expand(4, 3, 2, 2)
    cover, secret = split_cover_secret(batch)
    assert cover[:, 0, 0, 0].tolist() == [2.0, 3.0]  # data[n // 2:]
    assert secret[:, 0, 0, 0].tolist() == [0.0, 1.0]  # data[:n // 2]
    with pytest.raises(ValueError):
        split_cover_secret(torch.rand(3, 3, 2, 2))


@pytest.mark.parametrize("kwargs", [
    {"crop_size": 15}, {"val_crop_size": 0}, {"batch_size": 3}, {"val_batch_size": 1},
    {"num_workers": -1}, {"small_image_policy": "pad"}, {"extensions": ()},
])
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        DataConfig(**kwargs)


def test_config_defaults_are_official():
    config = DataConfig()
    assert (config.crop_size, config.val_crop_size) == (224, 1024)
    assert (config.batch_size, config.val_batch_size) == (16, 2)
    assert config.extensions == ("png",)
    assert DataConfig(extensions=(".PNG",)).extensions == ("png",)


def test_train_loader_batches(image_dir):
    config = DataConfig(train_dir=image_dir, crop_size=16, batch_size=4)
    loader = build_train_loader(config, generator=torch.Generator().manual_seed(0))
    batches = list(loader)
    assert len(batches) == 1  # 6 images, drop_last=True
    assert batches[0].shape == (4, 3, 16, 16)


def test_train_loader_reproducible(image_dir):
    config = DataConfig(train_dir=image_dir, crop_size=16, batch_size=2)

    def first_epoch():
        torch.manual_seed(1)
        loader = build_train_loader(config, generator=torch.Generator().manual_seed(0))
        return torch.cat(list(loader))

    assert torch.equal(first_epoch(), first_epoch())


def test_val_loader_is_ordered_and_deterministic(image_dir):
    config = DataConfig(val_dir=image_dir, val_crop_size=16, val_batch_size=2)
    a = torch.cat(list(build_val_loader(config)))
    b = torch.cat(list(build_val_loader(config)))
    assert a.shape == (6, 3, 16, 16)
    assert torch.equal(a, b)


def test_loader_rejects_too_few_images(image_dir):
    with pytest.raises(ValueError, match="fewer than the batch size"):
        build_train_loader(DataConfig(train_dir=image_dir, crop_size=16, batch_size=8))
    with pytest.raises(ValueError, match="train_dir"):
        build_train_loader(DataConfig())


def test_end_to_end_training_step(image_dir):
    config = DataConfig(train_dir=image_dir, crop_size=16, batch_size=4)
    model = HiNetSteganography(ArchitectureConfig(num_blocks=2))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch = next(iter(build_train_loader(config, torch.Generator().manual_seed(0))))
    cover, secret = split_cover_secret(batch)
    hidden, revealed = model(cover, secret)
    loss = hinet_losses(hidden, revealed, cover, secret)
    loss.total.backward()
    optimizer.step()

    assert torch.isfinite(loss.total)
    assert cover.shape == secret.shape == (2, 3, 16, 16)

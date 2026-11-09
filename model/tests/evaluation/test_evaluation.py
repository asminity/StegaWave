import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from model.architecture.config import ArchitectureConfig
from model.cli.evaluate import main
from model.data import DataConfig
from model.evaluation import (
    EvalConfig,
    evaluate,
    jpeg_round_trip,
    load_image,
    pairs_from_dirs,
    pairs_from_folder,
    quantize_8bit,
)
from model.evaluation.metrics import to_uint8, uint8_metrics
from model.training.checkpoint import load_trained_model
from model.training.config import ExperimentConfig, TrainConfig
from model.training.trainer import Trainer


def write_images(root: Path, count: int, size=(24, 24), seed=0) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = []
    for i in range(count):
        path = root / f"{i}.png"
        Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)).save(path)
        paths.append(path)
    return paths


@pytest.fixture(scope="module")
def trained(tmp_path_factory) -> tuple[Path, Path]:
    """A checkpoint trained for one epoch on synthetic images, plus an eval folder."""
    root = tmp_path_factory.mktemp("eval")
    write_images(root / "train", 8)
    write_images(root / "val", 6, seed=1)
    config = ExperimentConfig(
        architecture=ArchitectureConfig(num_blocks=2),
        data=DataConfig(train_dir=root / "train", crop_size=16, batch_size=4),
        train=TrainConfig(epochs=1, learning_rate=1e-3, output_dir=root / "run", device="cpu"),
    )
    Trainer(config, log=lambda _: None).fit()
    return root / "run" / "model.pt", root / "val"


# ---------------------------------------------------------------- helpers

def test_quantize_8bit():
    x = torch.tensor([-0.2, 0.0, 0.5, 0.123, 1.0, 1.4])
    q = quantize_8bit(x)
    torch.testing.assert_close(q * 255, (q * 255).round())
    assert q.min() == 0 and q.max() == 1


def test_quantized_tensor_equals_png_contents(tmp_path):
    x = torch.rand(1, 3, 8, 8) * 1.2 - 0.1
    Image.fromarray(to_uint8(x)).save(tmp_path / "x.png")
    reloaded = TF.to_tensor(Image.open(tmp_path / "x.png")).unsqueeze(0)
    torch.testing.assert_close(reloaded, quantize_8bit(x), rtol=0, atol=1e-7)


def test_jpeg_round_trip_is_lossy_but_close():
    x = quantize_8bit(torch.rand(2, 3, 16, 16))
    y = jpeg_round_trip(x, quality=90)
    assert y.shape == x.shape
    assert not torch.equal(x, y)
    assert (x - y).abs().mean() < 0.2


# ------------------------------------------------------------------ pairs

def test_pairs_from_folder_uses_official_order(tmp_path):
    paths = write_images(tmp_path, 5)
    pairs = pairs_from_folder(tmp_path)
    assert len(pairs) == 2  # the fifth image is unpaired
    assert (pairs[0].secret, pairs[0].cover) == (paths[0], paths[1])
    assert (pairs[1].secret, pairs[1].cover) == (paths[2], paths[3])


def test_pairs_from_dirs(tmp_path):
    covers = write_images(tmp_path / "c", 3)
    secrets = write_images(tmp_path / "s", 3, seed=1)
    pairs = pairs_from_dirs(tmp_path / "c", tmp_path / "s")
    assert [(p.cover, p.secret) for p in pairs] == list(zip(covers, secrets))
    write_images(tmp_path / "s2", 2)
    with pytest.raises(ValueError, match="counts must be equal"):
        pairs_from_dirs(tmp_path / "c", tmp_path / "s2")


def test_load_image_crop_and_full(tmp_path):
    (path,) = write_images(tmp_path, 1, size=(25, 23))
    assert load_image(path, 16, "error").shape == (1, 3, 16, 16)
    assert load_image(path, None, "error").shape == (1, 3, 22, 24)
    with pytest.raises(ValueError, match="0.png"):
        load_image(path, 32, "error")
    assert load_image(path, 32, "resize").shape == (1, 3, 32, 32)


@pytest.mark.parametrize("kwargs", [
    {"settings": ("lossy",)}, {"settings": (), "jpeg_qualities": ()},
    {"jpeg_qualities": (0,)}, {"crop_size": 15}, {"limit": 0},
])
def test_eval_config_validation(kwargs):
    with pytest.raises(ValueError):
        EvalConfig(**kwargs)


# -------------------------------------------------------------- evaluate

def test_evaluate_end_to_end(trained, tmp_path):
    checkpoint, images = trained
    config = EvalConfig(crop_size=16, jpeg_qualities=(90,), save_images=True, device="cpu")
    result = evaluate(checkpoint, pairs_from_folder(images), config, output_dir=tmp_path / "out")

    assert len(result.rows) == 3 * 3  # 3 pairs x (float, quantized, jpeg90)
    assert {r["setting"] for r in result.rows} == {"float", "quantized", "jpeg90"}
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["num_pairs"] == 3 and summary["checkpoint_epoch"] == 1
    for setting in ("float", "quantized", "jpeg90"):
        stats = summary["settings"][setting]
        for key in ("official_psnr_cover", "official_psnr_secret", "cover_psnr",
                    "cover_ssim", "cover_mae", "cover_rmse", "secret_psnr", "secret_ssim"):
            assert math.isfinite(stats[key]["mean"]), (setting, key)

    csv_lines = (tmp_path / "out" / "results.csv").read_text().splitlines()
    assert len(csv_lines) == 1 + 9
    folder = tmp_path / "out" / "images" / "00000"
    expected = {"cover.png", "secret.png"} | {
        f"{kind}_{s}.png" for kind in ("stego", "recovered") for s in ("float", "quantized", "jpeg90")
    }
    assert {p.name for p in folder.iterdir()} == expected


def test_quantized_setting_matches_reveal_from_saved_png(trained, tmp_path):
    checkpoint, images = trained
    config = EvalConfig(crop_size=16, settings=("quantized",), save_images=True,
                        device="cpu", limit=1)
    result = evaluate(checkpoint, pairs_from_folder(images), config, output_dir=tmp_path)

    model = load_trained_model(checkpoint)
    folder = tmp_path / "images" / "00000"
    stego = TF.to_tensor(Image.open(folder / "stego_quantized.png")).unsqueeze(0)
    secret = TF.to_tensor(Image.open(folder / "secret.png")).unsqueeze(0)
    cover = TF.to_tensor(Image.open(folder / "cover.png")).unsqueeze(0)
    with torch.no_grad():
        hidden = model.hide(cover, secret)
        z = model.sample_z(hidden.stego_wavelet, torch.Generator().manual_seed(0))
        recovered = model.reveal_image(stego, z=z).recovered_secret
    row = result.rows[0]
    assert uint8_metrics(recovered, secret)["psnr"] == pytest.approx(row["secret_psnr"])
    assert uint8_metrics(stego, cover)["psnr"] == pytest.approx(row["cover_psnr"])


def test_evaluate_is_reproducible_and_seed_dependent(trained):
    checkpoint, images = trained
    pairs = pairs_from_folder(images)

    def run(seed):
        return evaluate(checkpoint, pairs, EvalConfig(crop_size=16, seed=seed, device="cpu")).rows

    assert run(0) == run(0)
    assert run(0) != run(5)


def test_float_and_quantized_differ_only_in_stego_channel(trained):
    checkpoint, images = trained
    rows = evaluate(checkpoint, pairs_from_folder(images),
                    EvalConfig(crop_size=16, device="cpu", limit=1)).rows
    by_setting = {r["setting"]: r for r in rows}
    assert by_setting["float"]["official_psnr_cover"] != by_setting["quantized"]["official_psnr_cover"]
    # uint8 cover metrics see the same 8-bit stego in both settings
    assert by_setting["float"]["cover_psnr"] == pytest.approx(by_setting["quantized"]["cover_psnr"])


def test_full_image_requires_matching_sizes(trained, tmp_path):
    checkpoint, _ = trained
    write_images(tmp_path / "c", 1, size=(24, 24))
    write_images(tmp_path / "s", 1, size=(32, 24))
    pairs = pairs_from_dirs(tmp_path / "c", tmp_path / "s")
    with pytest.raises(ValueError, match="differ in size"):
        evaluate(checkpoint, pairs, EvalConfig(crop_size=None, device="cpu"))


def test_rejects_non_training_checkpoint(tmp_path):
    torch.save({"net": {}}, tmp_path / "pretrained_like.pt")
    write_images(tmp_path / "imgs", 2)
    with pytest.raises(ValueError, match="not a stegawave-hinet-v1"):
        evaluate(tmp_path / "pretrained_like.pt", pairs_from_folder(tmp_path / "imgs"))


def test_cli(trained, tmp_path, capsys):
    checkpoint, images = trained
    result = main([
        "--checkpoint", str(checkpoint), "--images", str(images), "--crop-size", "16",
        "--jpeg-quality", "75", "--device", "cpu", "--output-dir", str(tmp_path / "cli"),
    ])
    assert {r["setting"] for r in result.rows} == {"float", "quantized", "jpeg75"}
    assert (tmp_path / "cli" / "summary.json").is_file()
    assert "official_psnr_secret" in capsys.readouterr().out


def test_cli_requires_both_pair_dirs(trained, tmp_path):
    checkpoint, images = trained
    with pytest.raises(SystemExit):
        main(["--checkpoint", str(checkpoint), "--cover-dir", str(images)])

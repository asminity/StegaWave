import numpy as np
from PIL import Image

def test_numpy_array_creation():
    image_array = np.zeros((64, 64, 3), dtype=np.uint8)

    assert image_array.shape == (64, 64, 3)
    assert image_array.dtype == np.uint8


def test_pillow_image_creation():
    image = Image.new("RGB", (64, 64), color="white")

    assert image.size == (64, 64)
    assert image.mode == "RGB"
"""
StegaWAVE Model Checkpoint Verification Script.

Loads the saved checkpoint, validates parameter loading, and executes a dry
forward and reverse pass with synthetic wavelet inputs on CPU.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stegawave.pipeline import verify_checkpoint

CHECKPOINT = (
    PROJECT_ROOT / "checkpoints" / "model.pt"
    if (PROJECT_ROOT / "checkpoints" / "model.pt").is_file()
    else PROJECT_ROOT / "model.pt"
)


def main():
    verify_checkpoint(CHECKPOINT)


if __name__ == "__main__":
    main()

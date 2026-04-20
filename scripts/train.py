"""
One-time training script. Run after initial data ingestion to fit the TFT.

Usage:
    python scripts/train.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.trainer import run_training

if __name__ == "__main__":
    run_training()

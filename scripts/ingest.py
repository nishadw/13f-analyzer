"""
Initial full data ingestion. Run once to bootstrap the feature store.

Usage:
    python scripts/ingest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.pipeline import run_full_ingestion

if __name__ == "__main__":
    run_full_ingestion()

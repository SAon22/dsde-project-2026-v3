"""Paths and thresholds shared by every pipeline stage.

Override the data folder with the CCTV_DATA_DIR environment variable.
"""

from __future__ import annotations

import os
from datetime import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("CCTV_DATA_DIR", PROJECT_ROOT / "data"))

RAW_DIR = DATA_DIR / "raw"
IMAGES_DIR = RAW_DIR / "cctv_images"
CAMERA_INFO_CSV = RAW_DIR / "cctv_info.csv"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"

# Capture schedule observed in the dataset (25-31 Aug 2026).
CAPTURE_INTERVAL_MIN = 10
DAY_START = time(6, 0)
DAY_END = time(21, 0)  # last slot that starts at or before this time

# A failed capture is saved as a tiny, all-white 400x266 JPEG (~1.4 KB).
PLACEHOLDER_MAX_BYTES = 5_000
PLACEHOLDER_SIZE = (400, 266)
EXPECTED_SIZE = (352, 288)

# Mean grey level (0-255) below which a frame counts as dark.
DARK_MEAN_THRESHOLD = 60

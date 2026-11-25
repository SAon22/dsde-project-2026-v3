"""Load the manifest and pick which frames to run YOLO on."""

import pandas as pd

from cctv import config
from cctv.detect.counting import time_of_day


def load_usable_frames():
    """Read data/interim/manifest.parquet and keep only real, readable frames.

    We drop blank placeholders, unreadable files and duplicate slots. Frozen
    frames are dropped too because they repeat an earlier picture.
    A "light" column (day / dusk / night) is added for later analysis.
    """
    path = config.INTERIM_DIR / "manifest.parquet"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run first: python -m cctv.ingest.manifest --pixels")
    manifest = pd.read_parquet(path)

    usable = manifest["read_ok"] & ~manifest["is_placeholder"] & ~manifest["is_duplicate_slot"]
    if "is_frozen" in manifest.columns:
        usable = usable & ~manifest["is_frozen"]
    frames = manifest[usable].copy()
    frames["light"] = frames["captured_at"].apply(time_of_day)
    return frames


def one_frame_per_hour(frames):
    """Keep the first frame of each camera in each hour (about 105 per camera)."""
    frames = frames.sort_values("captured_at").copy()
    frames["hour"] = frames["captured_at"].dt.floor("h")
    return frames.drop_duplicates(subset=["camera_id", "hour"], keep="first")

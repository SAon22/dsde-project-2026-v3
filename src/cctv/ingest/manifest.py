"""Stage 1: list every image and check its quality.

Outputs (in data/interim/):
  manifest.parquet  one row per image file, with quality flags
  coverage.parquet  one row per camera x 10-minute slot: ok / placeholder / invalid / missing

Run:
  python -m cctv.ingest.manifest            # fast: reads file names and image sizes only
  python -m cctv.ingest.manifest --pixels   # also brightness and frozen-frame check (reads every image)
"""

import argparse
import hashlib
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from PIL import Image, ImageStat

from cctv import config

# File names look like: 1028_<location>_<detail>_20260825_061012.jpg
# The location part is Thai text that can itself contain "_" and "-", so we only
# trust the camera id at the start and the date and time at the end.
FILENAME_PATTERN = re.compile(r"^(\d+)_(.+)_(\d{8})_(\d{6})\.jpe?g$", re.IGNORECASE)


def parse_filename(name):
    """Return (camera_id, label, timestamp) from a file name, or None if it doesn't match."""
    match = FILENAME_PATTERN.match(name)
    if match is None:
        return None
    camera_id, label, date_text, time_text = match.groups()
    try:
        timestamp = datetime.strptime(date_text + time_text, "%Y%m%d%H%M%S")
    except ValueError:  # e.g. month 13
        return None
    return int(camera_id), label, timestamp


def read_image_info(path, with_pixels):
    """Open one image and return its size, plus brightness and a content hash if asked.

    The content hash (MD5 of the file bytes) is identical only when two files are
    byte-for-byte the same, which is what a frozen camera feed produces.
    """
    info = {"width": None, "height": None, "read_ok": True}
    try:
        with Image.open(path) as image:
            info["width"], info["height"] = image.size
            if with_pixels:
                grey = image.convert("L")
                info["mean_grey"] = ImageStat.Stat(grey).mean[0]  # 0 = black, 255 = white
                info["file_md5"] = hashlib.md5(path.read_bytes()).hexdigest()
    except Exception:  # broken or cut-off JPEG
        info["read_ok"] = False
    return info


def scan_images(images_dir, with_pixels=False):
    """Walk images_dir/<camera_id>/*.jpg and return one row per file."""
    rows = []
    for path in sorted(images_dir.glob("*/*")):
        if not path.is_file():
            continue
        parsed = parse_filename(path.name)
        row = {
            "folder_camera_id": int(path.parent.name) if path.parent.name.isdigit() else None,
            "camera_id": parsed[0] if parsed else None,
            "file_label": parsed[1] if parsed else None,
            "captured_at": parsed[2] if parsed else None,
            "rel_path": path.relative_to(images_dir).as_posix(),
            "bytes": path.stat().st_size,
            "parse_ok": parsed is not None,
        }
        row.update(read_image_info(path, with_pixels))
        rows.append(row)
    manifest = pd.DataFrame(rows)
    manifest["captured_at"] = pd.to_datetime(manifest["captured_at"])
    return manifest


def add_quality_flags(manifest):
    """Add True/False quality columns and the 10-minute slot of each frame."""
    m = manifest.sort_values(["camera_id", "captured_at"]).reset_index(drop=True)

    # A failed capture is saved as a tiny all-white 400x266 JPEG (about 1.4 KB).
    tiny_file = m["bytes"] < config.PLACEHOLDER_MAX_BYTES
    placeholder_size = (m["width"] == config.PLACEHOLDER_SIZE[0]) & (m["height"] == config.PLACEHOLDER_SIZE[1])
    m["is_placeholder"] = tiny_file | placeholder_size

    # Most cameras are 352x288. A few use another size; that is fine, just noted.
    normal_size = (m["width"] == config.EXPECTED_SIZE[0]) & (m["height"] == config.EXPECTED_SIZE[1])
    m["other_resolution"] = ~m["is_placeholder"] & ~normal_size

    m["camera_mismatch"] = m["camera_id"] != m["folder_camera_id"]

    # Round each timestamp down to its 10-minute slot (06:13 -> 06:10).
    m["slot"] = m["captured_at"].dt.floor(f"{config.CAPTURE_INTERVAL_MIN}min")
    # If two files land in the same slot, keep the first and flag the others.
    m["is_duplicate_slot"] = m.duplicated(subset=["camera_id", "slot"], keep="first")

    if "mean_grey" in m.columns:
        m["is_dark"] = m["mean_grey"] < config.DARK_MEAN_THRESHOLD
        # Frozen = exactly the same file as the camera's previous real frame.
        m["is_frozen"] = False
        previous_md5 = {}
        for i, row in m.iterrows():
            if row["is_placeholder"] or not row["read_ok"] or row["is_duplicate_slot"]:
                continue
            if previous_md5.get(row["camera_id"]) == row["file_md5"]:
                m.at[i, "is_frozen"] = True
            previous_md5[row["camera_id"]] = row["file_md5"]

    problem = m["is_placeholder"] | ~m["read_ok"] | ~m["parse_ok"] | m["is_duplicate_slot"]
    if "is_frozen" in m.columns:
        problem = problem | m["is_frozen"]
    m["is_valid"] = ~problem
    return m


def build_coverage(manifest):
    """Make the full camera x 10-minute grid and mark each slot.

    Status is ok, placeholder, invalid (frozen or unreadable), or missing (no file at all). Missing slots
    matter: they mean "no data", never "zero traffic".
    """
    m = manifest[manifest["slot"].notna() & ~manifest["is_duplicate_slot"]]

    # Every slot from 06:00 to 21:00, every 10 minutes, on every day in the data.
    all_slots = []
    for day in pd.date_range(m["slot"].min().normalize(), m["slot"].max().normalize(), freq="D"):
        start = day + pd.Timedelta(hours=config.DAY_START.hour, minutes=config.DAY_START.minute)
        end = day + pd.Timedelta(hours=config.DAY_END.hour, minutes=config.DAY_END.minute)
        all_slots += list(pd.date_range(start, end, freq=f"{config.CAPTURE_INTERVAL_MIN}min"))

    grid = pd.MultiIndex.from_product(
        [sorted(m["camera_id"].unique()), all_slots], names=["camera_id", "slot"]
    ).to_frame(index=False)

    columns = ["camera_id", "slot", "rel_path", "is_placeholder", "is_valid"]
    coverage = grid.merge(m[columns], on=["camera_id", "slot"], how="left")
    coverage["status"] = "ok"
    # A file that is not valid and not a placeholder is frozen (or, rarely, unreadable).
    has_file = coverage["rel_path"].notna()
    coverage.loc[has_file & (coverage["is_valid"] == False), "status"] = "invalid"
    coverage.loc[has_file & (coverage["is_placeholder"] == True), "status"] = "placeholder"
    coverage.loc[coverage["rel_path"].isna(), "status"] = "missing"
    return coverage[["camera_id", "slot", "status", "rel_path"]]


def print_summary(manifest, coverage):
    """Print the main counts so you can sanity-check the run."""
    print(f"files:             {len(manifest):,}")
    print(f"cameras:           {manifest['camera_id'].nunique()}")
    print(f"time range:        {manifest['captured_at'].min()} -> {manifest['captured_at'].max()}")
    print(f"unparsed names:    {(~manifest['parse_ok']).sum()}")
    print(f"unreadable:        {(~manifest['read_ok']).sum()}")
    print(f"placeholders:      {manifest['is_placeholder'].sum()}")
    print(f"duplicate slots:   {manifest['is_duplicate_slot'].sum()}")
    print(f"camera mismatch:   {manifest['camera_mismatch'].sum()}")
    print(f"other resolution:  {manifest['other_resolution'].sum()}")
    if "is_frozen" in manifest.columns:
        print(f"frozen frames:     {manifest['is_frozen'].sum()}")
        print(f"dark frames:       {manifest['is_dark'].sum()}")
    print(f"valid frames:      {manifest['is_valid'].sum():,}")
    print(f"coverage status:   {coverage['status'].value_counts().to_dict()}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-dir", type=Path, default=config.IMAGES_DIR)
    parser.add_argument("--out-dir", type=Path, default=config.INTERIM_DIR)
    parser.add_argument("--pixels", action="store_true", help="also check brightness and frozen frames (slower)")
    args = parser.parse_args(argv)

    if not args.images_dir.is_dir():
        raise SystemExit(f"Images folder not found: {args.images_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = add_quality_flags(scan_images(args.images_dir, with_pixels=args.pixels))
    coverage = build_coverage(manifest)
    manifest.to_parquet(args.out_dir / "manifest.parquet", index=False)
    coverage.to_parquet(args.out_dir / "coverage.parquet", index=False)
    print_summary(manifest, coverage)
    print(f"\nWrote {args.out_dir / 'manifest.parquet'} and {args.out_dir / 'coverage.parquet'}")


if __name__ == "__main__":
    main()

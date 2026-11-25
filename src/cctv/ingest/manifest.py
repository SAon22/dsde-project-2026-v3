"""Stage 1: build the image manifest and the 10-minute coverage grid.

Outputs (in data/interim/):
  manifest.parquet  one row per image file
  coverage.parquet  one row per camera x 10-minute slot: ok / placeholder / missing

Run:
  python -m cctv.ingest.manifest            # fast: reads file headers only
  python -m cctv.ingest.manifest --pixels   # also brightness + frozen-frame hash
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from PIL import Image, ImageStat

from cctv import config

# camera_id _ <location and detail, may hold "_" and "-"> _ YYYYMMDD _ HHMMSS .jpg
FILENAME_RE = re.compile(
    r"^(?P<camera_id>\d+)_(?P<label>.+)_(?P<date>\d{8})_(?P<time>\d{6})\.jpe?g$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedName:
    camera_id: int
    label: str
    captured_at: datetime


def parse_filename(name: str) -> ParsedName | None:
    """Parse an image file name. Returns None when it does not match."""
    m = FILENAME_RE.match(name)
    if not m:
        return None
    try:
        ts = datetime.strptime(m["date"] + m["time"], "%Y%m%d%H%M%S")
    except ValueError:
        return None
    return ParsedName(int(m["camera_id"]), m["label"], ts)


def _average_hash(img: Image.Image) -> str:
    """64-bit average hash as hex; equal hashes in a row suggest a frozen feed."""
    small = img.convert("L").resize((8, 8), Image.Resampling.BILINEAR)
    px = list(small.tobytes())
    mean = sum(px) / 64
    bits = "".join("1" if p >= mean else "0" for p in px)
    return f"{int(bits, 2):016x}"


def scan_images(images_dir: Path, with_pixels: bool = False) -> pd.DataFrame:
    """Walk images_dir/<camera_id>/*.jpg and return one row per file."""
    rows = []
    for path in sorted(images_dir.glob("*/*")):
        if not path.is_file():
            continue
        parsed = parse_filename(path.name)
        row = {
            "folder_camera_id": int(path.parent.name) if path.parent.name.isdigit() else None,
            "camera_id": parsed.camera_id if parsed else None,
            "captured_at": parsed.captured_at if parsed else pd.NaT,
            "file_label": parsed.label if parsed else None,
            "rel_path": path.relative_to(images_dir).as_posix(),
            "bytes": path.stat().st_size,
            "parse_ok": parsed is not None,
            "width": None,
            "height": None,
            "read_ok": True,
        }
        try:
            with Image.open(path) as img:
                row["width"], row["height"] = img.size
                if with_pixels:
                    row["mean_grey"] = ImageStat.Stat(img.convert("L")).mean[0]
                    row["ahash"] = _average_hash(img)
        except Exception:  # corrupt or truncated JPEG
            row["read_ok"] = False
        rows.append(row)
    return pd.DataFrame(rows)


def add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean flags and the 10-minute slot each frame belongs to."""
    df = df.copy()
    pw, ph = config.PLACEHOLDER_SIZE
    ew, eh = config.EXPECTED_SIZE
    df["is_placeholder"] = (df["bytes"] < config.PLACEHOLDER_MAX_BYTES) | (
        (df["width"] == pw) & (df["height"] == ph)
    )
    df["unexpected_size"] = ~df["is_placeholder"] & ~((df["width"] == ew) & (df["height"] == eh))
    df["camera_mismatch"] = df["camera_id"] != df["folder_camera_id"]
    df["slot"] = df["captured_at"].dt.floor(f"{config.CAPTURE_INTERVAL_MIN}min")
    df = df.sort_values(["camera_id", "captured_at"]).reset_index(drop=True)
    # Two files landing in the same slot: keep the first, flag the rest.
    df["is_duplicate_slot"] = df.duplicated(["camera_id", "slot"], keep="first") & df["slot"].notna()

    if "mean_grey" in df:
        df["is_dark"] = df["mean_grey"] < config.DARK_MEAN_THRESHOLD
    if "ahash" in df:
        # Compare with the previous usable frame (skip placeholders and duplicates).
        usable = ~df["is_placeholder"] & ~df["is_duplicate_slot"] & df["ahash"].notna()
        prev = df[usable].groupby("camera_id")["ahash"].shift()
        df["is_frozen"] = False
        df.loc[usable, "is_frozen"] = df.loc[usable, "ahash"] == prev

    bad = df["is_placeholder"] | ~df["read_ok"] | ~df["parse_ok"] | df["is_duplicate_slot"]
    if "is_frozen" in df:
        bad |= df["is_frozen"]
    df["is_valid"] = ~bad
    return df


def build_coverage(manifest: pd.DataFrame) -> pd.DataFrame:
    """Expected camera x slot grid with status ok / placeholder / missing.

    Missing slots matter: a gap must show as "no data", never as zero traffic.
    """
    m = manifest[manifest["slot"].notna() & ~manifest["is_duplicate_slot"]]
    if m.empty:
        return pd.DataFrame(columns=["camera_id", "slot", "status"])
    days = pd.date_range(m["slot"].min().normalize(), m["slot"].max().normalize(), freq="D")
    start = pd.Timedelta(hours=config.DAY_START.hour, minutes=config.DAY_START.minute)
    end = pd.Timedelta(hours=config.DAY_END.hour, minutes=config.DAY_END.minute)
    step = pd.Timedelta(minutes=config.CAPTURE_INTERVAL_MIN)
    offsets = pd.timedelta_range(start, end, freq=step)
    slots = pd.DatetimeIndex([d + o for d in days for o in offsets])

    grid = pd.MultiIndex.from_product(
        [sorted(m["camera_id"].dropna().unique()), slots], names=["camera_id", "slot"]
    ).to_frame(index=False)
    cov = grid.merge(
        m[["camera_id", "slot", "is_placeholder", "is_valid", "rel_path"]],
        on=["camera_id", "slot"],
        how="left",
    )
    cov["status"] = "ok"
    cov.loc[cov["is_placeholder"].fillna(False).astype(bool), "status"] = "placeholder"
    cov.loc[cov["rel_path"].isna(), "status"] = "missing"
    cov.loc[(cov["status"] == "ok") & ~cov["is_valid"].fillna(True).astype(bool), "status"] = "invalid"
    return cov[["camera_id", "slot", "status", "rel_path"]]


def summarise(manifest: pd.DataFrame, coverage: pd.DataFrame) -> str:
    lines = [
        f"files:            {len(manifest):,}",
        f"cameras:          {manifest['camera_id'].nunique()}",
        f"time range:       {manifest['captured_at'].min()} -> {manifest['captured_at'].max()}",
        f"unparsed names:   {(~manifest['parse_ok']).sum()}",
        f"unreadable:       {(~manifest['read_ok']).sum()}",
        f"placeholders:     {manifest['is_placeholder'].sum()}",
        f"duplicate slots:  {manifest['is_duplicate_slot'].sum()}",
        f"camera mismatch:  {manifest['camera_mismatch'].sum()}",
        f"unexpected size:  {manifest['unexpected_size'].sum()}",
    ]
    if "is_frozen" in manifest:
        lines.append(f"frozen frames:    {manifest['is_frozen'].sum()}")
    if "is_dark" in manifest:
        lines.append(f"dark frames:      {manifest['is_dark'].sum()}")
    lines.append(f"valid frames:     {manifest['is_valid'].sum():,}")
    lines.append("coverage status:  " + coverage["status"].value_counts().to_dict().__repr__())
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", type=Path, default=config.IMAGES_DIR)
    ap.add_argument("--out-dir", type=Path, default=config.INTERIM_DIR)
    ap.add_argument("--pixels", action="store_true", help="also compute brightness and frozen-frame hash (slower)")
    args = ap.parse_args(argv)

    if not args.images_dir.is_dir():
        raise SystemExit(f"Images folder not found: {args.images_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = add_quality_flags(scan_images(args.images_dir, with_pixels=args.pixels))
    coverage = build_coverage(manifest)
    manifest.to_parquet(args.out_dir / "manifest.parquet", index=False)
    coverage.to_parquet(args.out_dir / "coverage.parquet", index=False)
    print(summarise(manifest, coverage))
    print(f"\nWrote {args.out_dir / 'manifest.parquet'} and {args.out_dir / 'coverage.parquet'}")


if __name__ == "__main__":
    main()

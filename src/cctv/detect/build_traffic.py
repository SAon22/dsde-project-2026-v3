"""Step 3: build the final traffic table from the full YOLO run.

Run (after `python -m cctv.detect.yolo_runner --full`):
  python -m cctv.detect.build_traffic

Output: data/processed/traffic.parquet (and traffic.csv for Excel / BI tools)
One row per camera per 10-minute slot, 06:00-21:00, 25-31 Aug 2026.
Slots with no usable image keep EMPTY counts (not 0), so a camera outage is never
mistaken for an empty road.
"""

import pandas as pd

from cctv import config
from cctv.detect.counting import ALL_COUNTED, measure_all_frames, time_of_day
from cctv.detect.frames import load_usable_frames
from cctv.detect.roi import load_rois
from cctv.detect.yolo_runner import FULL_RUN_MODEL, frames_done, load_saved_boxes

CONFIDENCE = 0.20  # chosen in the pilot together with FULL_RUN_MODEL
PILOT_SUMMARY = config.INTERIM_DIR / "pilot" / "compare_summary.csv"
COUNT_COLUMNS = ALL_COUNTED + ["vehicles", "congestion", "vehicles_adjusted"]


def correction_factors(summary, model=FULL_RUN_MODEL, confidence=CONFIDENCE):
    """How much to scale YOLO's vehicle count for day, dusk and night.

    From the pilot we know, per light level, the average hand count and the average
    YOLO error (bias). YOLO's average is hand + bias, so
        factor = hand / (hand + bias)
    Example: at night you counted 4.35 and YOLO was 1.05 lower (3.30), so the
    factor is 4.35 / 3.30 = 1.32: night counts are scaled up by 32%.
    """
    rows = summary[(summary["model"] == model) & (summary["confidence"] == confidence)]
    factors = {}
    for row in rows.to_dict("records"):
        if row["light"] == "all":
            continue
        yolo_average = row["hand_vehicles_per_frame"] + row["bias_vehicles"]
        factors[row["light"]] = row["hand_vehicles_per_frame"] / yolo_average
    return factors


def load_correction_factors():
    """Read the pilot summary; fall back to no correction (factor 1) if it is missing."""
    if not PILOT_SUMMARY.exists():
        print(f"Warning: {PILOT_SUMMARY} not found, vehicles_adjusted = vehicles")
        return {"day": 1.0, "dusk": 1.0, "night": 1.0}
    return correction_factors(pd.read_csv(PILOT_SUMMARY))


def build_table(coverage, measured, cameras, factors):
    """Join the 10-minute grid, the per-frame counts and the camera information.

    coverage: camera_id, slot, status, rel_path (every expected slot, from the manifest step)
    measured: one row per detected frame with rel_path and the count columns
    cameras:  cctv_info.csv
    """
    # 1. Start from the full grid so missing slots stay in the table (with empty counts).
    count_columns = ["rel_path"] + ALL_COUNTED + ["vehicles", "congestion"]
    table = coverage.merge(measured[count_columns], on="rel_path", how="left")

    # 2. Time columns, worked out from the slot so they exist for missing slots too.
    table["date"] = table["slot"].dt.date
    table["hour"] = table["slot"].dt.hour
    table["weekday"] = table["slot"].dt.day_name()
    table["is_weekend"] = table["slot"].dt.dayofweek >= 5
    table["light"] = table["slot"].apply(time_of_day)

    # 3. Night correction: scale the count by the pilot factor for that light level.
    table["correction_factor"] = table["light"].map(factors).fillna(1.0)
    table["vehicles_adjusted"] = table["vehicles"] * table["correction_factor"]

    # 4. Camera name and position for maps.
    camera_columns = ["camera_id", "location", "location_en", "location_detail_en", "latitude", "longitude"]
    table = table.merge(cameras[camera_columns], on="camera_id", how="left")

    ordered = (
        ["camera_id", "location_en", "location", "location_detail_en", "latitude", "longitude",
         "slot", "date", "hour", "weekday", "is_weekend", "light", "status"]
        + ALL_COUNTED + ["vehicles", "vehicles_adjusted", "correction_factor", "congestion", "rel_path"]
    )
    return table[ordered].sort_values(["camera_id", "slot"]).reset_index(drop=True)


def print_summary(table, factors):
    """A few numbers to sanity-check the result."""
    ok = table[table["status"] == "ok"]
    print(f"rows: {len(table):,} ({len(ok):,} with counts, {len(table) - len(ok):,} empty slots)")
    print("correction factors:", {k: round(v, 3) for k, v in factors.items()})
    print("\nAverage vehicles per frame by light:")
    print(ok.groupby("light")[["vehicles", "vehicles_adjusted", "congestion"]].mean().round(2).to_string())
    print("\nBusiest 5 cameras (adjusted vehicles per frame):")
    busiest = ok.groupby(["camera_id", "location_en"])["vehicles_adjusted"].mean().sort_values(ascending=False)
    print(busiest.head(5).round(2).to_string())


def main():
    frames = load_usable_frames()
    done = frames_done("full", FULL_RUN_MODEL)
    not_done = frames[~frames["rel_path"].isin(done)]
    if len(not_done) > 0:
        raise SystemExit(
            f"{len(not_done):,} frames have no YOLO result yet. Run: python -m cctv.detect.yolo_runner --full"
        )

    print("Measuring counts and congestion for every frame (takes a few minutes)...")
    boxes = load_saved_boxes("full", FULL_RUN_MODEL)
    measured = measure_all_frames(frames, boxes, load_rois(), confidence=CONFIDENCE)

    coverage = pd.read_parquet(config.INTERIM_DIR / "coverage.parquet")
    cameras = pd.read_csv(config.CAMERA_INFO_CSV)
    factors = load_correction_factors()
    table = build_table(coverage, measured, cameras, factors)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(config.PROCESSED_DIR / "traffic.parquet", index=False)
    table.to_csv(config.PROCESSED_DIR / "traffic.csv", index=False, encoding="utf-8-sig")
    print_summary(table, factors)
    print(f"\nWrote {config.PROCESSED_DIR / 'traffic.parquet'} and traffic.csv")


if __name__ == "__main__":
    main()

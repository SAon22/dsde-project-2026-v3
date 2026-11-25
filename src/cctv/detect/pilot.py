"""Pilot on 200 frames: choose the YOLO model and confidence threshold.

Steps:
  python -m cctv.detect.pilot sample    # pick 200 frames, create the hand-count sheet
  (fill in data/interim/pilot/hand_counts.csv, rules in HAND_COUNT_RULES.md)
  python -m cctv.detect.pilot compare   # compare YOLO with your counts

Draw ROIs for your cameras BEFORE `sample`: the pilot stores each camera's ROI at that
moment, so your counts and YOLO's are always measured on exactly the same area.
"""

import argparse
import json
import random
import shutil

import pandas as pd
from PIL import Image, ImageDraw

from cctv import config
from cctv.detect.counting import ALL_COUNTED, MIN_BOX_HEIGHT, VEHICLE_CLASSES, measure_all_frames
from cctv.detect.frames import load_usable_frames
from cctv.detect.roi import get_roi, load_rois
from cctv.detect.yolo_runner import run_yolo

PILOT_DIR = config.INTERIM_DIR / "pilot"
NUMBER_OF_CAMERAS = 20
FRAMES_PER_CAMERA = {"day": 6, "dusk": 2, "night": 2}  # 10 per camera = 200 frames
HAND_COUNT_PER_CAMERA = {"day": 3, "dusk": 1, "night": 1}  # 5 per camera = 100 to count by hand
CONFIDENCES_TO_TRY = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
BOX_COLOURS = {"car": "lime", "motorcycle": "cyan", "bus": "orange", "truck": "magenta", "person": "white"}

RULES = """# Hand-count rules

Open each image in `overlays/` (shown at 2x size, the counting area outlined in red).
Count only vehicles INSIDE the red outline: the middle of the vehicle's bottom edge,
where it touches the road, must be inside.

Count a vehicle only if BOTH are true:
1. You can tell what type it is.
2. About half or more of it is visible (not hidden behind another vehicle or cut off
   by the image edge).

Classes:
- car: sedans, SUVs, pickups, taxis, vans, tuk-tuks, songthaews
- motorcycle: motorcycles, scooters and bicycles, with or without a rider
- bus: city buses, coaches, minibuses
- truck: goods vehicles bigger than a pickup (6+ wheels)
- person: pedestrians only. A rider on a motorcycle is NOT a person.

Fill every class column with a number (0 if none). Leave the whole row blank to skip
a frame. Put 1 in `unsure` if the image was hard (glare, rain, blur).
"""


# ----------------------------------------------------------------- sample


def choose_cameras(frames):
    """20 cameras spread from busiest to quietest (uses the quick pass ranking).

    Without the ranking we pick 20 at random. Mixing busy and quiet roads makes the
    pilot test YOLO on both crowded and empty scenes.
    """
    all_cameras = sorted(frames["camera_id"].unique())
    ranking_file = config.INTERIM_DIR / "camera_ranking.csv"
    if not ranking_file.exists():
        return sorted(random.sample(all_cameras, min(NUMBER_OF_CAMERAS, len(all_cameras))))
    ranked = [c for c in pd.read_csv(ranking_file)["camera_id"] if c in all_cameras]
    if len(ranked) <= NUMBER_OF_CAMERAS:
        return ranked
    # Take every k-th camera down the ranking, e.g. ranks 1, 6, 11, ... 100.
    step = (len(ranked) - 1) / (NUMBER_OF_CAMERAS - 1)
    return [ranked[round(i * step)] for i in range(NUMBER_OF_CAMERAS)]


def draw_overlay(image_path, save_path, roi, boxes=None):
    """Save a 2x-size copy of the image with the ROI in red (and boxes, if given)."""
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    image = image.resize((width * 2, height * 2))
    draw = ImageDraw.Draw(image)
    corners = [(x * width * 2, y * height * 2) for x, y in roi]
    draw.line(corners + [corners[0]], fill="red", width=2)
    for box in boxes or []:
        colour = BOX_COLOURS.get(box["cls_name"], "white")
        draw.rectangle([box["x1"] * 2, box["y1"] * 2, box["x2"] * 2, box["y2"] * 2], outline=colour, width=2)
        draw.text((box["x1"] * 2 + 2, box["y1"] * 2), f"{box['cls_name']} {box['conf']:.2f}", fill=colour)
    image.save(save_path, quality=92)


def sample(force):
    """Pick the pilot frames, draw overlays and write an empty hand-count sheet."""
    sheet_path = PILOT_DIR / "hand_counts.csv"
    if sheet_path.exists() and not force:
        raise SystemExit(f"{sheet_path} exists. Use --force to start over (your counts will be lost).")

    random.seed(42)  # same picks every time you run it
    frames = load_usable_frames()
    rois = load_rois()

    picked = []
    for camera_id in choose_cameras(frames):
        for light, how_many in FRAMES_PER_CAMERA.items():
            options = frames[(frames["camera_id"] == camera_id) & (frames["light"] == light)]
            chosen = options.sample(n=min(how_many, len(options)), random_state=42)
            chosen = chosen.sort_values("captured_at").copy()
            # The first few of each group go on the hand-count sheet.
            number_to_count = HAND_COUNT_PER_CAMERA[light]
            chosen["hand_count"] = [i < number_to_count for i in range(len(chosen))]
            picked.append(chosen)
    pilot = pd.concat(picked, ignore_index=True)
    pilot.insert(0, "pilot_id", [f"p{i:03d}" for i in range(1, len(pilot) + 1)])
    pilot["roi_json"] = [json.dumps(get_roi(c, rois)) for c in pilot["camera_id"]]
    pilot["roi_is_fallback"] = [c not in rois for c in pilot["camera_id"]]

    overlay_dir = PILOT_DIR / "overlays"
    shutil.rmtree(overlay_dir, ignore_errors=True)
    overlay_dir.mkdir(parents=True)
    for row in pilot.to_dict("records"):
        draw_overlay(config.IMAGES_DIR / row["rel_path"], overlay_dir / f"{row['pilot_id']}.jpg", json.loads(row["roi_json"]))

    columns = ["pilot_id", "rel_path", "camera_id", "captured_at", "light", "width", "height",
               "hand_count", "roi_is_fallback", "roi_json"]
    pilot[columns].to_csv(PILOT_DIR / "frames.csv", index=False, encoding="utf-8-sig")

    sheet = pilot[pilot["hand_count"]][["pilot_id", "camera_id", "captured_at", "light"]].copy()
    for column in ALL_COUNTED + ["unsure", "notes"]:
        sheet[column] = ""
    sheet.to_csv(sheet_path, index=False, encoding="utf-8-sig")
    (PILOT_DIR / "HAND_COUNT_RULES.md").write_text(RULES, encoding="utf-8")

    fallback_cameras = pilot.drop_duplicates("camera_id")["roi_is_fallback"].sum()
    print(f"Picked {len(pilot)} frames from {pilot['camera_id'].nunique()} cameras.")
    print(f"{len(sheet)} frames to count by hand: {sheet_path}")
    print(f"{fallback_cameras} of the pilot cameras still use the bottom-60% fallback ROI.")


# ----------------------------------------------------------------- compare


def read_hand_counts():
    """Read the filled rows of hand_counts.csv (rows with every class filled in)."""
    hand = pd.read_csv(PILOT_DIR / "hand_counts.csv", encoding="utf-8-sig")
    hand = hand.dropna(subset=ALL_COUNTED)
    for column in ALL_COUNTED:
        hand[column] = hand[column].astype(int)
    hand["vehicles"] = hand[VEHICLE_CLASSES].sum(axis=1)
    hand["unsure"] = hand["unsure"].fillna(0).astype(int)
    return hand


def score(predicted, hand):
    """Compare YOLO counts with hand counts, overall and per light level.

    MAE (mean absolute error) = on average, how many objects YOLO is off per frame.
    Bias = average of (YOLO - hand): negative means YOLO misses objects,
    positive means it counts things that are not there.
    """
    both = hand.merge(predicted[["pilot_id"] + ALL_COUNTED + ["vehicles"]], on="pilot_id", suffixes=("_hand", "_yolo"))
    groups = [("all", both)] + [(light, part) for light, part in both.groupby("light")]
    rows = []
    for light, part in groups:
        row = {"light": light, "frames": len(part), "hand_vehicles_per_frame": part["vehicles_hand"].mean()}
        for column in ["vehicles"] + ALL_COUNTED:
            difference = part[f"{column}_yolo"] - part[f"{column}_hand"]
            row[f"mae_{column}"] = difference.abs().mean()
            row[f"bias_{column}"] = difference.mean()
        rows.append(row)
    return pd.DataFrame(rows)


def compare(models, min_box_height):
    """Score every model x confidence, print the best, and save details to look at."""
    frames = pd.read_csv(PILOT_DIR / "frames.csv", encoding="utf-8-sig")
    hand = read_hand_counts()
    if hand.empty:
        raise SystemExit("No filled rows in hand_counts.csv yet.")
    print(f"Using {len(hand)} hand-counted frames")

    # Use the ROI stored at sampling time, so it matches what you counted on.
    rois = {}
    for row in frames.drop_duplicates("camera_id").to_dict("records"):
        rois[int(row["camera_id"])] = json.loads(row["roi_json"])

    scores = []
    for model in models:
        boxes = run_yolo(frames, model, run_name="pilot")
        for confidence in CONFIDENCES_TO_TRY:
            predicted = measure_all_frames(frames, boxes, rois, confidence, min_box_height)
            result = score(predicted, hand)
            result.insert(0, "confidence", confidence)
            result.insert(0, "model", model)
            scores.append(result)
    scores = pd.concat(scores, ignore_index=True).round(3)
    scores.to_csv(PILOT_DIR / "compare_summary.csv", index=False)

    overall = scores[scores["light"] == "all"].sort_values("mae_vehicles")
    print("\nBest settings first (MAE = vehicles YOLO is off by, per frame):")
    print(overall[["model", "confidence", "mae_vehicles", "bias_vehicles", "mae_car", "mae_motorcycle", "mae_person"]]
          .head(8).to_string(index=False))

    best_model, best_confidence = overall.iloc[0]["model"], overall.iloc[0]["confidence"]
    best_rows = scores[(scores["model"] == best_model) & (scores["confidence"] == best_confidence)]
    print(f"\nBest: {best_model} at confidence {best_confidence}. By time of day:")
    print(best_rows[["light", "frames", "hand_vehicles_per_frame", "mae_vehicles", "bias_vehicles"]].to_string(index=False))

    save_best_details(frames, hand, rois, best_model, best_confidence, min_box_height)


def save_best_details(frames, hand, rois, model, confidence, min_box_height):
    """For the best setting: a per-frame table (worst first) and images with boxes drawn."""
    boxes = run_yolo(frames, model, run_name="pilot")
    predicted = measure_all_frames(frames, boxes, rois, confidence, min_box_height)
    detail = hand.merge(predicted[["pilot_id", "vehicles", "congestion"]], on="pilot_id", suffixes=("_hand", "_yolo"))
    detail["error"] = detail["vehicles_yolo"] - detail["vehicles_hand"]
    detail["abs_error"] = detail["error"].abs()
    detail = detail.sort_values("abs_error", ascending=False)
    detail.to_csv(PILOT_DIR / "compare_frames.csv", index=False, encoding="utf-8-sig")

    out_dir = PILOT_DIR / "predictions"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir()
    kept = boxes[(boxes["conf"] >= confidence) & (boxes["y2"] - boxes["y1"] >= min_box_height)]
    for row in frames[frames["pilot_id"].isin(hand["pilot_id"])].to_dict("records"):
        frame_boxes = kept[kept["rel_path"] == row["rel_path"]].to_dict("records")
        draw_overlay(config.IMAGES_DIR / row["rel_path"], out_dir / f"{row['pilot_id']}.jpg",
                     rois[int(row["camera_id"])], frame_boxes)
    print(f"\nSaved compare_summary.csv, compare_frames.csv (worst frames first) and predictions/ in {PILOT_DIR}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["sample", "compare"])
    parser.add_argument("--force", action="store_true", help="sample: start over even if a sheet exists")
    parser.add_argument("--models", nargs="+", default=["yolo11s.pt", "yolo11m.pt"])
    parser.add_argument("--min-box-height", type=float, default=MIN_BOX_HEIGHT)
    args = parser.parse_args()
    if args.step == "sample":
        sample(args.force)
    else:
        compare(args.models, args.min_box_height)


if __name__ == "__main__":
    main()

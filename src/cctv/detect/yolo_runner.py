"""Run YOLO on frames and save every detected box.

Quick first pass (ranks cameras by traffic, to know which ROIs to draw first):
  python -m cctv.detect.yolo_runner --model yolo11s.pt

Boxes are saved with a low confidence cut-off (0.05) so the real threshold can be
chosen later in the pilot without running YOLO again.
"""

import argparse
import time

import pandas as pd

from cctv import config
from cctv.detect.counting import DEFAULT_CONFIDENCE, YOLO_CLASS_NAMES, measure_all_frames
from cctv.detect.frames import load_usable_frames, one_frame_per_hour
from cctv.detect.roi import load_rois

DETECT_DIR = config.INTERIM_DIR / "detect"
SAVE_CONFIDENCE = 0.05  # keep almost every box; we filter later
IMAGE_SIZE = 640  # YOLO upscales our 352x288 frames to this, which helps with small vehicles
CHUNK_SIZE = 1000  # frames per saved file; if the run stops, finished chunks are kept


def detect_boxes(model, image_paths):
    """Run YOLO on a list of image paths and return a list of box dicts."""
    boxes = []
    results = model.predict(
        [str(config.IMAGES_DIR / p) for p in image_paths],
        imgsz=IMAGE_SIZE,
        conf=SAVE_CONFIDENCE,
        classes=list(YOLO_CLASS_NAMES),  # only people and vehicles
        verbose=False,
    )
    for rel_path, result in zip(image_paths, results):
        # xyxy = box corners in pixels of the ORIGINAL image (YOLO scales them back for us)
        corners = result.boxes.xyxy.tolist()
        classes = result.boxes.cls.tolist()
        scores = result.boxes.conf.tolist()
        for (x1, y1, x2, y2), cls, score in zip(corners, classes, scores):
            boxes.append(
                {"rel_path": rel_path, "cls_name": YOLO_CLASS_NAMES[int(cls)], "conf": score,
                 "x1": x1, "y1": y1, "x2": x2, "y2": y2}
            )
    return boxes


def run_yolo(frames, model_name, run_name, batch_size=32):
    """Detect boxes on all `frames` and return them as a table. Can resume.

    Results go to data/interim/detect/<run_name>/<model>/. Every CHUNK_SIZE frames we
    save a part file and add the finished frames to done.txt, so if the run is
    stopped, running the same command again skips what is already done.
    """
    from ultralytics import YOLO  # imported here: only needed when we actually run YOLO

    out_dir = DETECT_DIR / run_name / model_name.replace(".pt", "")
    out_dir.mkdir(parents=True, exist_ok=True)
    done_file = out_dir / "done.txt"

    already_done = set()
    if done_file.exists():
        already_done = set(done_file.read_text(encoding="utf-8").splitlines())
    to_do = [p for p in frames["rel_path"] if p not in already_done]
    print(f"[{model_name}] {len(already_done)} frames already done, {len(to_do)} to go")

    if to_do:
        model = YOLO(model_name)  # downloads the weights the first time
        start_time = time.time()
        part_number = len(list(out_dir.glob("part_*.parquet")))
        for chunk_start in range(0, len(to_do), CHUNK_SIZE):
            chunk = to_do[chunk_start : chunk_start + CHUNK_SIZE]
            chunk_boxes = []
            for batch_start in range(0, len(chunk), batch_size):
                chunk_boxes += detect_boxes(model, chunk[batch_start : batch_start + batch_size])

            columns = ["rel_path", "cls_name", "conf", "x1", "y1", "x2", "y2"]
            pd.DataFrame(chunk_boxes, columns=columns).to_parquet(out_dir / f"part_{part_number:04d}.parquet")
            part_number += 1
            with open(done_file, "a", encoding="utf-8") as f:
                f.write("\n".join(chunk) + "\n")

            finished = chunk_start + len(chunk)
            speed = finished / (time.time() - start_time)
            print(f"  {finished}/{len(to_do)} frames ({speed:.0f} frames per second)")

    # Read back every part file and keep the boxes for the frames we were asked about.
    parts = [pd.read_parquet(p) for p in sorted(out_dir.glob("part_*.parquet"))]
    if not parts:
        return pd.DataFrame(columns=["rel_path", "cls_name", "conf", "x1", "y1", "x2", "y2"])
    all_boxes = pd.concat(parts, ignore_index=True)
    return all_boxes[all_boxes["rel_path"].isin(set(frames["rel_path"]))]


def rank_cameras(measured):
    """Average vehicles per frame for each camera, busiest first, with its location."""
    ranking = measured.groupby("camera_id").agg(
        frames=("rel_path", "count"),
        mean_vehicles=("vehicles", "mean"),
        mean_congestion=("congestion", "mean"),
    )
    day = measured[measured["light"] == "day"].groupby("camera_id")["vehicles"].mean()
    night = measured[measured["light"] == "night"].groupby("camera_id")["vehicles"].mean()
    ranking["day_vehicles"] = day
    ranking["night_vehicles"] = night
    ranking = ranking.reset_index().sort_values("mean_vehicles", ascending=False)
    ranking.insert(0, "rank", range(1, len(ranking) + 1))

    if config.CAMERA_INFO_CSV.exists():
        info = pd.read_csv(config.CAMERA_INFO_CSV)[["camera_id", "location_en", "latitude", "longitude"]]
        ranking = ranking.merge(info, on="camera_id", how="left")
    return ranking.round(3)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--batch", type=int, default=32, help="lower this if the GPU runs out of memory")
    args = parser.parse_args()

    frames = one_frame_per_hour(load_usable_frames())
    print(f"Quick pass on {len(frames):,} frames (1 per camera per hour)")
    boxes = run_yolo(frames, args.model, run_name="quick", batch_size=args.batch)

    measured = measure_all_frames(frames, boxes, load_rois(), confidence=DEFAULT_CONFIDENCE)
    ranking = rank_cameras(measured)
    out_path = config.INTERIM_DIR / "camera_ranking.csv"
    ranking.to_csv(out_path, index=False, encoding="utf-8-sig")  # utf-8-sig so Excel shows Thai correctly
    print(ranking.head(15).to_string(index=False))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

"""Turn YOLO boxes into numbers per frame: vehicle counts and a congestion level.

YOLO is run once and every box is saved with its confidence score (see yolo_runner.py).
The functions here then apply the filters, so we can try different confidence
thresholds without running YOLO again.
"""

import pandas as pd

from cctv.detect.roi import get_roi, make_roi_mask

# YOLO models trained on the COCO dataset use these class numbers.
YOLO_CLASS_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck"]
ALL_COUNTED = ["car", "motorcycle", "bus", "truck", "person"]

DEFAULT_CONFIDENCE = 0.25  # boxes below this score are ignored (tuned in the pilot)
MIN_BOX_HEIGHT = 6  # pixels; smaller boxes are far-away specks
RIDER_OVERLAP = 0.3  # a person box overlapping a motorcycle this much is its rider


def time_of_day(timestamp):
    """Label a timestamp as "day", "dusk" or "night".

    We use the clock, not image brightness, because in this dataset brightness
    barely separates day from night (median grey about 114 by day vs 96 at night).
    Assumption: sunset in Bangkok in late August is about 18:35.
    """
    minutes = timestamp.hour * 60 + timestamp.minute
    if minutes < 18 * 60 + 15:
        return "day"
    if minutes < 19 * 60:
        return "dusk"
    return "night"


def overlap_ratio(box_a, box_b):
    """How much of box_a is covered by box_b, from 0 (not at all) to 1 (fully).

    Boxes are (x1, y1, x2, y2) with (x1, y1) the top-left corner.
    """
    overlap_width = min(box_a[2], box_b[2]) - max(box_a[0], box_b[0])
    overlap_height = min(box_a[3], box_b[3]) - max(box_a[1], box_b[1])
    if overlap_width <= 0 or overlap_height <= 0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    return (overlap_width * overlap_height) / area_a


def is_rider(person_box, two_wheeler_boxes):
    """True if this person sits on a motorcycle or bicycle (so not a pedestrian)."""
    for wheel_box in two_wheeler_boxes:
        if overlap_ratio(person_box, wheel_box) >= RIDER_OVERLAP:
            return True
    return False


def measure_frame(frame_boxes, roi_mask):
    """Count objects and measure congestion for ONE frame.

    frame_boxes: list of dicts with keys cls_name, x1, y1, x2, y2 (already filtered
    by confidence and size). roi_mask: True/False grid from make_roi_mask().

    A box is "in the ROI" when the middle of its bottom edge is inside. That point
    is where the vehicle touches the road, so it works even when the top of a tall
    bus sticks out of the ROI.
    """
    height, width = roi_mask.shape
    inside = []
    for box in frame_boxes:
        bottom_middle_x = int(min(max((box["x1"] + box["x2"]) / 2, 0), width - 1))
        bottom_middle_y = int(min(max(box["y2"] - 1, 0), height - 1))
        if roi_mask[bottom_middle_y, bottom_middle_x]:
            inside.append(box)

    result = {name: 0 for name in ALL_COUNTED}
    two_wheelers = [(b["x1"], b["y1"], b["x2"], b["y2"]) for b in inside if b["cls_name"] in ("motorcycle", "bicycle")]
    for box in inside:
        name = box["cls_name"]
        if name in VEHICLE_CLASSES:
            result[name] += 1
        elif name == "person" and not is_rider((box["x1"], box["y1"], box["x2"], box["y2"]), two_wheelers):
            result["person"] += 1
    result["vehicles"] = sum(result[name] for name in VEHICLE_CLASSES)

    # Congestion = share of the ROI covered by vehicle boxes.
    # We paint every vehicle box onto an empty grid. Where boxes overlap, the pixel
    # is just painted twice, so overlapping cars are not counted double (a "union").
    covered = roi_mask.copy()
    covered[:, :] = False
    for box in inside:
        if box["cls_name"] in VEHICLE_CLASSES:
            top, bottom = int(max(box["y1"], 0)), int(min(box["y2"], height))
            left, right = int(max(box["x1"], 0)), int(min(box["x2"], width))
            covered[top:bottom, left:right] = True
    roi_pixels = roi_mask.sum()
    covered_roi_pixels = (covered & roi_mask).sum()
    result["congestion"] = float(covered_roi_pixels / roi_pixels) if roi_pixels > 0 else 0.0
    return result


def measure_all_frames(frames, boxes, rois, confidence=DEFAULT_CONFIDENCE, min_box_height=MIN_BOX_HEIGHT):
    """Run measure_frame() on every frame and return one row per frame.

    frames: table with rel_path, camera_id, width, height (other columns are kept).
    boxes:  table with rel_path, cls_name, conf, x1, y1, x2, y2.
    Frames where YOLO found nothing still get a row, with zeros.
    """
    # 1. Keep only confident boxes that are big enough.
    box_height = boxes["y2"] - boxes["y1"]
    good_boxes = boxes[(boxes["conf"] >= confidence) & (box_height >= min_box_height)]

    # 2. Group boxes by image so each frame can find its own boxes quickly.
    boxes_by_frame = {}
    for row in good_boxes.to_dict("records"):
        boxes_by_frame.setdefault(row["rel_path"], []).append(row)

    # 3. Measure each frame. ROI masks are reused per camera and image size.
    masks = {}
    rows = []
    for frame in frames.to_dict("records"):
        camera_id, width, height = int(frame["camera_id"]), int(frame["width"]), int(frame["height"])
        key = (camera_id, width, height)
        if key not in masks:
            masks[key] = make_roi_mask(get_roi(camera_id, rois), width, height)
        measured = measure_frame(boxes_by_frame.get(frame["rel_path"], []), masks[key])
        rows.append({**frame, **measured, "roi_is_fallback": camera_id not in rois})
    return pd.DataFrame(rows)

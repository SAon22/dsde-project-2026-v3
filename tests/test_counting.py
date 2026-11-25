import numpy as np
import pandas as pd

from cctv.detect.counting import is_rider, measure_all_frames, measure_frame, overlap_ratio, time_of_day
from cctv.detect.roi import FALLBACK_ROI, make_roi_mask


def box(name, x1, y1, x2, y2, conf=0.9):
    return {"cls_name": name, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf}


def test_overlap_ratio():
    assert overlap_ratio((0, 0, 10, 10), (5, 0, 20, 10)) == 0.5
    assert overlap_ratio((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_rider_is_not_a_pedestrian():
    motorcycle = (10, 20, 30, 40)
    assert is_rider((12, 10, 28, 35), [motorcycle])
    assert not is_rider((60, 10, 70, 35), [motorcycle])


def test_time_of_day():
    assert time_of_day(pd.Timestamp("2026-08-25 12:00")) == "day"
    assert time_of_day(pd.Timestamp("2026-08-25 18:30")) == "dusk"
    assert time_of_day(pd.Timestamp("2026-08-25 20:50")) == "night"


def test_fallback_mask_is_bottom_60_percent():
    mask = make_roi_mask(FALLBACK_ROI, 100, 100)
    assert not mask[30, 50] and mask[70, 50]
    assert abs(mask.mean() - 0.6) < 0.03


def test_measure_frame_counts_and_congestion():
    mask = make_roi_mask(FALLBACK_ROI, 100, 100)  # rows 40-99 are the ROI
    boxes = [
        box("car", 0, 50, 20, 70),  # inside
        box("car", 10, 50, 30, 70),  # inside, overlaps the first car
        box("car", 0, 0, 20, 20),  # bottom edge at y=20: outside the ROI
        box("motorcycle", 50, 80, 60, 95),
        box("person", 50, 70, 60, 90),  # rider on the motorcycle
        box("person", 80, 80, 85, 95),  # pedestrian
    ]
    result = measure_frame(boxes, mask)
    assert result["car"] == 2 and result["motorcycle"] == 1 and result["vehicles"] == 3
    assert result["person"] == 1
    # union of cars = 30x20 = 600 px, motorcycle 10x15 = 150 px, ROI = 100x60 = 6000 px
    assert abs(result["congestion"] - 750 / mask.sum()) < 1e-9


def test_measure_all_frames_filters_and_keeps_empty_frames():
    frames = pd.DataFrame({"rel_path": ["a.jpg", "b.jpg"], "camera_id": [1, 1], "width": [100, 100], "height": [100, 100]})
    boxes = pd.DataFrame([
        {"rel_path": "a.jpg", **box("car", 0, 50, 20, 70, conf=0.9)},
        {"rel_path": "a.jpg", **box("car", 40, 50, 60, 70, conf=0.1)},  # too unsure
        {"rel_path": "a.jpg", **box("car", 70, 60, 72, 63, conf=0.9)},  # 3 px tall: speck
    ])
    out = measure_all_frames(frames, boxes, rois={}, confidence=0.25)
    assert out["vehicles"].tolist() == [1, 0]
    assert out["roi_is_fallback"].all()

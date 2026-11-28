import pandas as pd

from cctv.detect.build_traffic import build_table, correction_factors


def test_correction_factors():
    summary = pd.DataFrame([
        {"model": "yolo11m.pt", "confidence": 0.20, "light": "all", "hand_vehicles_per_frame": 5.0, "bias_vehicles": -0.5},
        {"model": "yolo11m.pt", "confidence": 0.20, "light": "night", "hand_vehicles_per_frame": 4.35, "bias_vehicles": -1.05},
        {"model": "yolo11m.pt", "confidence": 0.20, "light": "day", "hand_vehicles_per_frame": 5.0, "bias_vehicles": 0.0},
        {"model": "yolo11s.pt", "confidence": 0.20, "light": "night", "hand_vehicles_per_frame": 4.0, "bias_vehicles": -2.0},
    ])
    factors = correction_factors(summary, "yolo11m.pt", 0.20)
    assert set(factors) == {"night", "day"}
    assert abs(factors["night"] - 4.35 / 3.30) < 1e-9
    assert factors["day"] == 1.0


def test_missing_slots_stay_empty_not_zero():
    coverage = pd.DataFrame({
        "camera_id": [1, 1, 1],
        "slot": pd.to_datetime(["2026-08-25 12:00", "2026-08-25 12:10", "2026-08-25 20:00"]),
        "status": ["ok", "missing", "ok"],
        "rel_path": ["1/a.jpg", None, "1/b.jpg"],
    })
    measured = pd.DataFrame({
        "rel_path": ["1/a.jpg", "1/b.jpg"],
        "car": [3, 1], "motorcycle": [1, 0], "bus": [0, 0], "truck": [0, 0], "person": [0, 0],
        "vehicles": [4, 1], "congestion": [0.2, 0.05],
    })
    cameras = pd.DataFrame({"camera_id": [1], "location": ["x"], "location_en": ["X"],
                            "location_detail_en": ["d"], "latitude": [13.7], "longitude": [100.5]})
    table = build_table(coverage, measured, cameras, {"day": 1.0, "dusk": 1.2, "night": 1.5})

    assert len(table) == 3
    assert pd.isna(table.loc[1, "vehicles"])  # missing slot: empty, not 0
    assert table.loc[0, "vehicles_adjusted"] == 4  # day: factor 1
    assert table.loc[2, "light"] == "night" and table.loc[2, "vehicles_adjusted"] == 1.5
    assert table.loc[0, "weekday"] == "Tuesday" and table.loc[0, "latitude"] == 13.7

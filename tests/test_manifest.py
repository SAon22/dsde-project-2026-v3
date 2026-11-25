from datetime import datetime

import pandas as pd
import pytest
from PIL import Image

from cctv.ingest.manifest import add_quality_flags, build_coverage, parse_filename, scan_images


@pytest.mark.parametrize(
    "name, cam, ts",
    [
        ("1028_แยกสะพานแดง_แยกสะพานแดง_20260825_061012.jpg", 1028, datetime(2026, 8, 25, 6, 10, 12)),
        # label containing "_" and "-" must not confuse the parser
        ("1039_ใกล้สะพาน-คลอง_รถมุ่ง_หน้า_20260826_205009.jpg", 1039, datetime(2026, 8, 26, 20, 50, 9)),
        ("11_x_20260831_210000.JPG", 11, datetime(2026, 8, 31, 21, 0, 0)),
    ],
)
def test_parse_filename_ok(name, cam, ts):
    camera_id, label, timestamp = parse_filename(name)
    assert camera_id == cam and timestamp == ts


@pytest.mark.parametrize("name", ["thumbs.db", "1028_x_20260825.jpg", "1028_x_20261399_061012.jpg"])
def test_parse_filename_rejects(name):
    assert parse_filename(name) is None


def _save(path, size, colour, quality=90):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, colour).save(path, "JPEG", quality=quality)


@pytest.fixture
def images_dir(tmp_path):
    d = tmp_path / "cctv_images"
    # Real-looking frames need >5 KB, so add noise.
    import random

    def noisy(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (352, 288))
        img.putdata([(random.randrange(256),) * 3 for _ in range(352 * 288)])
        img.save(path, "JPEG", quality=90)

    noisy(d / "1028" / "1028_a_b_20260825_060012.jpg")
    noisy(d / "1028" / "1028_a_b_20260825_061012.jpg")
    _save(d / "1028" / "1028_a_b_20260825_062012.jpg", (400, 266), "white")  # placeholder
    noisy(d / "1028" / "1028_a_b_20260825_062512.jpg")  # second file in 06:20 slot
    # 06:30 .. 21:00 missing on purpose
    return d


def test_manifest_and_coverage(images_dir):
    m = add_quality_flags(scan_images(images_dir))
    assert len(m) == 4
    assert m["parse_ok"].all() and m["read_ok"].all()
    assert m["is_placeholder"].sum() == 1
    assert m["is_duplicate_slot"].sum() == 1
    assert m["is_valid"].sum() == 2

    cov = build_coverage(m)
    assert len(cov) == 91  # 06:00..21:00 every 10 min on one day
    counts = cov["status"].value_counts().to_dict()
    assert counts == {"missing": 88, "ok": 2, "placeholder": 1}


def test_pixels_mode_flags_frozen(images_dir):
    # copy a frame into the next slot -> identical hash -> frozen
    src = images_dir / "1028" / "1028_a_b_20260825_061012.jpg"
    (images_dir / "1028" / "1028_a_b_20260825_063012.jpg").write_bytes(src.read_bytes())
    m = add_quality_flags(scan_images(images_dir, with_pixels=True))
    assert "mean_grey" in m and m["is_frozen"].sum() == 1 and m["is_dark"].sum() == 0

"""Regions of interest (ROI): the part of each camera's image where we count vehicles.

An ROI is a polygon saved as a list of [x, y] corners. The numbers are fractions of
the image size (0 = left/top, 1 = right/bottom), so one ROI works for any image size.
All ROIs live in configs/roi.json, keyed by camera id.
"""

import json

import numpy as np
from PIL import Image, ImageDraw

from cctv import config

ROI_FILE = config.PROJECT_ROOT / "configs" / "roi.json"

# Used for any camera that has no ROI yet: the bottom 60% of the image,
# because far-away road at the top is too small to count reliably.
FALLBACK_ROI = [[0.0, 0.4], [1.0, 0.4], [1.0, 1.0], [0.0, 1.0]]


def load_rois():
    """Read configs/roi.json and return {camera_id: polygon}. Empty if no file yet."""
    if not ROI_FILE.exists():
        return {}
    data = json.loads(ROI_FILE.read_text(encoding="utf-8"))
    rois = {}
    for camera_id, polygon in data.items():
        rois[int(camera_id)] = polygon
    return rois


def save_rois(rois):
    """Write {camera_id: polygon} to configs/roi.json (overwrites the file)."""
    ROI_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    for camera_id in sorted(rois):
        data[str(camera_id)] = [[round(x, 4), round(y, 4)] for x, y in rois[camera_id]]
    ROI_FILE.write_text(json.dumps(data, indent=1), encoding="utf-8")


def get_roi(camera_id, rois):
    """Return the camera's polygon, or the fallback polygon if none was drawn."""
    return rois.get(camera_id, FALLBACK_ROI)


def make_roi_mask(polygon, width, height):
    """Turn a polygon into a True/False grid the size of the image.

    mask[y, x] is True when pixel (x, y) is inside the ROI. We draw the polygon
    onto a blank image with PIL and convert it to a numpy array, which is simpler
    than doing point-in-polygon maths ourselves.
    """
    corners_in_pixels = [(x * width, y * height) for x, y in polygon]
    image = Image.new("L", (width, height), 0)  # black image
    ImageDraw.Draw(image).polygon(corners_in_pixels, fill=1)  # paint the ROI with 1
    return np.array(image) == 1

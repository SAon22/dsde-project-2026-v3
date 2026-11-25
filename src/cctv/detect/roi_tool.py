"""Click out the counting area (ROI) for each camera.

Run:
  python -m cctv.detect.roi_tool                   # cameras without an ROI, busiest first
  python -m cctv.detect.roi_tool --cameras 395 1188

In the window:
  left click   add a corner
  right click  remove the last corner
  Enter        save this camera (Enter with no corners = skip it)
  close window stop; everything saved so far stays in configs/roi.json

What to draw: the road near the camera, where vehicles are big enough to recognise.
Leave out far-away road, sidewalks and parking bays.
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from cctv import config
from cctv.detect.frames import load_usable_frames
from cctv.detect.roi import FALLBACK_ROI, load_rois, save_rois


def pick_example_frame(frames, camera_id):
    """Pick a weekday frame around noon: good light and normal traffic."""
    camera_frames = frames[frames["camera_id"] == camera_id]
    noon = camera_frames[(camera_frames["captured_at"].dt.hour == 12) & (camera_frames["captured_at"].dt.dayofweek < 5)]
    if noon.empty:
        noon = camera_frames
    return noon.iloc[len(noon) // 2]


def cameras_in_order(frames):
    """Busiest cameras first if the quick pass ranking exists, else by camera id."""
    all_cameras = sorted(frames["camera_id"].unique())
    ranking_file = config.INTERIM_DIR / "camera_ranking.csv"
    if not ranking_file.exists():
        return all_cameras
    ranked = pd.read_csv(ranking_file)["camera_id"].tolist()
    not_ranked = [c for c in all_cameras if c not in ranked]
    return ranked + not_ranked


def show_camera(image, camera_id, current_roi, title):
    """Open a window with the frame and the current ROI drawn as a red dashed line."""
    width, height = image.size
    figure = plt.figure(figsize=(10, 10 * height / width))
    plt.imshow(image)
    xs = [x * width for x, y in current_roi] + [current_roi[0][0] * width]
    ys = [y * height for x, y in current_roi] + [current_roi[0][1] * height]
    plt.plot(xs, ys, "r--")
    plt.title(title, fontsize=9)
    plt.axis("off")
    plt.tight_layout()
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cameras", type=int, nargs="*", help="only these cameras (also redraws existing ROIs)")
    args = parser.parse_args()

    frames = load_usable_frames()
    rois = load_rois()
    info = pd.read_csv(config.CAMERA_INFO_CSV).set_index("camera_id")

    if args.cameras:
        cameras = args.cameras
    else:
        cameras = [c for c in cameras_in_order(frames) if c not in rois]
    print(f"{len(rois)} cameras already have an ROI, {len(cameras)} to draw.")

    for number, camera_id in enumerate(cameras, start=1):
        frame = pick_example_frame(frames, camera_id)
        image = Image.open(config.IMAGES_DIR / frame["rel_path"])
        width, height = image.size
        title = f"[{number}/{len(cameras)}] camera {camera_id}: {info.at[camera_id, 'location_en']}"
        figure = show_camera(image, camera_id, rois.get(camera_id, FALLBACK_ROI), title)

        # ginput waits for clicks: left adds a point, right removes the last, Enter finishes.
        clicks = plt.ginput(n=-1, timeout=0, mouse_add=1, mouse_pop=3, mouse_stop=None)
        if not plt.fignum_exists(figure.number):
            print("Window closed, stopping.")
            break
        plt.close(figure)

        if len(clicks) == 0:
            print(f"camera {camera_id}: skipped")
            continue
        if len(clicks) < 3:
            print(f"camera {camera_id}: an area needs at least 3 corners, skipped")
            continue

        # Save corners as fractions of the image size (0 to 1).
        rois[camera_id] = [[x / width, y / height] for x, y in clicks]
        save_rois(rois)
        print(f"camera {camera_id}: saved ({len(rois)} cameras done)")


if __name__ == "__main__":
    main()

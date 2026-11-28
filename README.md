# Bangkok CCTV Image Analysis

Course project for 2110403 Data Science and Data Engineering (2026/1).
It turns raw BMA traffic-camera images (100 cameras, 25-31 Aug 2026) into structured data,
analyses spatial and temporal traffic patterns, and presents them in a dashboard.

## Pipeline

| Stage | Folder | Output |
| --- | --- | --- |
| 1. Ingest + quality flags | `src/cctv/ingest` | `data/interim/manifest.parquet`, `coverage.parquet` |
| 2. Clean metadata | `src/cctv/clean` | cleaned camera table |
| 3. Vehicle detection (YOLO) | `src/cctv/detect` | per-frame vehicle counts |
| 4. External data (scraping) | `src/cctv/scrape` | `data/external/` |
| 5. Features + joins | `src/cctv/features` | final tables for analysis |
| Dashboard | `dashboard/` | Streamlit app |
| Workflow | `pipelines/` | end-to-end DAG |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
pip install -e .
```

Copy the dataset into `data/raw/` (see `data/README.md`). The data folder is not tracked by Git.

## Run

```bash
python -m cctv.ingest.manifest            # fast: file names + image headers
python -m cctv.ingest.manifest --pixels   # also brightness and frozen-frame detection
pytest
```

## Vehicle detection (Step 1: pilot)

Install PyTorch with CUDA first (check your CUDA version with `nvidia-smi`), then the rest:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

```bash
python -m cctv.detect.yolo_runner        # quick pass: rank cameras by traffic
python -m cctv.detect.roi_tool           # draw the counting area per camera
python -m cctv.detect.pilot sample       # pick 200 frames + hand-count sheet
python -m cctv.detect.pilot compare      # YOLO vs hand counts
```

## Vehicle detection (Steps 2 and 3: full run and final table)

```bash
python -m cctv.detect.yolo_runner --full   # yolo11m on every usable frame (~20 min on GPU, resumable)
python -m cctv.detect.build_traffic        # -> data/processed/traffic.parquet and traffic.csv
```

Pilot result (99 hand-counted frames, yolo11m at confidence 0.20): average error 1.8 vehicles per
frame, bias -0.25 overall, about -24% at night. `vehicles_adjusted` scales counts by the pilot's
day / dusk / night factor; `vehicles` is the raw YOLO count.

| File | What it does |
| --- | --- |
| `detect/roi.py` | load/save the counting area (ROI) per camera, fallback = bottom 60% |
| `detect/counting.py` | turn YOLO boxes into counts and congestion for each frame |
| `detect/frames.py` | load usable frames from the manifest, pick 1 per hour |
| `detect/yolo_runner.py` | run YOLO and save boxes (resumable), quick camera ranking |
| `detect/roi_tool.py` | click tool to draw ROIs |
| `detect/pilot.py` | sample pilot frames, compare YOLO with hand counts |
| `detect/build_traffic.py` | final table: counts per camera per 10-minute slot, with night correction |

## Known data issues

- Some frames are blank white 400x266 placeholders (~1.4 KB) from failed captures, mostly 07:20-08:50.
- On 27 Aug about 15 frames are missing on most cameras (about 07:20-08:50).
- The on-screen clock on some cameras is wrong; use the timestamp in the file name.
- Missing or placeholder slots are "no data", never zero traffic (see `coverage.parquet`).

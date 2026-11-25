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

## Known data issues

- Some frames are blank white 400x266 placeholders (~1.4 KB) from failed captures, mostly 07:20-08:50.
- On 27 Aug about 15 frames are missing on most cameras (about 07:20-08:50).
- The on-screen clock on some cameras is wrong; use the timestamp in the file name.
- Missing or placeholder slots are "no data", never zero traffic (see `coverage.parquet`).

# data/

Not tracked by Git. Put the course dataset here so the layout is:

```
data/
├── raw/                 # never modified by code
│   ├── cctv_info.csv
│   └── cctv_images/<camera_id>/<camera_id>_<location>_<detail>_<YYYYMMDD>_<HHMMSS>.jpg
├── interim/             # manifest.parquet, coverage.parquet
├── processed/           # detections and final tables
└── external/            # scraped data
```

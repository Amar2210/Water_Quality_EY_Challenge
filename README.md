# EY Water Quality Challenge 2026

Predict 3 water quality parameters (Total Alkalinity, Electrical Conductance, Dissolved Reactive Phosphorus) for South African river locations. Core difficulty: validation locations are in a different geographic region than training — spatial generalization. See `experiment_logs/experiment_log.md` for full version history.

## Project Structure

```
EY_Challenge/
├── ey_challenge_notebooks/  # Original EY-provided files (DON'T MODIFY)
├── data/
│   ├── raw/README.md        # Pointer only (originals live in ey_challenge_notebooks/)
│   └── extracted/           # Generated features, LOCAL ONLY (*.csv git-ignored, *_feature_columns.txt tracked)
├── scripts/                 # Full pipeline, kept as history (01 → 24, see table below)
├── notebooks/               # Empty placeholder for future work (work currently in scripts/)
├── submissions/             # One canonical CSV per version (timestamped dupes removed after hash check)
├── best_submission/         # Author-designated best: submission_v9.csv
├── experiment_logs/         # Append-only experiment log + per-version results
├── requirements.txt
├── setup.bat                # Windows one-click setup (creates venv/, installs requirements)
└── README.md
```

## Quick Start (Windows)

```bat
setup.bat                  :: creates venv\ and installs requirements.txt
venv\Scripts\activate      :: future sessions
python scripts\01_analyze_data.py
```

Regenerate the current-best feature set locally (data stays local, not pushed):

```bat
python scripts\20_feature_engineering_v9.py
python scripts\21_model_v9.py
```

## Pipeline (scripts/ kept as full history)

| Stage | Scripts |
|---|---|
| Analysis | `01_analyze_data.py`, `check_landsat_quality.py` |
| Extraction | `03_extract_features.py`, `07_extract_slope.py`, `15_extract_landcover.py`, `15_extract_worldcover.py`, `16_extract_geology.py`, `extract_landsat_full.py`, `extract_sentinel2.py` |
| Feature engineering | `08_feature_engineering_v4.py`, `10_feature_engineering_v5.py`, `12_feature_engineering_v6.py`, `18_feature_engineering_v8.py`, `20_feature_engineering_v9.py` |
| Models | `02_model_v0.py`, `02_model_v0.1.py`, `04_model_v1.py`, `05_model_v2.py`, `06_model_v3.py`, `09_model_v4.py`, `11_model_v5.py`, `13_model_v6.py`, `14_model_v7_ec.py`, `17_model_v7b_ec.py`, `19_model_v8.py`, `21_model_v9.py`, `22_model_v10.py`, `23_model_v11.py`, `24_model_v12_image_only.py`, `06_ensemble_v1v2.py` |

Notes: `02_model_v0` vs `02_model_v0.1`, `06_model_v3` vs `06_ensemble_v1v2`, and `15_landcover` vs `15_worldcover` are intentional kept variants. Best submission is **v9** (`best_submission/submission_v9.csv`); v10/v11 in `submissions/` are newer attempts — compare `experiment_logs/` before promoting.

## Data note (why repo looks small)

`data/extracted/*.csv` (~125MB) and vendor CSVs are **git-ignored** (see `.gitignore`) to keep the public repo slim and under GitHub's 100MB-per-file block. Column lists (`*_feature_columns.txt`), code, submissions, and logs ARE tracked. Your local CSVs are untouched — this only affects what gets pushed.

## No license (yet)

Public with no `LICENSE` = all rights reserved (visible, not reusable). Adding MIT later is free and takes a minute when you want reuse.

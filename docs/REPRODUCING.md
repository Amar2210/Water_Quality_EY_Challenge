# Running and reproducing the historical workflow

This guide reproduces the **V9 research workflow**, including its known evaluation limitation. It does not promise the saved CSV will be bit-identical under a different software stack or after re-extracting data from live APIs. Read [REVIEW.md](REVIEW.md) before interpreting scores.

## 1. Quick integrity checks

From the project root, with any supported Python installation:

```bat
python scripts\validate_submission.py
python -m unittest discover -s tests -v
```

These require only the standard library and tracked files. The validator defaults to the saved reference and the EY submission template. To check a regenerated output:

```bat
python scripts\validate_submission.py submissions\submission_v9.csv
```

The tests deliberately do not import training/extraction scripts: many execute work at import time, and some connect to remote services immediately.

## 2. Environment

For a new environment using `requirements.txt`, prefer **Python 3.11 or 3.12**. The retained scikit-learn 1.5.2 pin predates Python 3.13 support. The existing local environment is Python 3.13 with newer packages; it is not an exact installation of the requirements file.

Windows:

```bat
setup.bat
venv\Scripts\activate
```

Other platforms:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For exact future reproducibility, record `python --version` and save `python -m pip freeze` alongside that experiment. The full setup has not been freshly installed as part of this cleanup.

## 3. Original data

Place the authorized EY-provided CSVs in `ey_challenge_notebooks/`:

- `water_quality_training_dataset.csv`
- `landsat_features_training.csv`
- `landsat_features_validation.csv`
- `terraclimate_features_training.csv`
- `terraclimate_features_validation.csv`
- `submission_template.csv` — already tracked

The four original EY notebooks and their requirements file are unchanged. The extraction notebooks can regenerate their Landsat/climate feature CSVs when run with that folder as their working directory; the water-quality labels must come from the challenge distribution, not those notebooks.

The main V8 → V9 feature path uses the original Landsat CSVs plus expanded extraction caches. The original TerraClimate PET CSVs are also used by the initial analysis and earlier experiments.

## 4. Choose the shortest available restart point

### A. V9 feature caches already exist

Required:

```text
data/extracted/v9_train_features.csv
data/extracted/v9_val_features.csv
data/extracted/v9_feature_columns.txt
```

Run:

```bat
python scripts\21_model_v9.py
python scripts\validate_submission.py submissions\submission_v9.csv
```

### B. V8 feature caches exist, but V9 does not

```bat
python scripts\20_feature_engineering_v9.py
python scripts\21_model_v9.py
```

### C. Extraction caches exist, but feature tables do not

V8 requires these local files in `data/extracted/`:

| Source | Required cache files |
|---|---|
| Full Landsat | `landsat_full_training.csv`, `landsat_full_validation.csv` |
| Expanded TerraClimate | `terraclimate_expanded_training.csv`, `terraclimate_expanded_validation.csv` |
| Elevation | `elevation_data.csv` |
| Soil | `soil_data.csv` |
| Land cover | `landcover_data.csv` |
| Geology/proxies | `geology_data.csv` |
| Terrain derivatives | `slope_data.csv` — optional in code, required to match the saved feature set |

Then run:

```bat
python scripts\18_feature_engineering_v8.py
python scripts\20_feature_engineering_v9.py
python scripts\21_model_v9.py
```

**V8 is built directly from extraction caches, not from the V6 feature table.** There is no need to run every historical numbered script.

### D. No extraction caches exist

After obtaining the original data and installing dependencies:

```bat
python scripts\03_extract_features.py
python scripts\07_extract_slope.py
python scripts\extract_landsat_full.py
python scripts\check_landsat_quality.py
python scripts\15_extract_landcover.py
python scripts\16_extract_geology.py
python scripts\18_feature_engineering_v8.py
python scripts\20_feature_engineering_v9.py
python scripts\21_model_v9.py
python scripts\validate_submission.py submissions\submission_v9.csv
```

Dependencies and outputs:

```text
EY location/date/target CSVs
    ├─ 03_extract_features ─── TerraClimate + elevation + soil
    ├─ 07_extract_slope ────── terrain derivatives
    ├─ extract_landsat_full ── Landsat bands/indices
    ├─ 15_extract_landcover ── surrounding land cover
    └─ 16_extract_geology ──── geology + soil-derived proxies (needs soil cache)
                  ↓
    Original EY Landsat fallback + extraction caches
                  ↓
    18_feature_engineering_v8 ── environmental feature tables
                  ↓
    20_feature_engineering_v9 ── historical neighbor features
                  ↓
    21_model_v9 ── generated submission + timestamped result log
```

This is the historical extraction order, **not a guarantee that current external responses are usable**. Check source/year provenance, soil-unit conversions, and lithology classifications before modeling. Live extraction depends on Planetary Computer, Open-Elevation, ISRIC/SoilGrids, and Macrostrat availability. Landsat can take hours. Some extractors resume existing files, while others overwrite caches; back up valuable local CSVs before re-extracting.

## 5. Outputs and promotion

- Feature-building scripts overwrite their version's local feature tables and tracked column-list files.
- V9 training writes `submissions/submission_v9.csv` and a timestamped result CSV in `experiment_logs/`.
- It does **not** overwrite `best_submission/submission_v9.csv`.
- An ignore rule prevents regenerated V9 outputs from being added after the tracked duplicate is removed from the Git index. Review and commit the cleanup before rerunning: `.gitignore` does not untrack a file still present in the index.
- Promote a new reference only after verifying evaluation, file integrity, and any official score. Keep the old reference and its metadata recoverable.

All scripts resolve data paths from their own file location, rather than a personal Windows path. Examples assume the project-root working directory. On other platforms, use `/` instead of `\` in script paths.

## 6. Historical alternatives

See [`scripts/archive/README.md`](../scripts/archive/README.md). Archived scripts retain experimental algorithms and known limitations; relocation/path fixes are not a claim that they are correct or fully tested. Some overwrite older submission outputs or emit timestamped copies. Sentinel-2 additionally requires `stackstac`, which is not needed by the active pipeline.

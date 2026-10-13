# EY Water Quality Challenge 2026

**Predicting river-water chemistry in South Africa using satellite imagery and open environmental data.**

The challenge was to predict three measurements at a given river location and date:

| Measurement | In plain English |
|---|---|
| **Total Alkalinity** | How well the water can neutralize acids. |
| **Electrical Conductance** | How easily the water carries electricity, which reflects its dissolved ions. |
| **Dissolved Reactive Phosphorus** | A readily available form of phosphorus that can contribute to algal growth. |

These are the **outputs to predict**, not three input features. Training inputs include location, date, and features extracted from open datasets. This is a regression project, not a drinking-water safety classifier.

## What makes this project interesting

The work grew from an imagery-based baseline into an environmental modeling pipeline: Landsat bands and temperature, rainfall and drought history, soil properties, terrain, and surrounding land cover. Later experiments also explored geology and predictions anchored to measurements at other stations.

The real challenge was **generalizing to new places**. The local data contains **9,319 labeled samples from 162 stations (2011–2015)** and **200 prediction rows at 24 different locations**. Those prediction locations are approximately **82–282 km from the nearest training station**. A model that recognizes familiar stations is not necessarily useful in the new region.

**Read the [plain-English project summary](docs/PROJECT_SUMMARY.md) for the full story.**

## Current status — an honest research baseline

- **Saved reference:** [`best_submission/submission_v9.csv`](best_submission/submission_v9.csv), chosen by the author as the restart baseline. It has 200 complete prediction rows in the template's order.
- **V9 method:** combine environmental ML with distance-weighted neighboring-station estimates and optional residual correction.
- **Important limitation:** V9's neighbor features were built using the full training set before cross-validation. Excluding a station's own measurements does **not** exclude the held-out region's measurements. Its reported CV gains are not leakage-free evidence of cross-region accuracy.
- **Newer is not automatically better:** V10's proper regional holdouts were negative; V11's saved results were also negative. V12 has code but no saved results or submission in this checkout.
- **No independently verified V9 leaderboard score is stored here.** “Best” means author-designated, not a proven ranking across all versions.

See the [review and cleanup report](docs/REVIEW.md) for results, data-quality issues, and the remaining work. The original [experiment log](experiment_logs/experiment_log.md) is preserved as history, with a close-out correction appended.

## Repository layout

```text
EY_Challenge/
├── scripts/                  # Data checks, extraction, V8 → V9 feature building, V9 model
│   └── archive/              # 20 earlier/alternative experiments; not needed for V9
├── ey_challenge_notebooks/   # Original EY notebooks + locally supplied CSVs; unchanged
├── data/extracted/           # Local extraction/feature caches; CSVs are git-ignored
├── best_submission/          # One tracked V9 reference; never auto-overwritten
├── submissions/              # Earlier version outputs; regenerated V9 stays local
├── experiment_logs/          # Historical notes and recorded results
├── docs/                     # Plain-English summary, reproduction guide, review
├── tests/                    # Offline integrity tests
├── requirements.txt
└── setup.bat
```

Unique experiments were **archived, not destroyed**. Only the hash-verified extra V9 submission copy and unused placeholder/pointer files were deleted. See the [archive index](scripts/archive/README.md).

## Quick checks — no ML installation needed

From the project folder:

```bat
python scripts\validate_submission.py
python -m unittest discover -s tests -v
```

The validator checks columns, row count, location/date ordering, and finite non-negative predictions. It does **not** measure accuracy.

## Run the historical V9 workflow

For a fresh installation of the pinned dependencies, use **Python 3.11 or 3.12**:

```bat
setup.bat
venv\Scripts\activate
python scripts\01_analyze_data.py
```

If the local V9 feature caches already exist:

```bat
python scripts\21_model_v9.py
python scripts\validate_submission.py submissions\submission_v9.csv
```

If rebuilding from existing extraction caches:

```bat
python scripts\18_feature_engineering_v8.py
python scripts\20_feature_engineering_v9.py
python scripts\21_model_v9.py
```

**V8 is the prerequisite for V9**; V4/V5/V6 feature tables are not required. The model writes `submissions/submission_v9.csv`, not the protected reference in `best_submission/`.

A fresh clone does **not** contain the training CSV or generated feature CSVs. Obtain the EY-provided files through the challenge's authorized distribution, then follow the [reproduction guide](docs/REPRODUCING.md) for the complete extraction order. Live extraction can take hours and depends on external APIs.

## To finish the modeling side properly

1. Rebuild target-derived neighbor features **inside each spatial training fold**; fit preprocessing there too.
2. Correct geology parsing and SoilGrids units before claiming those features worked.
3. Compare simple mean, environmental-only, and interpolation baselines using the same untouched regional holdouts.
4. Record the final submission, actual leaderboard result if available, and exact package versions together.

The strongest takeaway is the end-to-end geospatial research and the discovery that **spatial evaluation and data quality matter more than adding another model**.

No project license has been added. Original EY material and external datasets retain their own terms.

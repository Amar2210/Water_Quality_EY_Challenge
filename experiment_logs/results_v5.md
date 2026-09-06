# V5 Results — V4 + Sentinel-2

**Date**: 2026-02-20
**Script**: `scripts/11_model_v5.py`
**Feature Engineering**: `scripts/10_feature_engineering_v5.py`

---

## Approach
- Added Sentinel-2 bands (higher spatial resolution, red-edge bands)
- Hypothesis: S2 captures water quality signals Landsat misses

## Results
- **Spatial CV Mean R²: ~-0.35** (no improvement over V4)
- **Sentinel-2 data was 97.5% NaN** — extraction failed

## Key Takeaway
S2 extraction was essentially useless due to cloud cover / revisit timing / extraction issues. Always check data availability BEFORE investing time in feature engineering. No point pursuing S2 further with current extraction approach.

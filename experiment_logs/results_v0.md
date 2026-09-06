# V0 / V0.1 Results — Baseline XGBoost

**Date**: 2026-02-14
**Script**: `scripts/02_model_v0.py`, `scripts/02_model_v0.1.py`

---

## Approach
- XGBoost with existing EY-provided Landsat bands + PET
- Added temporal features (month, season, sin/cos encoding)
- Added derived spectral features (NIR/Green ratio, SWIR ratio, brightness)
- Log transforms on DRP and EC
- Spatial CV with 8 spatial clusters (GroupKFold)

## Results
- **Spatial CV Mean R²: ~-0.60** (all targets negative)
- Not submitted to leaderboard

## Key Takeaway
Landsat spectral features alone are insufficient for spatial generalization. Need location-characterizing features.

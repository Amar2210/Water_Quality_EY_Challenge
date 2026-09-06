# V1 Results — Expanded Climate + Elevation + Soil

**Date**: 2026-02-15
**Script**: `scripts/04_model_v1.py`
**Feature Engineering**: `scripts/03_extract_features.py`

---

## Approach
- Added full TerraClimate (temp, precip, VPD, soil moisture, etc.)
- Added SRTM elevation + slope
- Added SoilGrids (clay, sand, SOC, pH, CEC, nitrogen)
- XGBoost with multiple regularization configs
- Still included coordinate features (Lat, Lon)

## Results

| Target | Best R² |
|--------|---------|
| Total Alkalinity | ~-0.46 |
| Electrical Conductance | ~-0.43 |
| Dissolved Reactive Phosphorus | ~-0.31 |
| **MEAN** | **~-0.40** |

**Leaderboard: +0.006** (first positive submission!)

## Key Takeaway
Soil and elevation features improved predictions significantly. However, coordinates were still dominating feature importance — model was partially memorizing locations, which happened to help on leaderboard because some validation locations were near training locations.

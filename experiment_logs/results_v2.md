# V2 Results — Coordinate Removal + Regularization

**Date**: 2026-02-15
**Script**: `scripts/05_model_v2.py`

---

## Approach
- REMOVED all coordinate features (Lat, Lon, lat_abs)
- Added Ridge regression and ElasticNet
- Stronger XGBoost regularization (max_depth=3, high penalties)
- Added climate×soil interaction features

## Results

| Target | Best R² |
|--------|---------|
| Total Alkalinity | -0.3038 |
| Electrical Conductance | -0.4225 |
| Dissolved Reactive Phosphorus | -0.1900 |
| **MEAN** | **-0.3054** |

**Leaderboard: -0.087** (worse than V1 despite better spatial CV!)

## Key Takeaway — THE PARADOX
Better spatial CV (-0.30 vs -0.40) but worse leaderboard (-0.087 vs +0.006). Removing coordinates removed the "neighborhood proximity" signal that helped V1 on the leaderboard. The validation region happened to have locations close enough to training locations that coordinate-based memorization was useful. This means we need BOTH spatial generalization AND some form of location awareness.

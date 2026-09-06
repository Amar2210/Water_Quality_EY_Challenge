# V4 Results — Research-Backed Feature Engineering

**Date**: 2026-02-18
**Script**: `scripts/09_model_v4.py`
**Feature Engineering**: `scripts/08_feature_engineering_v4.py`

---

## Approach
- Research paper-driven features: slope×clay (erosion), terrain roughness, elevation bins
- Climate aggregates (3-month rolling sums/means)
- Soil-derived: sand/clay ratio, buffer capacity (CEC×SOC), weathering index
- Physics-based interactions
- Log-transformed elevation

## Results
- **Spatial CV Mean R²: ~-0.35**
- Leaderboard: TBD

## Key Takeaway
Marginal improvement. Research-backed features are physically meaningful but feature engineering alone has diminishing returns. Terrain/soil features consistently dominate importance rankings across all versions.

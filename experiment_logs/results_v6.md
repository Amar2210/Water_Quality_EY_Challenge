# V6 Results — Full Landsat Bands + Thermal + V4 Climate/Terrain/Soil

**Date**: 2026-02-20
**Script**: `scripts/13_model_v6.py`
**Feature Engineering**: `scripts/12_feature_engineering_v6.py`

---

## Data Summary

| Metric | Train | Val |
|--------|-------|-----|
| Rows | 9,319 | 200 |
| Columns | 179 | 176 |
| Available features | 173 | 173 |
| Full Landsat coverage | 85.7% | 87.5% |
| Surface temp coverage | 32.0% | 45.5% |

---

## Feature Tiers

| Tier | Name | Feature Count |
|------|------|--------------|
| T1 | CORE | 38 |
| T2 | EXTENDED | 73 |
| T3 | FULL | 151 |

---

## Spatial Clustering (5 clusters)

| Cluster | Samples | Center Lat | Center Lon |
|---------|---------|------------|------------|
| 0 | 3,479 | -26.5°S | 27.7°E |
| 1 | 1,472 | -33.1°S | 20.1°E |
| 2 | 1,929 | -29.5°S | 30.2°E |
| 3 | 1,726 | -29.0°S | 25.7°E |
| 4 | 713 | -24.4°S | 30.9°E |

---

## Target 1: Total Alkalinity

**Skewness**: 0.54 → no log transform

| Rank | Config | Mean R² | Fold Scores |
|------|--------|---------|-------------|
| 1 ★ | T1_CORE_elasticnet | -0.2399 | [-0.027, -0.689, -0.047, -0.527, +0.090] |
| 2 | T3_FULL_aggressive | -0.2713 | [-0.312, -0.583, -0.055, -0.442, +0.035] |
| 3 | T3_FULL_moderate | -0.3136 | [-0.278, -0.609, -0.013, -0.732, +0.064] |
| 4 | T3_FULL_conservative | -0.4119 | [-0.237, -0.725, -0.121, -0.762, -0.214] |
| 5 | T1_CORE_ridge | -0.4379 | [-0.003, -0.900, -0.035, -1.204, -0.048] |

**Note**: Fold 4 (cluster 4, 713 samples) actually achieves **positive R²** (+0.090) — this cluster may have feature distributions closer to what the model can learn.

---

## Target 2: Electrical Conductance

**Skewness**: 0.93 → no log transform

| Rank | Config | Mean R² | Fold Scores |
|------|--------|---------|-------------|
| 1 ★ | T3_FULL_aggressive | -0.4960 | [-0.490, -1.681, -0.016, -0.174, -0.118] |
| 2 | T3_FULL_conservative | -0.5227 | [-0.461, -1.697, -0.009, -0.235, -0.212] |
| 3 | T3_FULL_moderate | -0.5297 | [-0.495, -1.736, -0.054, -0.199, -0.164] |

**Fold 1 (cluster 1, 1472 samples at -33.1°S, 20.1°E) is the disaster fold** — always R² < -1.4. This suggests the Western Cape / Cape Town region has very different EC characteristics from the rest of the training data.

### Top 15 Features (T3_FULL_aggressive)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | soil_soc | 0.1253 |
| 2 | soil_buffer_capacity | 0.0611 |
| 3 | elev_range_500m | 0.0535 |
| 4 | terrain_roughness | 0.0411 |
| 5 | soil_sand | 0.0403 |
| 6 | slope_deg | 0.0399 |
| 7 | log_elevation | 0.0361 |
| 8 | elevation_m | 0.0281 |
| 9 | soil_nitrogen | 0.0272 |
| 10 | soil_sand_clay_ratio | 0.0264 |
| 11 | soil_clay | 0.0257 |
| 12 | elev_bin | 0.0249 |
| 13 | slope_x_clay | 0.0226 |
| 14 | ppt_3m_sum | 0.0211 |
| 15 | soil_phh2o | 0.0188 |

**Observation**: Zero spectral/Landsat features in top 15 for EC. Entirely terrain + soil + climate.

---

## Target 3: Dissolved Reactive Phosphorus

**Skewness**: 1.64 → log transform applied

| Rank | Config | Mean R² | Fold Scores |
|------|--------|---------|-------------|
| 1 ★ | T1_CORE_aggressive_log | -0.2015 | [-0.785, -0.063, +0.023, -0.072, -0.110] |
| 2 | T1_CORE_conservative_log | -0.2091 | [-0.790, -0.040, +0.015, -0.119, -0.112] |
| 3 | T3_FULL_conservative_log | -0.2123 | [-0.745, +0.067, -0.169, -0.143, -0.072] |

**Note**: Log transform is critical for DRP — non-log configs score around -0.7 (3x worse).
**Fold 0 (cluster 0, 3479 samples) is the disaster fold** — always R² < -0.7.

### Top 15 Features (T1_CORE_aggressive_log)

| Rank | Feature | Importance | Note |
|------|---------|------------|------|
| 1 | elevation_m | 0.1208 | |
| 2 | soil_sand | 0.0871 | |
| 3 | soil_soc | 0.0828 | |
| 4 | slope_deg | 0.0696 | |
| 5 | soil_phh2o | 0.0576 | |
| 6 | soil_cec | 0.0531 | |
| 7 | has_surface_temp | 0.0451 | |
| 8 | soil_clay | 0.0431 | |
| 9 | ls_st_kelvin | 0.0213 | NEW in V6 |
| 10 | unified_MNDWI | 0.0198 | NEW in V6 |
| 11 | ey_MNDWI | 0.0198 | |
| 12 | ls_MNDWI | 0.0178 | NEW in V6 |
| 13 | ey_swir16 | 0.0177 | |
| 14 | vpd | 0.0169 | |
| 15 | ey_nir | 0.0168 | |

**Observation**: New V6 features (surface temp, unified MNDWI) do appear but with low importance (~0.02). DRP also terrain/soil dominated.

---

## Target Stacking: EC → DRP

| Config | R² |
|--------|-----|
| DRP without stacking | -0.2015 |
| DRP with EC stacking | -0.2591 |
| **Delta** | **-0.0576 (worse)** |

Stacking did not help — skipped.

---

## Final Summary

| Target | Best Config | R² |
|--------|-------------|-----|
| Total Alkalinity | T1_CORE_elasticnet | -0.2399 |
| Electrical Conductance | T3_FULL_aggressive | -0.4960 |
| Dissolved Reactive Phosphorus | T1_CORE_aggressive_log | -0.2015 |
| **MEAN** | | **-0.3125** |

### Version Comparison (Spatial CV Mean R²)

| Version | Mean R² | Delta |
|---------|---------|-------|
| V0.1 | ~-0.60 | — |
| V1 | ~-0.40 | +0.20 |
| V2 | ~-0.30 | +0.10 |
| V4 | ~-0.35 | -0.05 |
| V5 | ~-0.35 | 0.00 |
| **V6** | **-0.3125** | **+0.04** |

---

## Prediction Ranges on Validation Set

| Target | Min | Max | Mean |
|--------|-----|-----|------|
| Total Alkalinity | 0.0 | 155.9 | 70.7 |
| Electrical Conductance | 125.3 | 984.8 | 352.5 |
| Dissolved Reactive Phosphorus | 10.2 | 35.1 | 18.2 |

---

## Submissions Generated

- `submissions/submission_v6.csv`
- `submissions/submission_v6_20260220_172418.csv` (timestamped backup)
- `submissions/submission_ensemble_v6.csv` (ensemble of V1+V2+V4+V5+V6)

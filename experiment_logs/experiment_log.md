# EY Water Quality Challenge — Experiment Log

> **Single append-only document** tracking every strategy, logic, and outcome.
> Each version section is added chronologically. Do NOT overwrite — always append.

---

## Challenge Overview

- **Goal**: Predict 3 water quality parameters (Total Alkalinity, Electrical Conductance, Dissolved Reactive Phosphorus) for river locations in South Africa
- **Metric**: Mean R² across all 3 targets
- **Key difficulty**: Validation locations are in a DIFFERENT geographic region than training data → spatial generalization is the core challenge
- **Leaderboard benchmark**: Top competitor ~0.82 R²
- **Baseline (EY provided)**: ~0.20 R² (in-sample, random split — misleading)

---

## Version History Summary

| Version | Date       | Spatial CV Mean R² | Leaderboard | Key Idea |
|---------|------------|-------------------|-------------|----------|
| V0/V0.1 | 2026-02-14 | ~-0.60           | N/A         | Benchmark XGBoost + existing Landsat/PET |
| V1      | 2026-02-15 | -0.40            | +0.006      | Expanded climate + elevation + soil |
| V2      | 2026-02-15 | -0.30            | -0.087      | Removed coordinates, Ridge/ElasticNet |
| V4      | 2026-02-18 | ~-0.35           | **~+0.04** ★ BEST | Research-backed climate/terrain/soil interactions |
| V5      | 2026-02-20 | ~-0.35           | TBD         | V4 + Sentinel-2 (97.5% NaN — useless) |
| V6      | 2026-02-20 | -0.3125          | TBD         | Full Landsat bands + thermal + hybrid extraction |

---

## V0 / V0.1 — Baseline XGBoost (2026-02-14)

### Logic / Hypothesis
- Start with what EY provided: 6 Landsat bands/indices + PET from TerraClimate
- Add temporal features (month, season, year, sin/cos encoding)
- Add derived spectral features (NIR/Green ratio, SWIR ratio, brightness, pseudo-NDVI, bare soil index)
- Use XGBoost instead of RandomForest (handles NaN natively, usually better on tabular data)
- Log-transform skewed targets (DRP skewness 1.64, EC skewness 0.93)
- Honest evaluation via spatial CV (GroupKFold with 8 spatial clusters)

### What Changed vs EY Benchmark
- XGBoost instead of RandomForest
- Log transforms on DRP and EC
- Spatial cross-validation instead of random 70/30 split
- 15+ features from original 4

### Results
- Spatial CV Mean R²: ~-0.60 (all targets negative)
- This was expected — Landsat spectral features alone cannot explain water quality across different geographies

### Learnings
- Negative R² in spatial CV = model is WORSE than predicting the global mean
- Spectral features alone don't capture WHY different locations have different water quality
- Need location-characterizing features (elevation, soil type, geology, land cover)
- The EY benchmark's +0.20 R² was from random split which leaks spatial information

---

## V1 — Expanded Climate + Elevation + Soil (2026-02-15)

### Logic / Hypothesis
- The hypothesis: if we describe each location by WHAT IT IS (soil type, elevation, climate) rather than WHERE IT IS (coordinates), the model should generalize better
- Extract additional features from Planetary Computer: full TerraClimate variables (precipitation, temperature, VPD, soil moisture, etc.), SRTM elevation, SoilGrids properties (clay, sand, SOC, pH, CEC, nitrogen)
- These are "spatially invariant" in the sense that a new location with similar soil/climate/elevation should behave similarly to training locations with those properties

### Features Added
- Full TerraClimate: temperature (min/max), precipitation, VPD, soil moisture, snow water equivalent, wind speed, actual ET, deficit, runoff, PDSI
- SRTM elevation
- SoilGrids: clay %, sand %, SOC, pH (H2O), CEC, nitrogen
- Slope derived from elevation

### Results
- Spatial CV Mean R²: -0.40
- Leaderboard score: **+0.006** (first positive submission!)
- Individual targets: TA ~-0.46, EC ~-0.43, DRP ~-0.31

### Learnings
- Big improvement from V0 (-0.60 → -0.40)
- Despite negative spatial CV, leaderboard was slightly positive — CV is pessimistic because validation region may share more feature similarity than held-out clusters
- Soil and elevation features showed up in top feature importances
- Coordinates (lat/lon) were still in the model and dominating — possible memorization

---

## V2 — Coordinate Removal + Regularization (2026-02-15)

### Logic / Hypothesis
- Coordinates were dominating feature importance — the model was memorizing locations
- Hypothesis: removing coordinates forces the model to learn physical relationships
- Added Ridge regression and ElasticNet (linear models generalize better spatially)
- Stronger XGBoost regularization (max_depth=3, high penalties)
- Added climate×soil interaction features

### What Changed
- REMOVED all coordinate features (Latitude, Longitude, lat_abs)
- Added Ridge and ElasticNet as model options
- Stronger XGBoost regularization configurations
- Climate-soil interaction features

### Results
- Spatial CV Mean R²: -0.30 (improved from V1's -0.40)
- Leaderboard score: **-0.087** (WORSE than V1!)

### Learnings — CRITICAL INSIGHT
- **Paradox**: Better spatial CV but worse leaderboard
- **Explanation**: V1's coordinates gave a weak "neighborhood proximity" signal. Even though it caused overfitting in spatial CV, some test locations were CLOSE enough to training locations that coordinates helped. Removing coordinates removed the only feature giving useful signal for nearby test points.
- **Conclusion**: We need BOTH good generalization AND some spatial signal. The answer isn't removing location info — it's encoding location in a transferable way (through geology, catchment area, watershed features)
- V2 validated that climate/soil/spectral features alone aren't enough
- Linear models (Ridge, ElasticNet) sometimes matched XGBoost → the data may have more linear than nonlinear patterns for spatial transfer

---

## V3 — Skipped / Merged into V4

V3 was planned with geology data, land use (ESA WorldCover), nearest-neighbor blending, and more Landsat bands. The research paper brainstorming session redirected efforts into V4 with a more systematic research-backed approach.

---

## V4 — Research-Backed Feature Engineering (2026-02-18)

### Logic / Hypothesis
- Studied research papers on water quality prediction using remote sensing
- Key paper finding: For phosphate (DRP), the most important features were conductivity, slope, month, and flow rate — accounting for ~85% of variable importance
- Hypothesis: engineer features grounded in hydrology/water chemistry research rather than generic ML features
- Focus on physics-based interactions: slope×soil interactions (erosion proxy), elevation bins (geological context), climate aggregates (3-month rolling means), terrain roughness

### Features Added
- Terrain: slope, aspect, terrain roughness, elevation range (500m window), elevation bins
- Climate aggregates: 3-month rolling sums/means for precipitation, temperature
- Interactions: slope×clay (erosion proxy), slope×SOC, elevation×precipitation
- Soil-derived: sand/clay ratio, soil buffer capacity (CEC×SOC), soil weathering index
- Log-transformed elevation

### Results
- Spatial CV Mean R²: ~-0.35
- **Leaderboard: ~+0.04** ★ BEST LEADERBOARD SCORE SO FAR

### Learnings
- Marginal improvement from more engineered features
- Terrain features (slope, roughness, elevation range) showed moderate importance
- Soil SOC and buffer capacity appeared in top features — physically meaningful
- Diminishing returns on feature engineering with XGBoost
- The problem may be more about model architecture than features

---

## V5 — V4 + Sentinel-2 Bands (2026-02-20)

### Logic / Hypothesis
- Sentinel-2 has higher spatial resolution (10m vs 30m for Landsat) and more bands (red-edge bands useful for chlorophyll/vegetation)
- Hypothesis: S2 data might capture water quality signals that Landsat misses
- Used Planetary Computer to extract S2 data for training and validation locations

### What Changed
- Added Sentinel-2 band extraction (B2-B12, red-edge bands)
- Computed S2-specific indices (NDVI, NDWI, etc.)

### Results
- Spatial CV Mean R²: ~-0.35 (no improvement over V4)
- **Sentinel-2 coverage was 97.5% NaN** — almost no usable data

### Learnings
- **S2 extraction largely failed** — cloud cover, revisit timing, or extraction issues meant virtually no S2 data was available for our date ranges
- No improvement because S2 features were almost entirely missing
- Lesson: check data availability BEFORE building features
- Not worth pursuing S2 further unless extraction methodology is fixed

---

## V6 — Full Landsat Bands + Thermal (2026-02-20)

### Logic / Hypothesis
- Instead of Sentinel-2 (which failed), go deeper with Landsat data
- EY only provided 4 bands — but Landsat has more: Blue, Red, Coastal Aerosol, Surface Temperature
- Blue band: water clarity, turbidity indicators (NSMI, clarity index)
- Red band: real NDVI (instead of pseudo-NDVI), chlorophyll ratio, sediment index
- Surface temperature: correlated with EC (dissolved salts), thermal stratification
- Coastal aerosol: atmospheric scattering (Landsat 8 only, ~35% coverage)
- Hybrid approach: full Landsat extraction as primary + EY-provided bands as fallback

### Features Added (New in V6)
- Landsat Blue band → NSMI, water clarity index, blue/green ratio
- Landsat Red band → true NDVI, chlorophyll ratio (nir/red), sediment index, bare soil index
- Surface temperature (Kelvin) → thermal×climate interactions
- Coastal aerosol band (L8 only)
- Unified spectral indices: MNDWI, NDWI from full Landsat
- `has_surface_temp` flag (binary indicator)
- New interactions: thermal×VPD, thermal×elevation, blue×terrain

### Data Coverage
- Full Landsat coverage: train=85.7%, val=87.5%
- Surface temperature: train=32.0%, val=45.5%
- Total features available: 173

### Model Configurations Tested
- 3 Feature tiers: T1_CORE (38 features), T2_EXTENDED (73), T3_FULL (151)
- XGBoost configs: aggressive, moderate, conservative regularization
- Linear models: Ridge, ElasticNet (T1 only)
- Log vs raw transforms for each target
- Target stacking attempted (EC → DRP)

### Results

**Spatial CV:**

| Target | Best Config | R² |
|--------|-------------|-----|
| Total Alkalinity | T1_CORE_elasticnet | -0.2399 |
| Electrical Conductance | T3_FULL_aggressive | -0.4960 |
| Dissolved Reactive Phosphorus | T1_CORE_aggressive_log | -0.2015 |
| **MEAN** | | **-0.3125** |

**Best ever spatial CV mean: -0.3125** (improved from V4's -0.35)

### Key Feature Importances

**EC (best: T3_FULL_aggressive):**
Top features: soil_soc (0.13), soil_buffer_capacity (0.06), elev_range_500m (0.05), terrain_roughness (0.04), soil_sand (0.04), slope_deg (0.04)
→ Dominated by terrain and soil, NOT spectral bands

**DRP (best: T1_CORE_aggressive_log):**
Top features: elevation_m (0.12), soil_sand (0.09), soil_soc (0.08), slope_deg (0.07), soil_phh2o (0.06)
→ Also dominated by terrain/soil. New Landsat features (surface temp, MNDWI) appeared but with low importance (0.02)

### Learnings
- **V6 achieved best spatial CV so far** (-0.3125 vs previous best -0.30 from V2)
- ElasticNet won for Total Alkalinity → linear relationship between core features and TA
- Full Landsat bands helped marginally but terrain/soil features still dominate
- Surface temperature has only 32% coverage — helps where available but too sparse
- Target stacking (using EC predictions to help DRP) did NOT help (-0.026 worse)
- The feature importance pattern is consistent: **terrain and soil drive everything**
- New Landsat spectral features rank low in importance despite being physically meaningful
- We may be approaching the ceiling of what XGBoost + feature engineering can do with this spatial CV setup

### Submissions Generated
- `submissions/submission_v6.csv`
- `submissions/submission_ensemble_v6.csv` (ensemble of V1+V2+V4+V5+V6)

---

## Persistent Patterns Across All Versions

1. **Spatial CV is always negative** — the model consistently fails to beat the mean when predicting on entirely held-out geographic clusters
2. **Terrain/soil features dominate** — elevation, slope, soil SOC, soil clay/sand consistently rank highest regardless of model version
3. **Spectral (Landsat) features contribute little** — despite being the EY-intended signal, spectral bands/indices rarely appear in top 10 features
4. **Linear models sometimes match XGBoost** — for TA especially, ElasticNet matches or beats XGBoost, suggesting the relationship might be more linear than tree-based
5. **Leaderboard ≠ Spatial CV** — V1 had worse CV but positive leaderboard, V2 had better CV but negative leaderboard. The gap between CV and leaderboard is unpredictable.
6. **Diminishing returns on features** — going from 15 features (V0) to 173 features (V6) improved mean R² by ~0.3, but the last 100+ features added almost nothing

---

## Ideas Not Yet Tried

- [ ] Optuna hyperparameter tuning on winning configs
- [ ] LightGBM / CatBoost as alternative boosting frameworks
- [ ] Neural network approaches (tabular transformers, TabNet)
- [ ] Nearest-neighbor blending (KNN of training targets weighted by feature similarity)
- [ ] Geology data (lithology maps — could be huge for alkalinity)
- [ ] Land use / land cover from ESA WorldCover
- [ ] Watershed/catchment delineation features
- [ ] Water body proximity / river network features
- [ ] Ensemble of diverse model types (linear + tree + KNN)
- [ ] Per-target specialized models with target-specific feature selection
- [ ] Quantile regression for robust predictions
- [ ] Different spatial CV strategy (leave-one-region-out may be too harsh)
- [ ] Coordinate features re-added but with heavy regularization (V1 lesson)

---

## V7 — EC-Focused Model with Research-Backed Seasonal & Drought Features (2026-02-22)

### Research Insights Driving V7

**1. Cool Dry Season (May–October) is optimal for remote sensing in Southern Africa:**
- Atmospheric interference minimized
- Nutrient loading → algal response relationship more predictable
- EC models perform better during dry/low-flow: ionic concentrations higher, more stable
- High-flow events introduce dilution + extreme turbidity that decouples EC-reflectance relationship

**2. El Niño Drought (late 2014 onwards):**
- South Africa experienced El Niño-induced drought starting late 2014
- Declining water levels → increased dissolved ion concentration
- Provides "high-end" extremes of EC and TA
- Stronger spectral signal for ML models during this period

**3. EC-Specific Spectral Science:**
- High EC = high salinity/mineral loads → increased refractive index → enhanced backscatter in SWIR
- NIR and SWIR regions are sensitive to ionic concentrations (co-occurring fine sediments, altered surface tension)
- Temperature is critical: EC is temperature-dependent. Landsat TIRS provides this.
- Prioritize SWIR + NIR bands, plus surface temperature as inputs

**4. Atmospheric Correction:**
- Atmosphere can contribute up to 90% of the signal over water
- Landsat 8 Coastal band (Band 1) enables precise aerosol estimation
- Landsat 7 should use LEDAPS/LaSRC for consistency across 2011–2015

### Logic / Hypothesis
- Build an EC-ONLY model first (not multi-target)
- Use focused ~40 features instead of V6's 173 (feature discipline)
- Add cool_dry_season flag (May–Oct) and drought_period flag (Oct 2014+)
- Create season×spectral and drought×spectral interaction features
- Prioritize SWIR, NIR, surface temperature features (research-backed for EC)
- Add LightGBM alongside XGBoost, Ridge, ElasticNet
- Report RMSE and MAE alongside R² for all configurations
- Log-transform EC as an option (skewness ~0.93)

### New Features Added (V7-specific)
- `is_cool_dry_season`: Binary flag for May–October (optimal remote sensing window)
- `is_drought_period`: Binary flag for samples from October 2014 onwards (El Niño)
- `cool_dry_x_swir16`: Season × SWIR interaction (dry season amplifies mineral signal)
- `cool_dry_x_nir`: Season × NIR interaction
- `cool_dry_x_ndmi`: Season × moisture index interaction
- `drought_x_swir16`: Drought × SWIR (concentrated ions → stronger SWIR)
- `drought_x_ec_proxy`: Drought × aridity (double concentration effect)
- `st_x_ec_climate`: Temperature × VPD (thermal-driven EC variation)
- Multi-window climate: 1-month, 3-month (already in V6), to be used judiciously

### EC Feature Set (~40 features, physically justified)

**Spectral (EC-relevant, SWIR/NIR priority):**
unified_swir16, unified_swir22, unified_nir, unified_green,
ratio_swir16_nir, ratio_swir22_swir16, ratio_nir_swir22,
unified_NDMI, unified_MNDWI, ls_st_celsius, has_surface_temp, ls_NSMI

**Terrain/Soil (consistent top performers across V0–V6):**
elevation_m, slope_deg, terrain_roughness, elev_range_500m,
soil_soc, soil_sand, soil_clay, soil_cec, soil_phh2o, soil_nitrogen,
soil_buffer_capacity, soil_sand_clay_ratio

**Climate:**
ppt, tmax, tmin, soil (moisture), vpd, aet, def,
ppt_3m_sum, tmax_3m_avg, aridity_index

**Temporal/Seasonal:**
month_sin, month_cos, is_cool_dry_season (NEW), is_drought_period (NEW)

**Research-backed interactions:**
st_x_swir16, swir16_x_tmax, slope_x_clay, elev_x_ppt,
cool_dry_x_swir16 (NEW), cool_dry_x_nir (NEW), cool_dry_x_ndmi (NEW),
drought_x_swir16 (NEW), drought_x_aridity (NEW)

### Metrics
- R² (primary, leaderboard metric)
- RMSE (scale-aware error)
- MAE (robust to outliers)
- All reported per spatial CV fold + mean

### Model Configurations
- XGBoost: aggressive, moderate, conservative regularization
- LightGBM: default + tuned (NEW)
- Ridge regression
- ElasticNet
- Both raw and log-transformed EC target

### What Changed vs V6 (Code-Level Detail)

**Feature changes:**
- Shifted from multi-target (TA, EC, DRP) to **EC-only** model
- Reduced from V6's 173 features to 3 disciplined tiers:
  - `EC_CORE`: ~32 features (SWIR/NIR/temperature priority + terrain/soil + climate)
  - `EC_EXTENDED`: ~40 features (adds seasonal/drought flags + interactions)
  - `EC_FULL`: ~55 features (adds climate lags, more spectral indices)
- New features engineered on top of V6 data:
  - `is_cool_dry_season` — binary, months 5–10 (May–Oct)
  - `is_drought_period` — binary, `Sample Date >= 2014-10-01`
  - `cool_dry_x_swir16`, `cool_dry_x_nir`, `cool_dry_x_ndmi`, `cool_dry_x_swir_ratio` — season × spectral
  - `drought_x_swir16`, `drought_x_swir22`, `drought_x_ndmi`, `drought_x_aridity` — drought × spectral
  - `st_celsius_safe` — surface temp with fallback to `tmax` then 25°C default
  - `st_x_vpd`, `st_x_aridity` — temperature × climate interactions
  - `swir_mean`, `swir_diff` — SWIR band aggregates (mineral vs moisture discrimination)
  - `ppt_anomaly` — ratio of current month ppt to 3-month average (anomaly detection)
  - `temp_anomaly` — `tmax - tmax_3m_avg` (departure from seasonal norm)
- Dropped: all Sentinel-2 features, many V6 legacy EY-band ratios, visibility/coastal indices
- `soil_buffer_capacity` redefined as `phh2o × cec` (was `cec × soc` in V6's derivation)

**Data / splits changes:**
- Spatial clusters increased: **8 clusters** (was 5 in V6)
- CV folds increased: **8-fold GroupKFold** (was 5-fold in V6)
- KMeans `n_init=20` (was 10 in V6) — more stable cluster assignments
- Still uses V6's `v6_train_features.csv` and `v6_val_features.csv` as base data

**Target transform:**
- Tests both raw EC and `log1p(EC)` for every configuration
- EC skewness = 0.93 (below 1.0 threshold used in V6, but tested anyway)

**Model changes:**
- Added **LightGBM** (2 configs: `lgbm_default` and `lgbm_conservative`)
- XGBoost configs restructured — renamed and retuned:
  - `xgb_aggressive`: depth=4, lr=0.05, min_child=10, alpha=1, lambda=5 (V6's "moderate" equivalent)
  - `xgb_moderate`: depth=3, lr=0.03, min_child=20, alpha=2, lambda=10 (new, more conservative)
  - `xgb_conservative`: depth=2, lr=0.02, min_child=30, alpha=5, lambda=20 (heavily regularized, new)
- Linear models restricted to `EC_CORE` tier only (code: `'tiers': ['EC_CORE']`)
- Ridge alpha increased from 1.0 to **10.0** (stronger regularization)

**Training / evaluation changes:**
- NaN handling: manual column-mean imputation per fold (was XGBoost native handling in V6)
- StandardScaler applied for linear models per fold
- XGBoost uses `eval_set=[(X_va, y_va)]` (monitors validation loss, enables early stopping awareness)
- LightGBM uses `callbacks=[lgb.log_evaluation(period=0)]` to suppress output
- **RMSE and MAE** reported alongside R² for every configuration (new)
- Top-3 ensemble: trains top 3 configs on full data, averages predictions for validation
- Generates **two submissions**: best single model + top-3 ensemble average

### Results

V7 was **executed** on 2026-02-22 (timestamp `20260222_222509`). Submission files exist:
- `submissions/submission_v7_ec.csv` (best single model)
- `submissions/submission_v7_ec_ensemble.csv` (top-3 ensemble)

The detailed results log (`v7_ec_results_20260222_222509.txt`) has **encoding corruption** — the R², RMSE, MAE values and per-config rankings are unreadable. The file header shows the correct structure but all data rows are missing.

**What we can infer from code:**
- V6's EC result was R² = **-0.4960** (referenced in V7 comparison table)
- V7 tested ~36 configurations: 3 tiers × (3 XGB + 2 LightGBM) × 2 transforms + 2 linear × 1 tier × 2 transforms
- The code's comparison table has a placeholder: `V7 (EC) | {best_res['mean_r2']:+.4f}` — actual value lost

**Results not fully recoverable due to corrupted log file. Must re-run to confirm.**

### Learnings

**Confirmed from code structure (not requiring result values):**
- V7 embraced "feature discipline" — the hypothesis that fewer, physically-justified features outperform kitchen-sink approaches. This is a direct response to V6's finding that terrain/soil dominated importance across 173 features.
- The shift to EC-only modeling acknowledges that each target may need specialized feature engineering. TA and DRP were deferred to V7b and V7c respectively.
- More conservative XGBoost configs were introduced (depth=2, lr=0.02) — suggests V6's aggressive configs may have been overfitting in spatial CV.
- LightGBM was added as an alternative framework. The conservative LightGBM config (`num_leaves=8`) is very shallow, testing whether EC relationships are nearly linear.
- Ridge alpha jumped 10× (from 1.0 to 10.0) — stronger regularization for spatial generalization.
- 8 clusters / 8 folds (vs V6's 5/5) provides more granular spatial evaluation but smaller test sets per fold.

**Learnings requiring re-run to validate:**
- Whether EC_CORE (32 features) truly beats EC_FULL (55 features) — referenced in V7b's docstring as confirmed, but numeric proof is in the corrupted log
- Whether seasonal/drought flags appear in top feature importances
- Whether LightGBM outperforms XGBoost for EC
- Whether log transform helps EC (skewness = 0.93, borderline)

---

## V7b — EC + Land Use + Geology + KNN (2026-02-22)

### Logic / Hypothesis
V7 showed:
- CORE 32 features beats 69+ features (feature discipline works)
- Terrain/soil STILL dominate feature importance
- Seasonal/drought flags (temporal) didn't help spatial generalization

V7b hypothesis: The missing ingredient is **spatial context about human activity and bedrock type**:
- ESA WorldCover land use at 1km/5km/10km buffers (urban, cropland, bare/mining, forest)
- Geology/lithology from Macrostrat API (carbonate vs granite vs sedimentary)
- KNN blending: nearest-neighbor predictions to anchor model outputs

### New Data Extracted
1. **ESA WorldCover 2021** (10m resolution via Planetary Computer)
   - % tree cover, shrubland, grassland, cropland, built-up, bare/sparse, water
   - At 1km, 5km, 10km buffer radii
   - Derived: human_impact index, natural_veg index, urban_ratio

2. **Geology/Lithology** (Macrostrat API)
   - Raw lithology type, classified into: carbonate, siliciclastic, fine_sedimentary,
     felsic_igneous, mafic_igneous, ultramafic, evaporite, volcanic, metamorphic, unconsolidated
   - EC impact score (1-5), Alkalinity impact score (1-5)
   - One-hot encoded for ML
   - Soil-derived geology proxies as fallback

### Feature Tiers
- A_V7_CORE: 32 features (V7 winners, baseline)
- B_CORE_LANDUSE: V7 Core + all land use features
- C_CORE_GEOLOGY: V7 Core + geology features
- D_CORE_ALL: V7 Core + land use + geology
- E_FOCUSED: V7 Core + 5km land use only + geology

### KNN Approach
- KNeighborsRegressor with K=5,10,20,50 tested
- Distance-weighted (closer training points get more influence)
- Features scaled with StandardScaler for proper distance computation
- Hybrid blend: model_weight * model_preds + knn_weight * knn_preds
- Different blend ratios tested (30/70 through 90/10)

### What Changed vs V7 (Code-Level Detail)

**Feature changes:**
- Base feature set is now `V7_CORE` (32 features) — hardcoded as the winners from V7
- New data source: **ESA WorldCover 2021** land use at 3 buffer radii:
  - 6 land cover classes × 3 radii = 18 percentage features: `lc_{1km,5km,10km}_pct_{built_up,cropland,bare_sparse,tree_cover,grassland,water}`
  - 3 derived features × 3 radii = 9 features: `lc_{radius}_{human_impact,natural_veg,urban_ratio}`
  - Total: up to 27 land use features
- New data source: **Geology/lithology** from Macrostrat API:
  - Features prefixed `geo_`, filtered to >10% non-NaN coverage
  - Includes `geo_ec_impact` (EC impact score 1–5), one-hot lithology types
- 5 feature tiers (vs V7's 3):
  - `A_V7_CORE`: 32 features (V7 baseline)
  - `B_CORE_LANDUSE`: V7 Core + all land use features
  - `C_CORE_GEOLOGY`: V7 Core + geology features
  - `D_CORE_ALL`: V7 Core + land use + geology (everything)
  - `E_FOCUSED`: V7 Core + 5km land use only + geology
- V7's seasonal/drought features (`is_cool_dry_season`, `is_drought_period`, interactions) are **NOT re-created** in V7b — they were V7-specific features engineered in `add_v7_features()` which V7b does not call. V7b only uses V6 base features + V7_CORE selection + new land use/geology.

**Data changes:**
- WorldCover data merged via rounded lat/lon (6 decimal places) with intermediate `lat_round`/`lon_round` columns
- Geology data merged similarly, with `drop_duplicates(subset=['lat_round', 'lon_round'])` to prevent row inflation
- Coverage check: features with <10% non-NaN in training data are excluded from geology tier
- **WorldCover data file (`worldcover_data.csv`) does NOT exist** — extraction script `15_extract_worldcover.py` exists but was not run. This means V7b will run without land use features (tiers B, D, E will collapse to A/C equivalents).
- Geology data file **does exist** (`geology_data.csv`)

**Model changes:**
- Dropped `xgb_aggressive` config entirely — only `xgb_conservative` and `xgb_moderate` remain
- Dropped `elasticnet` — only `ridge` kept for linear
- LightGBM kept: `lgbm_conservative` only (no default config)
- **No log transform tested** — V7b only tests raw EC (unlike V7 which tested both)
- **KNN added** as new model type:
  - `KNeighborsRegressor` with `weights='distance'`, `metric='euclidean'`
  - Tested K = 5, 10, 20, 50
  - Features scaled with `StandardScaler` for distance computation
  - Uses `A_V7_CORE` features for KNN distance

**Training / evaluation changes:**
- Same 8-cluster, 8-fold spatial CV as V7
- **Hybrid blending** tested: `model_weight × model_preds + knn_weight × knn_preds`
  - Weights tested: 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 for model (rest for KNN)
  - Blend evaluated on OOF predictions (not per-fold CV — uses full OOF mask)
  - **Note**: Blend R² is computed on the union of non-NaN OOF predictions, not per-fold, so it's not directly comparable to per-fold CV R²
- Generates **3 submissions**: model-only, KNN-only, best-blend
- Feature importance shown for best non-KNN model with top 25 features, tagged `[LAND USE]` or `[GEOLOGY]`

### Results

**V7b was NOT executed.** Evidence:
- No V7b result log files exist in `experiment_logs/`
- No V7b submission files exist in `submissions/`
- WorldCover data (`worldcover_data.csv`) is missing — the script would run but without land use features
- Geology data does exist, so tiers A and C would work

**Results not available. Must run `17_model_v7b_ec.py` after running `15_extract_worldcover.py`.**

### Learnings

**Structural observations from code (not requiring results):**
- V7b represents a shift from "better features" to "better spatial context" — the hypothesis that land use and geology are the missing spatial variables that explain *why* two locations with identical terrain/soil have different EC
- The multi-buffer approach (1km, 5km, 10km) for land use is sound — immediate surroundings vs landscape-scale context may matter differently
- KNN blending is a pragmatic approach: if the model can't generalize spatially, perhaps anchoring to the nearest training points (in feature space) provides a useful prior
- The blend approach evaluates on OOF predictions rather than per-fold CV, which may overestimate blend quality since it doesn't account for fold-level variance
- Dropping log transform is a simplification — V7 tested both but V7b assumed raw is sufficient (possibly based on V7 results showing raw performed similarly)
- V7's seasonal/drought features are silently dropped — `add_v7_features()` is not called in V7b. If V7 showed these helped, they should be re-added.

**Learnings to be validated after re-run:**
- Whether land use features (especially 5km buffer) improve EC prediction
- Whether geology features (lithology type, EC impact score) improve EC prediction
- Whether KNN provides useful signal (positive R² would be significant)
- Whether model+KNN blending outperforms either alone
- Optimal K value for KNN
- Optimal blend ratio

---

## Persistent Patterns Update (Post-V7/V7b)

The following observations extend the earlier "Persistent Patterns" section:

7. **Per-target specialization is the new direction** — V7 broke from the multi-target approach (V0–V6) to build EC-specific models. This acknowledges that EC, TA, and DRP may have fundamentally different feature-target relationships.
8. **Feature discipline > feature quantity** — V7's core thesis (32 focused features vs 173) reflects diminishing returns observed in V6. V7b continues this by testing additive tiers (core → core+landuse → core+geology → all).
9. **Model diversity expanding** — V7 added LightGBM, V7b added KNN. The ensemble approach is shifting from "same model, different features" to "different models, blended predictions."
10. **Data extraction pipeline is a bottleneck** — WorldCover extraction script exists but hasn't been run, blocking V7b. Geology was extracted successfully. Each new data source requires its own extraction → merge → validate pipeline.

---

## Ideas Not Yet Tried (Updated)

- [x] ~~LightGBM / CatBoost~~ — LightGBM added in V7 (CatBoost still untried)
- [x] ~~Per-target specialized models~~ — V7 does EC-only
- [x] ~~Nearest-neighbor blending~~ — V7b adds KNN blending
- [x] ~~Land use / land cover from ESA WorldCover~~ — V7b has code, extraction script exists (not yet run)
- [x] ~~Geology data~~ — Extracted and integrated in V7b
- [ ] Optuna hyperparameter tuning on winning configs
- [ ] CatBoost as alternative boosting framework
- [ ] Neural network approaches (tabular transformers, TabNet)
- [ ] Watershed/catchment delineation features
- [ ] Water body proximity / river network features
- [ ] Quantile regression for robust predictions
- [ ] Different spatial CV strategy (leave-one-region-out may be too harsh)
- [ ] Coordinate features re-added with heavy regularization
- [ ] V7b for TA (Total Alkalinity target-specific model)
- [ ] V7c for DRP (using EC predictions as stacked feature)
- [ ] Re-run V7 with fixed results logging (encoding issue)
- [ ] Run WorldCover extraction before V7b execution

---

## V8 — Land Cover + Geology + All Targets (2026-02-22)

### Logic / Hypothesis

V8 represents a strategic pivot from V7/V7b's EC-only approach back to **all 3 targets**, while incorporating the land cover and geology data that V7b planned but never executed. The core hypotheses:

- **Land cover is the missing spatial context.** Two locations with identical soil/terrain but different land use (urban vs forest vs cropland) should have vastly different water quality. Agricultural runoff drives DRP. Urban areas drive EC. Natural forest = low EC.
- **Geology is the primary natural driver of alkalinity.** Without bedrock type, the model can't distinguish limestone rivers (high alkalinity) from granite rivers (low alkalinity). Geology also affects baseline mineral content → EC.
- **Cross-domain interactions unlock the signal.** It's not just "cropland %" that matters — it's "cropland % × recent rainfall" (fertilizer runoff timing). Similarly, "carbonate rock × rainfall" captures limestone dissolution → alkalinity.
- **V7's seasonal features are worth keeping.** Cool dry season flag and El Niño drought flag are incorporated into the base feature engineering.

### What Changed vs V7/V7b (Code-Level Detail)

**Architecture shift:**
- Returns to **all 3 targets** (TA, EC, DRP) — V7/V7b were EC-only
- Effectively supersedes V7b: incorporates land cover + geology ideas in a unified pipeline
- New feature engineering script (`18_feature_engineering_v8.py`) builds on V6's `12_feature_engineering_v6.py`, NOT on V7's runtime feature additions

**Feature engineering — New data sources:**

1. **Land Cover (ESA CCI 2013, 300m resolution)** — file: `landcover_data.csv`
   - Note: V7b planned to use ESA WorldCover 2021 (10m, `worldcover_data.csv`) but that file never existed. V8 uses the successfully extracted ESA CCI 2013 data instead.
   - Raw percentages at 3 buffer radii: `lc_{1km,5km,10km}_pct_{cropland,forest,shrubland,grassland,sparse_veg,wetland,urban,bare,water,snow_ice}`
   - Derived indices: `lc_{radius}_human_impact`, `lc_{radius}_natural_veg`, `lc_{radius}_urban_ratio`, `lc_{radius}_agri_ratio`, `lc_{radius}_water_wetland`
   - **Multi-scale contrast** (new): `lc_contrast_{cropland,urban,forest,bare}` = 1km % − 10km % (detects edge locations: city border, farm boundary)
   - **Shannon diversity** (new): `lc_5km_shannon` = entropy of land cover proportions at 5km (higher = more mixed land use)

2. **Geology (Macrostrat API)** — file: `geology_data.csv` (same as V7b)
   - One-hot encoded: `geo_lith_category` → `lith_*` dummies, `geo_rock_type` → `rock_*` dummies, `geo_proxy_ph_class` → `ph_class_*` dummies
   - Numeric: `geo_ec_impact` (1–5), `geo_alk_impact` (1–5), `geo_age_top_ma`, `geo_age_bottom_ma`
   - Derived: `geo_age_range` = bottom − top (Ma), `geo_log_age` = log1p(age_top)
   - Proxy features: `geo_proxy_cec_per_clay` (clipped ≤10), `geo_proxy_mineral_index` (clipped ≤100)
   - Soil columns from geology renamed with `geo_` prefix to avoid collision with SoilGrids data

**Feature engineering — New interactions:**

- **Land cover × climate** (the key V8 innovation):
  - `crop_x_ppt`, `crop_x_ppt_lag1` — cropland × rainfall = fertilizer runoff timing → DRP
  - `urban_x_ppt` — urban × rainfall = urban runoff → EC
  - `human_x_tmax` — human impact × temperature = evaporation concentrates pollutants
  - `natveg_x_soilmoist` — natural vegetation × soil moisture = natural filtering capacity
  - `bare_x_slope` — bare land × slope = erosion risk → sediment + minerals → EC
  - `crop_x_soil_n`, `crop_x_soil_ph` — cropland × soil properties = nutrient leaching potential
  - `urban_x_swir16` — urban × SWIR = urban reflectance signature
  - `wetland_x_ppt` — wetland × precipitation = natural nutrient filter

- **Geology × climate/terrain:**
  - `carb_x_ppt` — carbonate × rainfall = limestone dissolution → alkalinity
  - `alk_impact_x_tmax`, `ec_impact_x_tmax` — geology impact × temperature = weathering rate
  - `ec_impact_x_elev` — geology EC impact × elevation
  - `carb_x_crop` — carbonate × cropland = agricultural land on limestone
  - `ec_impact_x_swir` — geology × spectral signature

- **Combined spatial context indices:**
  - `anthro_pressure` = `(urban×3 + cropland×2 + bare×1.5) × geo_ec_impact/3` — composite anthropogenic load
  - `natural_buffer` = `forest + wetland×2 + soil_cec/10` — composite natural filtering capacity

**Feature engineering — V7 temporal features incorporated:**
- `is_cool_dry` — binary, months 5–10 (same as V7's `is_cool_dry_season`)
- `is_el_nino_drought` — binary, but **bounded**: `2014-10-01 ≤ date ≤ 2015-12-31` (V7 used open-ended `≥ 2014-10-01`; V8 is more precise)
- V7's season×spectral and drought×spectral interaction features (`cool_dry_x_swir16`, `drought_x_swir16`, etc.) are **NOT carried over** — V8 does not create these

**Data / splits changes:**
- **5 clusters, 5-fold GroupKFold** (reverts to V6's setup; V7 used 8/8)
- KMeans `n_init=10` (V6 default; V7 used 20)
- Feature count: **271 total** (V6 had 173, V7 had ~55 max)

**Model changes:**
- **4 XGBoost configs** with explicit `gamma` parameter (new vs V7):
  - `XGB_t1_deep`: depth=6, lr=0.05, min_child=5, gamma=0.1, on all 271 features
  - `XGB_t2_balanced`: depth=5, lr=0.05, min_child=10, gamma=0.2, on medium tier
  - `XGB_t3_focused`: depth=4, lr=0.05, min_child=15, gamma=0.3, on focused tier
  - `XGB_t3_conservative`: depth=3, lr=0.03, min_child=20, gamma=0.5, on focused tier
- **2 LightGBM configs**: `LGBM_t1` (depth=6, all features) and `LGBM_t3` (depth=4, focused)
- **Ridge**: alpha=**100.0** (V7 was 10.0, V6 was 1.0 — steadily increasing regularization)
- **ElasticNet**: alpha=1.0, l1_ratio=0.5 on focused tier
- **No log transform** tested for any target (V7 tested both raw and log; V6 used log for skewed targets)
- **No target stacking** implemented (despite being mentioned in the docstring as planned)
- **No KNN blending** (V7b had this)

**Training / evaluation changes:**
- `SimpleImputer(strategy='median')` fit on full training data per config (not per-fold column means like V7)
- Validation predictions: **averaged across all CV folds** (each fold trains on ~80% data, predicts val; average of 5 fold predictions). This is more robust than V7's single final-model prediction.
- **Weighted ensemble**: top-3 configs per target, weights = `max(R², 0.001)` normalized. This means even configs with negative R² contribute (floored at 0.001), which may dilute the ensemble.
- Physical range clipping: TA [0, 1000], EC [0, 10000], DRP [0, 5000]
- Results saved to CSV log (not text file like V7 — avoids encoding corruption)

### Results

V8 was **executed** on 2026-02-22 (timestamp `20260222_231718`). Results from `v8_results_20260222_231718.csv`:

| Target | Best Config | R² (Spatial CV) | V6 R² | Δ vs V6 |
|--------|-------------|-----------------|-------|---------|
| Total Alkalinity | XGB_t2_balanced | **-0.3959** | -0.2399 | -0.1560 ⬇ |
| Electrical Conductance | XGB_t3_focused | **-0.2285** | -0.4960 | +0.2675 ⬆ |
| Dissolved Reactive Phosphorus | XGB_t2_balanced | **-0.5905** | -0.2015 | -0.3890 ⬇ |
| **MEAN** | | **-0.4050** | **-0.3125** | **-0.0925 ⬇** |

**EC improved dramatically (+0.2675)**, but TA and DRP both regressed badly. The overall mean R² is **worse than V6** (-0.4050 vs -0.3125).

Key observations from the results:
- **EC: significant improvement.** The focused tier (land cover + geology + key terrain/soil + key Landsat + climate) with moderate regularization (depth=4) performed best. This supports the hypothesis that land cover and geology add meaningful spatial context for EC prediction.
- **TA: regression.** V6's best for TA was ElasticNet (linear model) on CORE features. V8 used only XGBoost/LightGBM on much larger feature sets. The linear model's superiority for TA suggests a predominantly linear TA relationship that XGBoost on 271 features overfits.
- **DRP: severe regression.** -0.5905 is the worst DRP result across all versions. V6's best DRP was XGBoost aggressive with log transform on CORE features. V8 dropped log transforms entirely, and DRP has the highest skewness (~1.64). Additionally, V8 did not implement target stacking (EC → DRP) despite it being referenced in the docstring.

Submissions generated:
- `submissions/submission_v8.csv`
- `submissions/submission_v8_20260222_231718.csv`

### Learnings

**Confirmed by results:**

1. **Land cover + geology significantly help EC prediction.** EC jumped from -0.4960 to -0.2285 — the single biggest improvement for any target in any version. This validates the V7b hypothesis that spatial context about human activity and bedrock type are the "missing ingredients" for EC. The `XGB_t3_focused` tier winning (not t1_deep with all 271 features) confirms that a curated set of these new features is more effective than throwing everything in.

2. **Dropping log transforms hurts skewed targets.** DRP's severe regression (-0.2015 → -0.5905) is likely partly caused by removing log transforms. DRP has skewness ~1.64, which V6 handled with `log1p`. V8 models all targets in raw scale. This is a **regression bug** — V6's proven log transform should have been preserved.

3. **Dropping linear models for TA is a mistake.** V6's best TA model was ElasticNet on CORE features (R² = -0.2399). V8's Ridge uses alpha=100 on the medium tier, but the best V8 TA result came from XGBoost (R² = -0.3959). The steadily increasing Ridge alpha (1.0 → 10.0 → 100.0 across V6/V7/V8) may have over-regularized past the sweet spot. TA appears to have a predominantly linear relationship with core features that XGBoost on 271 features disrupts.

4. **Feature count inflation without discipline hurts.** V8 went from V7's 32–55 focused features to 271 total features. While the focused tier was curated, the winning configs for TA and DRP used the medium tier (not focused) — suggesting the tier curation didn't help those targets, and the sheer feature volume caused overfitting in spatial CV.

5. **Cross-fold validation prediction averaging is a good practice.** V8's approach of averaging validation predictions across all 5 CV folds (each trained on ~80% data) is more robust than training a single final model on 100% data. This should be preserved in future versions.

6. **Weighted ensemble with floored weights is problematic.** Using `max(R², 0.001)` means configs with R² = -0.59 still contribute to the ensemble (weight ≈ 0.001). While the weight is tiny, including fundamentally broken models can only hurt. Future ensembles should exclude negative-R² models entirely.

**Actionable insights for V9:**
- Keep land cover + geology features (proven for EC)
- Restore log transforms for DRP (and test for TA)
- Restore ElasticNet/Ridge as primary model for TA
- Implement target stacking: EC predictions → DRP model (as V6 planned)
- Consider per-target models: EC uses V8's focused tier with geology/land cover, TA uses V6-style ElasticNet on core features, DRP uses log transform + core features + EC predictions
- Ensemble should exclude models with R² < 0

---

## Version History Summary (Updated)

| Version | Date | Spatial CV Mean R² | Leaderboard | Key Idea |
|---------|------|-------------------|-------------|----------|
| V0/V0.1 | 2026-02-14 | ~-0.60 | N/A | Benchmark XGBoost + existing Landsat/PET |
| V1 | 2026-02-15 | -0.40 | +0.006 | Expanded climate + elevation + soil |
| V2 | 2026-02-15 | -0.30 | -0.087 | Removed coordinates, Ridge/ElasticNet |
| V4 | 2026-02-18 | ~-0.35 | **~+0.04** ★ BEST LB | Research-backed climate/terrain/soil interactions |
| V5 | 2026-02-20 | ~-0.35 | TBD | V4 + Sentinel-2 (97.5% NaN — useless) |
| V6 | 2026-02-20 | **-0.3125** | TBD | Full Landsat bands + thermal + hybrid extraction |
| V7 | 2026-02-22 | *(corrupted)* | N/A | EC-only, focused features, LightGBM, seasonal flags |
| V7b | 2026-02-22 | *(not run)* | N/A | EC + WorldCover + geology + KNN (blocked) |
| V8 | 2026-02-22 | -0.4050 | TBD | Land cover + geology + all targets (EC improved, TA/DRP regressed) |

---

## Persistent Patterns Update (Post-V8)

11. **Per-target model architecture matters more than shared features.** V8 proved that one-size-fits-all XGBoost on 271 features hurts TA and DRP while helping EC. Each target has distinct optimal configurations: TA → linear models on core features, EC → tree models on spatial context features, DRP → log-transformed tree models with target stacking.
12. **New data sources have target-specific value.** Land cover and geology dramatically helped EC (+0.27 R²) but contributed nothing measurable to TA or DRP. Feature engineering should be target-aware.
13. **Removing proven techniques causes regression.** V8's DRP disaster (-0.59 vs -0.20) demonstrates that dropping log transforms that worked in V6 is a costly oversight. Future versions should preserve all proven techniques as baseline, only adding new ones.

---

## V9 — KNN Spatial Anchor + Residual Learning (2026-02-24)

### Research Insights Driving V9

The fundamental realization behind V9 is that **top teams scoring 0.82 R² cannot be achieving that from environmental features alone** — water quality has too much unexplained variance from purely physical/chemical predictors. They must be leveraging spatial proximity to training data. V9 makes this explicit.

Key insight: If test location A is 3km from training location B where EC=500 and alkalinity=200, then EC≈500 and alkalinity≈200 at location A is already a strong baseline. Environmental features then explain the residual deviation.

### Logic / Hypothesis

- **KNN target features as "spatial anchors"**: For each sample (train and validation), compute actual water quality statistics (mean, IDW-weighted mean, min, max, std, range) at the K=3,5,10,20 nearest training stations. This gives the model a starting point — "rivers nearby have EC≈X" — rather than predicting from scratch.
- **IDW (Inverse Distance Weighted) spatial interpolation as baseline**: Use 1/d² weighted average of K=5 nearest neighbors as a standalone prediction. This captures regional patterns without any ML at all.
- **Residual learning**: Train XGBoost to predict the *residual* (actual − KNN prediction) using environmental features. Much easier to predict small deviations than raw target values.
- **Two-approach architecture**: Compare pure ML (KNN features as inputs alongside environmental features) vs. KNN baseline + residual correction, then blend the best.
- **Restore log-transform for DRP**: V8 dropped log transforms and DRP regressed to -0.59. V9 restores `log1p` for DRP (skewness ~1.64).

### What Changed vs V8 (Code-Level Detail)

**Architecture shift — dual approach:**
- **Approach A (Pure ML)**: KNN target features + environmental features fed into XGBoost/LightGBM as a flat feature vector. The model learns how to weight KNN anchors vs environmental signals.
- **Approach B (KNN + Residual)**: Pure KNN-IDW as baseline → XGBoost predicts residual (actual − IDW) from environmental features → Final = IDW + α × residual. Alpha blending tested from 0.0 (pure KNN) to 1.0 (full residual correction).
- **Final ensemble**: Blends Approach A and B predictions. If one approach has R² > other + 0.05, use 70/30 split favoring the better one; otherwise 50/50.

**Feature engineering — New features (88 KNN features):**

1. **Per-location aggregation**: For each unique training location (162 total), computed mean, std, median, count of all 3 targets. These per-location statistics form the "target library" that KNN draws from.

2. **KNN target features** (core innovation, 74 features from KNN computation):
   - For each target × each K ∈ {3, 5, 10, 20}:
     - `knnK_{target}_mean`: Simple mean of K nearest training locations' target values
     - `knnK_{target}_idw`: Inverse-distance-squared weighted mean (closer stations weighted more)
     - `knnK_{target}_min`, `_max`: Range of neighboring values
     - `knnK_{target}_std`: Standard deviation (spatial heterogeneity indicator)
     - `knnK_{target}_range`: Max − Min of neighbors
   - Total: 3 targets × 4 K values × 6 statistics = 72 features
   - Plus `dist_nearest_km` and `dist_5th_nearest_km` = 74 features

3. **Spatial distance computation**: KD-tree (cKDTree) built from training location coordinates converted to approximate km (lat × 111.32, lon × 111.32 × cos(mean_lat)). **Leave-one-out** for training data: each training sample excludes its own location (distance > 10m threshold) when computing KNN, preventing target leakage.

4. **V9-specific derived features** (14 additional features):
   - `spatial_anomaly_{target}`: (IDW prediction − global mean) / |global mean| — how unusual is this location regionally?
   - `knn_{target}_x_wet`: KNN prediction × wet season flag
   - `knn_{target}_x_ppt`: KNN prediction × precipitation / 100 (dilution interaction)
   - `knn_confidence`: 1 / (1 + dist_nearest_km) — trust signal for KNN predictions
   - `log_dist_nearest`: log1p(dist_nearest_km)
   - `knn_cv_{target}`: Coefficient of variation of K=5 neighbors (homogeneity indicator)

5. **All V8 features preserved**: Land cover (ESA CCI), geology (Macrostrat), full Landsat, TerraClimate, soil, interactions — 271 environmental features carried forward unchanged.

**Total feature count: 359** (271 V8 environmental + 88 KNN/spatial)

Feature breakdown:
| Group | Count |
|-------|-------|
| KNN Target Features (NEW) ★ | 88 |
| Land Cover | 53 |
| Geology | 25 |
| Full Landsat | 39 |
| EY Landsat | 6 |
| Unified Landsat | 7 |
| TerraClimate | 48 |
| Soil | 13 |
| Interactions | 43 |
| Other/Derived | 46 |

**Data / splits changes:**
- **5 clusters, 5-fold GroupKFold** (same as V6/V8; V7 used 8/8)
- KMeans `n_init=10` (same as V8)
- Base data: V8's `v8_train_features.csv` and `v8_val_features.csv` with KNN features appended
- Train shape: (9319, 365), Val shape: (200, 362)

**Model configurations — Approach A:**
- 4 feature tiers per target:
  - `knn_only`: Target-specific KNN features only (~26 features per target)
  - `knn_env_focused`: KNN + focused environmental features (land cover 5km/10km, geology, core terrain/soil/climate, key Landsat, interactions) — ~122 env features
  - `knn_env_all`: KNN + all 271 environmental features
  - `all_features`: All 359 features
- 3 model types per tier:
  - `XGB_reg`: depth=5, lr=0.05, n_est=400, alpha=2, lambda=10, min_child=10, gamma=0.2
  - `XGB_cons`: depth=3, lr=0.03, n_est=200, alpha=5, lambda=20, min_child=20, gamma=0.5
  - `LGBM`: depth=5, lr=0.05, n_est=400, alpha=2, lambda=10, min_child=15
- Total: 4 tiers × 3 models = 12 configs per target × 3 targets = 36 configs

**Model configurations — Approach B:**
- KNN-IDW baseline: K=5, inverse distance squared weights
- Residual model: XGB_cons (depth=3, lr=0.03, conservative regularization)
- Alpha blending tested: 0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0
- Features: focused_env only (~122 features)
- 7 alpha values × 3 targets = 21 configs

**Target transforms:**
- DRP: `log1p` applied (restored from V6, fixes V8 regression)
- TA, EC: raw scale

**Training / evaluation changes:**
- `SimpleImputer(strategy='median')` fit per config (not per fold)
- Validation predictions: averaged across all 5 CV folds (same as V8)
- Physical range clipping: TA [0, 1000], EC [0, 10000], DRP [0, 5000]
- NaN fallback: If Approach B KNN predictions have NaN for any validation point, fall back to Approach A predictions for those points

### KNN Feature Sanity Check

**IDW-actual correlation (training data, leave-one-out):**
| Target | IDW-Actual Correlation |
|--------|------------------------|
| Total Alkalinity | 0.5615 |
| Electrical Conductance | 0.5809 |
| Dissolved Reactive Phosphorus | 0.3511 |

Interpretation: KNN-IDW explains ~56% of TA variance and ~58% of EC variance from spatial proximity alone. DRP is weaker at 35%, suggesting DRP is more driven by temporal/event-based factors (rainfall → runoff) than spatial patterns.

**Validation distances (to nearest training location):**
| Metric | Value |
|--------|-------|
| Min | 81.73 km |
| Median | 194.33 km |
| Mean | 192.41 km |
| Max | 281.54 km |
| < 10 km | 0/200 (0%) |
| < 50 km | 0/200 (0%) |

Critical insight: **No validation point is within 50 km of any training location.** The minimum distance is ~82 km. This means KNN predictions for validation are extrapolating from distant stations, which limits KNN effectiveness. Despite this, KNN-IDW still provides useful signal — demonstrating that regional water quality patterns persist over 100+ km scales.

### Results

V9 was **executed** on 2026-02-24 (timestamp `20260224_231610`). Results from terminal output and `v9_results_20260224_231610.csv`:

**Approach A — Pure ML with KNN features (Spatial CV):**

| Target | Best Config | R² | Per-fold R² |
|--------|-------------|-----|-------------|
| Total Alkalinity | knn_env_all_XGB_cons | **-0.1821** | [+0.144, -0.606, -0.091, -0.097, -0.260] |
| Electrical Conductance | knn_env_all_XGB_cons | **-0.1487** | [+0.055, -0.517, -0.013, -0.284, +0.015] |
| Dissolved Reactive Phosphorus | all_features_XGB_cons | **-0.1536** | [-0.726, +0.056, -0.030, -0.111, +0.043] |

Key observations:
- `knn_env_all` (KNN + all environmental) with conservative XGBoost wins for TA and EC
- `all_features` (everything including all KNN variants) wins for DRP
- Conservative XGBoost (`XGB_cons`) wins across all targets — confirming V7/V8's finding that heavy regularization is needed for spatial generalization
- Fold 2 (index 1) is consistently the worst fold across all targets (R² = -0.606, -0.517) — this spatial cluster is likely the most dissimilar from the rest
- KNN-only features (`knn_only`) perform WORSE than KNN+environmental, confirming that KNN anchors need environmental context to be useful

**Approach B — KNN Baseline + Residual Model (Spatial CV):**

| Target | Pure KNN-IDW R² | Best Blend α | Best Blend R² | Per-fold R² |
|--------|-----------------|--------------|---------------|-------------|
| Total Alkalinity | **+0.3088** | α=0.0 | **+0.0770** | [+0.111, +0.154, +0.204, -0.287, +0.203] |
| Electrical Conductance | **+0.3144** | α=0.2 | **+0.1282** | [+0.172, +0.223, +0.423, -0.308, +0.132] |
| Dissolved Reactive Phosphorus | **+0.0682** | α=0.0 | **-0.4044** | [-0.200, -0.040, -0.084, -0.207, -1.491] |

**This is a breakthrough moment:** Pure KNN-IDW achieves **positive R²** for all 3 targets — TA +0.31, EC +0.31, DRP +0.07. This is the first time any approach has achieved positive R² in spatial CV for TA and EC. KNN-IDW alone, with zero ML at all, beats every ML model from V0–V8.

However, the residual correction (alpha > 0) hurts for TA and DRP:
- TA: best at α=0.0 (pure KNN), any residual correction degrades performance (R² drops from +0.077 to -0.041 at α=1.0)
- EC: best at α=0.2 (mild residual correction helps), improving over pure KNN from +0.119 to +0.128
- DRP: best at α=0.0 (pure KNN), residual correction catastrophically hurts (R² drops from -0.404 at α=0.0 to -0.867 at α=1.0)

Note: The CV-evaluated blend R² (+0.077, +0.128) is lower than pure KNN-IDW R² (+0.309, +0.314) because the per-fold CV evaluation applies leave-one-cluster-out, while pure KNN-IDW R² was computed on the full training set. The per-fold R² values are the reliable metric.

**Final Ensemble (Approach A + B blend):**

| Target | Approach A R² | Approach B R² | Blend Ratio | Val Prediction Range |
|--------|---------------|---------------|-------------|---------------------|
| Total Alkalinity | -0.1821 | +0.0770 | 70% B + 30% A | [44.8, 214.8], mean=116.4 |
| Electrical Conductance | -0.1487 | +0.1282 | 70% B + 30% A | [190.2, 574.9], mean=350.9 |
| Dissolved Reactive Phosphorus | -0.1536 | -0.4044 | 70% A + 30% B | [14.2, 37.7], mean=23.5 |

For TA and EC, Approach B (KNN + residual) is clearly better and gets 70% weight. For DRP, Approach A (pure ML) is better and gets 70% weight.

**Comparison with previous versions:**

| Target | V6 Best | V8 Best | V9 Approach A | V9 Approach B | V9 Δ vs V6 |
|--------|---------|---------|---------------|---------------|------------|
| TA | -0.2399 | -0.3959 | -0.1821 | **+0.0770** | +0.3169 ⬆ |
| EC | -0.4960 | -0.2285 | -0.1487 | **+0.1282** | +0.6242 ⬆ |
| DRP | -0.2015 | -0.5905 | **-0.1536** | -0.4044 | +0.0479 ⬆ |

### Submissions Generated
- `submissions/submission_v9.csv` (final ensemble)
- `submissions/submission_v9_20260224_231610.csv` (timestamped copy)
- Earlier run also exists: `submissions/submission_v9_20260222_235150.csv` (identical results)

### Learnings

**Confirmed by results:**

1. **KNN-IDW is the single most important innovation so far.** Pure spatial interpolation (zero ML) achieves R² = +0.31 for both TA and EC in spatial CV — outperforming every ML configuration from V0–V8. This validates the core hypothesis: water quality at a location is primarily determined by water quality at nearby locations. Environmental features explain residual variation, not the primary signal.

2. **Residual correction mostly hurts, not helps.** For TA and DRP, the best alpha is 0.0 (no residual correction — pure KNN). For EC, a mild α=0.2 correction provides a marginal +0.01 improvement. The residual models are likely overfitting to spatial patterns in the training data that don't transfer. The environmental features that should explain "why this location deviates from its neighbors" (seasonal rainfall, land use) aren't capturing enough transferable signal.

3. **Conservative XGBoost consistently wins in Approach A.** `XGB_cons` (depth=3, lr=0.03, heavy regularization) beats regular XGBoost and LightGBM across all targets. This is consistent with V7/V8 findings — spatial generalization requires strong regularization to prevent memorizing training cluster patterns.

4. **Validation locations are 82–282 km from training data.** The complete absence of any validation point within 50 km of training data means KNN predictions are necessarily coarse. Yet they still work — suggesting that river systems in the same broad geological/climatic region share similar water quality characteristics at 100+ km scales. This is consistent with geology and land use being the dominant drivers.

5. **DRP remains the hardest target.** IDW-actual correlation is only 0.35 (vs 0.56–0.58 for TA/EC). Pure KNN-IDW R² is only +0.07. DRP is driven more by temporal events (rainfall → agricultural runoff → phosphorus loading) than spatial patterns, making KNN interpolation less effective. The log-transform restoration helped (V9 DRP -0.15 vs V8 DRP -0.59), but DRP needs event-based features (lagged rainfall, antecedent dry period, etc.) more than spatial anchoring.

6. **Fold 2 is the universal problem fold.** Across nearly all configs and targets, fold 2 (spatial cluster 1) has the worst R². This cluster likely represents a geographic region with fundamentally different water quality drivers than the rest of the training data (possibly a different geological province or land use pattern). Investigating this cluster's characteristics could reveal what features are missing.

7. **KNN + all environmental features > KNN alone.** In Approach A, `knn_env_all` and `knn_env_focused` consistently beat `knn_only`, confirming that KNN anchors need environmental context. However, `all_features` (adding all KNN variants for all targets simultaneously) only helps DRP — for TA and EC, adding cross-target KNN features introduces noise.

**Actionable insights for V10+:**
- KNN-IDW should be the default baseline for all future versions — any ML model must beat +0.31 R² to be useful
- Focus residual modeling on EC only (the one target where α > 0 helps)
- For TA: pure KNN-IDW is hard to beat; explore whether KNN with K > 5 or different distance weighting (e.g., 1/d instead of 1/d²) improves further
- For DRP: shift focus from spatial features to temporal features (antecedent rainfall, dry spell length, seasonal nutrient loading patterns)
- Investigate fold 2 (cluster 1): what makes this region different? Is it a data quality issue or genuine distributional shift?
- Consider using KNN from *feature space* (not just geographic space) — locations far apart geographically but similar in geology/land use might be more informative neighbors
- The 70/30 ensemble blend (B/A for TA/EC, A/B for DRP) should be submitted and compared against pure KNN-IDW to test whether the blend adds or detracts on the leaderboard

---

## Version History Summary (Updated)

| Version | Date | Spatial CV Mean R² | Leaderboard | Key Idea |
|---------|------|-------------------|-------------|----------|
| V0/V0.1 | 2026-02-14 | ~-0.60 | N/A | Benchmark XGBoost + existing Landsat/PET |
| V1 | 2026-02-15 | -0.40 | +0.006 | Expanded climate + elevation + soil |
| V2 | 2026-02-15 | -0.30 | -0.087 | Removed coordinates, Ridge/ElasticNet |
| V4 | 2026-02-18 | ~-0.35 | **~+0.04** ★ BEST LB | Research-backed climate/terrain/soil interactions |
| V5 | 2026-02-20 | ~-0.35 | TBD | V4 + Sentinel-2 (97.5% NaN — useless) |
| V6 | 2026-02-20 | -0.3125 | TBD | Full Landsat bands + thermal + hybrid extraction |
| V7 | 2026-02-22 | *(corrupted)* | N/A | EC-only, focused features, LightGBM, seasonal flags |
| V7b | 2026-02-22 | *(not run)* | N/A | EC + WorldCover + geology + KNN (blocked) |
| V8 | 2026-02-22 | -0.4050 | TBD | Land cover + geology + all targets (EC improved, TA/DRP regressed) |
| **V9** | **2026-02-24** | **+0.0770 / +0.1282 / -0.1536** | **TBD** | **★ KNN spatial anchor + residual learning. First positive CV R² for TA/EC.** |

---

## Persistent Patterns Update (Post-V9)

14. **Spatial interpolation beats ML for spatial generalization.** V9's pure KNN-IDW (R² = +0.31 for TA/EC) outperforms every ML model from V0–V8. The primary signal in water quality is *geographic proximity*, not environmental features. ML models should be used to *refine* spatial predictions, not replace them.
15. **Residual learning is harder than expected.** Despite the intuition that "predicting small deviations should be easier than raw values," the residual models mostly hurt. Environmental features apparently capture correlations that break across spatial clusters, causing residual models to add noise rather than signal. Only EC benefits from mild residual correction.
16. **The validation gap is ~100–280 km.** With no validation points within 50 km of training data, all predictions are genuinely out-of-distribution. The success of KNN-IDW at this distance scale confirms that water quality has meaningful large-scale spatial structure (likely driven by geology and regional land use).
17. **DRP requires a different paradigm.** Unlike TA and EC (which are well-captured by spatial proximity), DRP is event-driven (rainfall → runoff → phosphorus). Future DRP modeling should prioritize temporal features over spatial features.

---

## Ideas Not Yet Tried (Updated)

- [x] ~~LightGBM~~ — Added in V7, used in V9
- [x] ~~Per-target specialized models~~ — V7 (EC-only), V9 (per-target approaches)
- [x] ~~Nearest-neighbor blending~~ — V9's core innovation (KNN-IDW)
- [x] ~~Land use / land cover~~ — V8 (ESA CCI)
- [x] ~~Geology data~~ — V7b/V8 (Macrostrat)
- [x] ~~Log-transform for DRP~~ — Restored in V9
- [ ] Optuna hyperparameter tuning on V9's winning configs
- [ ] CatBoost as alternative boosting framework
- [ ] Neural network approaches (tabular transformers, TabNet)
- [ ] Watershed/catchment delineation features
- [ ] Water body proximity / river network features
- [ ] Quantile regression for robust predictions
- [ ] Different spatial CV strategy (leave-one-region-out may be too harsh)
- [ ] KNN in *feature space* (not geographic space) — geologically similar but distant locations
- [ ] Vary KNN distance weighting (1/d vs 1/d² vs Gaussian kernel)
- [ ] Increase K for validation (K=20, K=50) given the 82+ km distances
- [ ] Temporal features for DRP: antecedent rainfall, dry spell length, cumulative precipitation
- [ ] Investigate fold 2 / cluster 1 — why is it always the worst?
- [ ] Pure KNN-IDW submission (no ML, just interpolation) as a baseline check
- [ ] Multi-scale KNN: different K values for different targets (K=3 for EC, K=20 for DRP)
- [ ] Coordinate features re-added with heavy regularization

---

## V11 — Feature-Space KNN: Environmental Similarity (2026-02-25)

### Research Insights Driving V11

**Why V10 (pure geographic KNN) failed on the leaderboard:**

V10 had excellent LOO R² (TA +0.33, EC +0.38, DRP +0.20) but the predictions were nearly **flat** — TA range [75–144], EC range [220–497], DRP range [22–27]. At 82–282 km away, every validation point finds roughly the same 30 distant training neighbors, so IDW-weighted averages converge to a regional mean. The model couldn't distinguish between validation locations.

**The core insight for V11:**

Geographic proximity is almost meaningless at 82+ km. Two validation points 10 km apart with different geology should get *different* predictions. But geographic KNN gives them *identical* predictions because they share the same distant neighbors.

**Solution: Find neighbors by environmental similarity, not geography.**

A training station 300 km away on the same limestone formation with similar cropland % is more informative than a training station 100 km away on completely different geology. V8 already proved geology dramatically helps EC (+0.27 R²) — feature-space KNN makes this systematic.

### Logic / Hypothesis

- **Feature-space KNN**: Build a KD-tree from location-characterizing features (geology, terrain, soil, land cover). Find the K most environmentally similar training stations for each query, regardless of geographic distance.
- **Higher prediction variance**: Two validation points on different geology will find different feature-space neighbors, even though their geographic neighbors overlap. This creates the prediction variance that V10 lacked.
- **Blend with geographic KNN**: Test fixed blend ratios (w_geo from 0.0 to 1.0) to find the optimal mix of geographic and environmental similarity.
- **Mild ML correction for EC only**: V9 showed alpha=0.2 helps EC. Test residual correction on top of the best blend.
- **No new data, no new features**: Reuses V9's 359 features. The innovation is in *how neighbors are found*, not in what features exist.

### What Changed vs V9/V10 (Code-Level Detail)

**Architecture: Three-layer prediction**
1. Geographic KNN-IDW (V10 style, K=30, tuned power per target)
2. Feature-Space KNN-IDW (NEW — K and power tuned via spatial CV)
3. Blend: `w_geo × geo_pred + (1 - w_geo) × feat_pred`
4. Optional: `+ alpha × XGBoost_residual_correction` (tested for all targets)

**Similarity features (location-invariant, 25 features):**
- Terrain: `elevation_m`, `slope_deg`, `terrain_roughness`
- Soil: `soil_clay`, `soil_sand`, `soil_soc`, `soil_phh2o`, `soil_cec`, `soil_nitrogen`
- Land cover 5km: `lc_5km_pct_cropland`, `lc_5km_pct_forest`, `lc_5km_pct_grassland`, `lc_5km_pct_urban`, `lc_5km_pct_bare`, `lc_5km_pct_water`, `lc_5km_human_impact`, `lc_5km_natural_veg`
- Geology: `geo_ec_impact`, `geo_alk_impact`, `geo_is_carbonate`, `geo_is_siliciclastic`, `geo_is_fine_sedimentary`, `geo_is_felsic_igneous`, `geo_is_mafic_igneous`

All features StandardScaled before KD-tree construction (so soil_clay % and elevation_m contribute equally to distance).

**Per-location database:**
- 162 unique training locations, each with median environmental features + mean target values
- Feature-space KD-tree built from these 162 location feature vectors
- Leave-one-location-out within each CV fold for ML residual training

**Parameters tuned:**
- Feature-space K: {3, 5, 7, 10, 15, 20}
- Feature-space power: {1.0, 1.5, 2.0}
- Blend w_geo: {0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0}
- ML alpha: {0.0, 0.1, 0.2, 0.3}

**ML correction model (same as V9):**
- XGBoost conservative: depth=3, lr=0.03, n_est=200, alpha=5, lambda=20, min_child=15
- ~50 focused environmental features (terrain + soil + land cover + geology + climate + Landsat + interactions)
- Target: residual (actual − blended KNN prediction)

**Spatial CV:**
- 5 clusters, 5-fold GroupKFold (consistent with V6/V8/V9)
- Within each fold: trees built only from fold-training locations
- Feature-space scaler fit on fold-training locations only (no leakage)

### Results

**PENDING — Run `python scripts/23_model_v11.py` and paste output.**

### Expected Outcomes

- Feature-space KNN should produce **wider prediction ranges** than V10's geographic KNN
- TA: feature-space KNN should improve (geology drives alkalinity — carbonate vs granite)
- EC: blend should improve (land cover + geology both matter for EC)
- DRP: uncertain (DRP is event-driven, spatial similarity may not help as much)
- If feature-space KNN alone beats geographic KNN, this validates the core hypothesis

---

*Last updated: 2026-02-25 (V11 pre-results, code written, awaiting execution)*

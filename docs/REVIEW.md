# Repository review and cleanup report

This review checks the code, documentation, saved CSVs, and local data caches. Historical notes are evidence of recorded experiments, not independently reproduced accuracy results. The original experiment log has been preserved rather than retroactively rewritten.

## 1. Verified project facts

| Item | Checked in the local files |
|---|---|
| Training data | 9,319 rows, 162 distinct coordinate pairs, three complete target columns |
| Training period | 2011–2015 |
| Prediction template | 200 rows, 24 distinct locations |
| V8 feature tables | Train: 9,319 × 277; prediction: 200 × 274; 271 predictor columns |
| V9 feature tables | Train: 9,319 × 365; prediction: 200 × 362; 359 predictor columns |
| Prediction-to-training distance | Minimum 81.73 km; median 194.33 km; maximum 281.54 km, using the project's approximate distance conversion |
| Saved V9 submission | 200 × 6; no missing values; identifiers match the template in order |

The additional columns in training feature tables are identifiers and the three target measurements, not additional predictors. Geographic distances are approximate; they are not river-network distances.

## 2. What the recorded scores actually show

R² is the recorded evaluation metric; the challenge notes describe its mean across the three targets. **Fold-averaged R², pooled R², and leave-one-station-out R² are different measurements.** A negative R² is worse than predicting the evaluated observations' mean.

### Environmental models

- V2's per-version report records mean spatial CV R² of approximately **−0.3054**.
- V6 records approximately **−0.3125**.
- V8's result CSV records approximately **−0.4050** overall. Its Conductance result improved relative to V6, but the other targets regressed.

These are historical model-selection results, with changing feature sets, transforms, and sometimes fold settings. They are not controlled ablations proving the effect of each source.

### V9: preserve the artifact, qualify the evaluation

From `experiment_logs/v9_results_20260224_231610.csv`:

| Target | Approach A: ML with neighbor features | Approach B: spatial estimate + residual |
|---|---:|---:|
| Total Alkalinity | −0.1821 | +0.0770 |
| Electrical Conductance | −0.1487 | +0.1282 |
| Dissolved Reactive Phosphorus | −0.1536 | −0.4044 |

**These are not leakage-free regional CV scores.** `20_feature_engineering_v9.py` builds its station statistics and neighbor library from the entire labeled training file. `21_model_v9.py` then splits rows into spatial folds, without rebuilding that library.

Training features exclude the query station, but can still use measurements from **other stations in the held-out region**. The same problem affects Approach B's baseline and residuals. The imputer is also fitted before the fold split.

The logged positive “pure KNN” pooled R² values are leave-one-location-out diagnostics, not leave-one-region-out accuracy. Averaging those precomputed predictions by region does not repair the neighbor-library leakage. The final 70/30 ensemble itself does not have a separately logged clean CV score.

This does **not** mean hidden prediction-set labels were used: that set has no supplied target labels. A full training library is normal for final inference. The problem is interpreting the internal evaluation as an honest unseen-region test.

The two saved V9 result CSVs have identical scores after removing their timestamps. They were retained as run records, not mislabeled as byte-identical duplicates.

### V10 and V11: harder holdouts remained difficult

V10 rebuilds its station library from each spatial training fold. Its parameters are selected using a different, easier station-holdout test:

| Target | V10 station-holdout R² | V10 regional R² | V11 recorded final regional R² |
|---|---:|---:|---:|
| Total Alkalinity | +0.3315 | −0.5440 | −0.3524 |
| Electrical Conductance | +0.3797 | −0.3023 | −0.2175 |
| Dissolved Reactive Phosphorus | +0.1961 | −0.6209 | −0.3497 |

Sources: `v10_results_20260223_224444.csv` and `v11_results_20260225_224102.csv` in `experiment_logs/`.

V11's scores exist, so the old log's “PENDING” statement is stale. Its feature-space trees and primary preprocessing are constructed from fold-training locations. However, many configurations are selected on the same folds: these remain tuning results, not an untouched final test. Additional implementation issues below also prevent promoting its saved predictions.

### Leaderboard evidence

Historical notes report V1 at **+0.006**, V2 at **−0.087**, and V4 around **+0.04**. V4's standalone report still says “TBD,” and no official leaderboard export is stored here. A V10 code comment mentions a V9 score, but that is not independent confirmation of a leaderboard result.

V9 remains the **author-designated saved baseline**, not a demonstrated leaderboard winner. No V12 results or submission exist in this checkout. V7's text result file contains only a damaged header; V7b has no saved result artifacts.

## 3. Data and implementation findings

### Geology categories did not succeed

All **186** local geology rows have `geo_lith_category = unknown`. Rock type is missing everywhere, the lithology indicator columns are all zero, and the Conductance/Alkalinity impact scores are constant defaults (3 and 2).

The raw lithology field contains numeric-looking values rather than usable rock names. Verify the API response schema and required lookups. Age fields and soil-derived proxies do vary, so not every geology-prefixed column is empty; nevertheless, the current artifacts do **not** demonstrate successful bedrock classification or isolate its predictive value.

### Soil units need correction before physical interpretation

The soil pH column contains values **55–83**, consistent with scaled storage values rather than pH in physical units. The geology proxy code applies bins between 0 and 14 without converting those values, and `geo_proxy_ph_class` is missing in all local rows.

Read the provider's units/conversion factors, convert explicitly, and add range checks before using physical thresholds. Similar unit checks are needed for texture, carbon, nitrogen, and CEC. Soil properties are missing at **65 of 186 locations** (34.9%). These issues were documented, not silently fixed, to avoid changing the historical baseline.

### Imagery availability and quality

- Full Landsat reflected-light bands: **85.7% training coverage**, **87.5% prediction coverage**.
- Surface temperature: **32.0% training**, **45.5% prediction**.
- Local Sentinel-2 spectral columns: **100% missing**. Older notes state 97.5%, which does not match this cache. Check mission/catalog coverage for 2011–2015 before attributing failure to clouds or scheduling another extraction.
- Landsat extraction uses scene-level cloud filtering, not the full per-pixel cloud/water masking demonstrated in the original notebooks. A small mixed land/water window is not necessarily a clean river-surface measurement.
- Land-cover extraction targets ESA CCI 2013 but can fall back to WorldCover 2021. The CSV lacks source/year provenance, so its exact source is not independently recoverable from the cache. Its surrounding windows are bounding boxes, not delineated upstream catchments.

### Later model code needs repairs

The archived V11 final-inference section computes residuals using `y_all` left over from the last target's evaluation loop. That last target is Phosphorus, so Alkalinity and Conductance corrections are trained against the wrong target vector. Its reported CV scores do not establish that the saved submission uses the same correct procedure.

V11 and V12 also query temporary feature-space trees using a different imputer/scaler than the one fitted for those trees in portions of residual training. Tree points and query points must use the same transformation.

V12's “image-only” name is an experimental description, not a strict provenance guarantee: it uses soil/terrain maps, coordinates, measured-neighbor targets, and precomputed V9 target-derived features. Its Approach A inherits V9's leakage, and stacked target predictions need fold-isolated training. It has no saved results here.

None of these model/data changes were attempted during repository housekeeping. They need separate, versioned experiments and evaluation.

## 4. Cleanup performed

- Moved **20 superseded or alternative scripts** into `scripts/archive/`. The main workflow retains ten existing scripts plus a new submission validator. See the [archive index](../scripts/archive/README.md) for every moved file.
- Adjusted archived script roots for their new location; repaired old vendor-folder references and personal absolute paths throughout the Python scripts. Historical algorithms were retained.
- Deleted `submissions/submission_v9.csv` only after verifying it was byte-identical to `best_submission/submission_v9.csv`. The protected reference remains unchanged.
- Deleted `notebooks/.gitkeep` and the unused `data/raw/README.md` pointer after consolidating their information here. Their now-empty directories were removed.
- Removed duplicate coordinate-rounding work and a duplicate platform-column selection in V8; removed unused imports and unused blend-weight calculations from the active workflow.
- V9 reruns now emit one submission file, not an extra timestamped duplicate. Generated V9 and historical timestamped copies are git-ignored. Timestamped result logs remain useful and are still written.
- Removed unused CatBoost, Optuna, and SHAP requirements; explicitly listed `requests` and the land-cover NetCDF fallback dependency `h5netcdf`.
- Updated Windows setup to anchor itself to the project folder, use the venv's Python, and stop on installation errors.
- Rewrote the README, added a plain-English summary and a complete reproduction guide, and appended a corrective close-out note to the original experiment log.
- Added a standard-library submission validator and offline tests. No paid services, new datasets, or model promotion were introduced.

**Preserved:** every unique experiment, all original EY notebooks/data, all local extracted CSVs, all other submissions, experiment logs, the virtual environment, and the saved V9 reference. Pre-cleanup hashes were recorded outside the repository to verify protected files afterward. Git history also retains deleted tracked scaffolding and the duplicate file; nothing was committed or staged automatically.

## 5. Validation and limits

Completed checks:

- **19 offline tests passed**, covering submission validation, parsing every Python file, removal of stale paths, and path resolution from another directory.
- The saved V9 reference passed the **200-row submission integrity check**.
- **52 protected original files** (EY files, extracted data/feature lists, saved reference, and other submissions) matched their pre-cleanup SHA-256 hashes byte for byte.
- The original experiment log remained an unchanged prefix; only the close-out section was appended.
- All **20 archived sources** matched their original algorithms after accounting for the intended path/configuration edits.
- All **14 local Markdown links** in the rewritten documentation resolved.
- `01_analyze_data.py`, `check_landsat_quality.py`, and the validator ran successfully from a non-project working directory.
- V8 and V9 features were rebuilt **in an isolated temporary project**. Both training/prediction tables matched the saved data within `rtol=1e-10`, `atol=1e-8`; the 271- and 359-column feature lists were unchanged.
- A **reduced-cost V9 execution smoke test** (750 sampled training rows, two estimators per fit, temporary monkeypatched model factories) exercised prediction generation and logging. It produced one valid 200-row submission, kept the reference unchanged, and recorded the evaluation caveat. Those temporary predictions were discarded; their scores are not experiment results.
- `git diff --check` passed. Ignore patterns were checked independently of the existing Git index; the tracked duplicate's removal takes effect in the index when the cleanup is staged/committed.

Feature/model smoke checks used the existing Python 3.13.5 environment (pandas 2.3.3, NumPy 2.3.5, scikit-learn 1.8.0, XGBoost 3.2.0, LightGBM 4.6.0), not a fresh installation of the pinned requirements.

**Not performed:** full-cost retraining, live remote extraction, leaderboard submission, or a fresh dependency/setup installation. The archived experiments were not executed. Integrity checks do not establish predictive accuracy.

## 6. Recommended final modeling pass

1. Preserve the V9 CSV as a historical reference, not a claim of validated accuracy.
2. Fix data provenance, geology parsing, soil-unit conversions, and imagery masking.
3. Use fixed regional holdouts with all target-derived features rebuilt from fold-training stations only. Fit imputation/scaling inside those folds.
4. Produce fold-local training neighbor estimates with station exclusion or inner cross-fitting; isolate stacked-target training too.
5. Repair V11's target-vector and scaler inconsistencies before reusing its residual approach.
6. Compare a mean baseline, an environmental-only model, and an interpolation model using identical folds. Evaluate the actual final blend, not just its individual components.
7. Select configurations on inner folds and report performance on an untouched outer holdout. Save predictions, exact dependency versions, fold assignments, and any official leaderboard result together.

This would turn the existing research work into a defensible final experiment rather than another speculative version.

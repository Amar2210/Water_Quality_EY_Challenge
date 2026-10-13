# Archived experiments

These **20 scripts are not required by the V9 workflow**. They were moved out of the main `scripts/` folder to reduce clutter without deleting unique research.

Their data paths were adjusted for this location and the renamed `ey_challenge_notebooks/` folder. Their experimental algorithms and historical claims were otherwise preserved. This is an archive, **not a second recommended production pipeline**.

Older logs refer to `scripts/<filename>`; the corresponding file now lives at `scripts/archive/<filename>` unless it remains in the active workflow.

| File | Why it is archived |
|---|---|
| `02_model_v0.py` | Initial XGBoost baseline. |
| `02_model_v0.1.py` | A distinct baseline variant; not a byte-identical duplicate. |
| `04_model_v1.py` | Early expanded climate/soil/terrain model. |
| `05_model_v2.py` | Coordinate removal and regularization experiment. |
| `06_model_v3.py` | Intermediate spatial/neighbor experiment; no saved V3 submission in this checkout. |
| `06_ensemble_v1v2.py` | Historical blend of two earlier submissions. |
| `08_feature_engineering_v4.py` | Superseded feature builder. |
| `09_model_v4.py` | Historical V4 model. |
| `10_feature_engineering_v5.py` | Sentinel-2 variant of earlier features. |
| `11_model_v5.py` | Historical V5 model. |
| `12_feature_engineering_v6.py` | Earlier Landsat feature builder; V8 rebuilds directly from source caches. |
| `13_model_v6.py` | Historical V6 model and cross-version ensemble. |
| `14_model_v7_ec.py` | Conductance-only experiment; saved detailed results are incomplete. |
| `15_extract_worldcover.py` | Standalone WorldCover branch; expected `worldcover_data.csv` is absent locally. |
| `17_model_v7b_ec.py` | Conductance/WorldCover/geology/KNN branch; no saved run artifacts. |
| `19_model_v8.py` | Historical V8 model; V8's feature builder remains active because V9 needs it. |
| `22_model_v10.py` | Pure interpolation experiment, with both station and regional holdouts. |
| `23_model_v11.py` | Environmental-similarity KNN experiment; saved results exist, but final inference needs repairs. |
| `24_model_v12_image_only.py` | Unfinished smaller-feature/stacking branch; no saved results or submission. |
| `extract_sentinel2.py` | Optional failed-data branch; local spectral columns are entirely missing. |

## Before rerunning

1. Read the [review](../../docs/REVIEW.md), especially the V9 leakage, V11 target-vector/scaler issues, and data-unit findings.
2. Check the required version's feature caches and optional libraries. Sentinel-2 imports `stackstac` (`python -m pip install stackstac`), which the active pipeline does not require.
3. Back up any submission or feature files that the script may overwrite.
4. Run from the project root, for example:

   ```bat
   python scripts\archive\22_model_v10.py
   ```

   V10 has an extensive parameter sweep; it is not an integrity smoke test.

Archived scripts may still print their original suggested commands or emit duplicate timestamped submissions. These historical commands should be interpreted using the location mapping above. Timestamped output copies are git-ignored; dated result logs remain useful run records.

The original tracked sources also remain recoverable from Git history. No unique experiment was discarded.

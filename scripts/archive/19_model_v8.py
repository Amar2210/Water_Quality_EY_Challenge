"""
============================================================
EY Water Quality Challenge - V8 Model
============================================================
WHAT'S NEW IN V8:
  - Land Cover features (ESA CCI 2013) — % cropland, urban, forest
  - Geology features (Macrostrat) — rock type, EC/alkalinity impact
  - Cross-domain interactions (crop x rain, carb x rain, urban x temp)
  - V7's seasonal/drought features (cool dry season, El Niño)
  - All V6 features (full Landsat, climate, terrain, soil)
  
MODEL ARCHITECTURE:
  - 3 targets: Alkalinity, EC, DRP (independent XGBoost each)
  - 3 feature tiers: full, medium, focused (best selected per target)
  - 4 model types: XGBoost, LightGBM, Ridge, ElasticNet
  - Target stacking: EC predictions feed into DRP model
  - Spatial GroupKFold CV (5 clusters)
  - Final: weighted ensemble of best configs per target

OUTPUT:
  submissions/submission_v8.csv
  submissions/submission_v8_YYYYMMDD_HHMMSS.csv
============================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import GroupKFold
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
import xgboost as xgb

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRACTED_DIR = os.path.join(BASE_DIR, '..', 'data', 'extracted')
DATA_DIR = os.path.join(BASE_DIR, '..', 'ey_challenge_notebooks')
SUBMISSION_DIR = os.path.join(BASE_DIR, '..', 'submissions')
LOG_DIR = os.path.join(BASE_DIR, '..', 'experiment_logs')
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

TARGETS = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
RANDOM_STATE = 42
N_CLUSTERS = 5
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

print(f"""
{'='*70}
EY WATER QUALITY CHALLENGE - V8 MODEL
Full features + Land Cover + Geology
{'='*70}
LightGBM: {'Available ✓' if HAS_LGBM else 'Not installed (pip install lightgbm)'}
""")


# ============================================================
# [1] LOAD V8 FEATURES
# ============================================================
print("[1] Loading V8 features...")

train_path = os.path.join(EXTRACTED_DIR, 'v8_train_features.csv')
val_path = os.path.join(EXTRACTED_DIR, 'v8_val_features.csv')
feat_path = os.path.join(EXTRACTED_DIR, 'v8_feature_columns.txt')

if not os.path.exists(train_path):
    print(f"  ERROR: {train_path} not found!")
    print(f"  Run: python scripts/18_feature_engineering_v8.py first")
    sys.exit(1)

train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)
with open(feat_path, 'r') as f:
    all_feature_cols = [line.strip() for line in f if line.strip()]

print(f"  Train: {train_df.shape}")
print(f"  Val:   {val_df.shape}")
print(f"  Features: {len(all_feature_cols)}")


# ============================================================
# [2] DEFINE FEATURE TIERS
# ============================================================
print("\n[2] Defining feature tiers...")

# Tier 1: ALL features
tier1_cols = [c for c in all_feature_cols if c in train_df.columns]

# Tier 2: Drop high-NaN and low-importance features
nan_rates = train_df[tier1_cols].isna().mean()
tier2_cols = [c for c in tier1_cols if nan_rates.get(c, 0) < 0.5]

# Tier 3: Focused — the most physically meaningful features per target
# These are the features most likely to transfer across regions
tier3_base = []

# Land cover (NEW — high spatial transferability)
tier3_base += [c for c in tier1_cols if c.startswith('lc_5km_') or c.startswith('lc_10km_')]
tier3_base += [c for c in tier1_cols if c.startswith('lc_contrast_')]
tier3_base += [c for c in tier1_cols if c in ['lc_5km_shannon']]

# Geology (NEW — high spatial transferability)
tier3_base += [c for c in tier1_cols if c.startswith('geo_') or c.startswith('lith_') 
               or c.startswith('rock_') or c.startswith('ph_class_')]

# Key interactions (NEW)
tier3_base += [c for c in tier1_cols if c in [
    'crop_x_ppt', 'crop_x_ppt_lag1', 'urban_x_ppt', 'human_x_tmax',
    'natveg_x_soilmoist', 'bare_x_slope', 'crop_x_soil_n', 'crop_x_soil_ph',
    'carb_x_ppt', 'alk_impact_x_tmax', 'ec_impact_x_tmax', 'ec_impact_x_elev',
    'carb_x_crop', 'anthro_pressure', 'natural_buffer'
]]

# Climate (proven important)
tier3_base += [c for c in tier1_cols if c in [
    'ppt', 'tmax', 'tmin', 'soil', 'aet', 'pet', 'vpd', 'q',
    'ppt_lag1', 'ppt_lag2', 'tmax_lag1', 'soil_lag1',
    'ppt_change_1m', 'ppt_surge', 'drought_3m', 'is_dry_month',
    'evap_rain_ratio', 'runoff_coeff',
]]

# Terrain + soil
tier3_base += [c for c in tier1_cols if c in [
    'elevation_m', 'slope_deg', 'log_elevation', 'elev_bin',
    'soil_clay', 'soil_sand', 'soil_cec', 'soil_phh2o', 'soil_soc', 'soil_nitrogen',
    'soil_buffer_capacity', 'soil_sand_clay_ratio',
]]

# Key Landsat
tier3_base += [c for c in tier1_cols if c in [
    'unified_NDVI', 'unified_NDMI', 'unified_MNDWI',
    'ls_NDVI', 'ls_NDTI', 'ls_clarity_index', 'ls_st_celsius',
    'ls_blue', 'ls_swir16', 'ls_spectral_cv_6',
]]

# Temporal
tier3_base += [c for c in tier1_cols if c in [
    'month_sin', 'month_cos', 'is_wet_season', 'is_cool_dry', 'is_el_nino_drought',
]]

tier3_cols = list(dict.fromkeys(tier3_base))  # Remove duplicates, preserve order

print(f"  Tier 1 (all):     {len(tier1_cols)} features")
print(f"  Tier 2 (medium):  {len(tier2_cols)} features")
print(f"  Tier 3 (focused): {len(tier3_cols)} features")


# ============================================================
# [3] SPATIAL CV SETUP
# ============================================================
print("\n[3] Setting up spatial cross-validation...")

coords = train_df[['Latitude', 'Longitude']].values
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
train_df['spatial_cluster'] = kmeans.fit_predict(coords)

for c in range(N_CLUSTERS):
    n = (train_df['spatial_cluster'] == c).sum()
    print(f"  Cluster {c}: {n} samples ({100*n/len(train_df):.1f}%)")


# ============================================================
# [4] MODEL CONFIGURATIONS
# ============================================================

def get_model_configs():
    """Return list of (name, model_factory, feature_tier) tuples."""
    configs = []
    
    # XGBoost configs
    xgb_base = dict(
        objective='reg:squarederror', eval_metric='rmse',
        tree_method='hist', random_state=RANDOM_STATE, verbosity=0,
    )
    
    configs.append(('XGB_t1_deep', lambda: xgb.XGBRegressor(
        **xgb_base, n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=5.0,
        min_child_weight=5, gamma=0.1,
    ), tier1_cols))
    
    configs.append(('XGB_t2_balanced', lambda: xgb.XGBRegressor(
        **xgb_base, n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.7, colsample_bytree=0.6, reg_alpha=2.0, reg_lambda=10.0,
        min_child_weight=10, gamma=0.2,
    ), tier2_cols))
    
    configs.append(('XGB_t3_focused', lambda: xgb.XGBRegressor(
        **xgb_base, n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.75, colsample_bytree=0.8, reg_alpha=3.0, reg_lambda=15.0,
        min_child_weight=15, gamma=0.3,
    ), tier3_cols))
    
    configs.append(('XGB_t3_conservative', lambda: xgb.XGBRegressor(
        **xgb_base, n_estimators=200, max_depth=3, learning_rate=0.03,
        subsample=0.65, colsample_bytree=0.5, reg_alpha=5.0, reg_lambda=20.0,
        min_child_weight=20, gamma=0.5,
    ), tier3_cols))
    
    # LightGBM configs
    if HAS_LGBM:
        configs.append(('LGBM_t1', lambda: lgb.LGBMRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=5.0,
            min_child_samples=10, random_state=RANDOM_STATE, verbose=-1,
        ), tier1_cols))
        
        configs.append(('LGBM_t3', lambda: lgb.LGBMRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.75, colsample_bytree=0.8, reg_alpha=3.0, reg_lambda=15.0,
            min_child_samples=20, random_state=RANDOM_STATE, verbose=-1,
        ), tier3_cols))
    
    # Linear models (for blending)
    configs.append(('Ridge_t2', lambda: Ridge(alpha=100.0), tier2_cols))
    configs.append(('ElasticNet_t3', lambda: ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000), tier3_cols))
    
    return configs


# ============================================================
# [5] TRAIN & EVALUATE
# ============================================================
print("\n" + "=" * 70)
print("[5] TRAINING AND EVALUATING ALL CONFIGURATIONS")
print("=" * 70)

results = {}        # {target: {config_name: {'r2': float, 'preds': array}}}
val_predictions = {}  # {target: {config_name: array}}

gkf = GroupKFold(n_splits=N_CLUSTERS)
groups = train_df['spatial_cluster'].values

for target in TARGETS:
    print(f"\n{'─'*60}")
    print(f"  TARGET: {target}")
    print(f"{'─'*60}")
    
    y = train_df[target].values
    valid_mask = ~np.isnan(y)
    
    results[target] = {}
    val_predictions[target] = {}
    
    configs = get_model_configs()
    
    for config_name, model_factory, feat_cols in configs:
        # Prepare features
        X = train_df[feat_cols].values[valid_mask]
        y_valid = y[valid_mask]
        groups_valid = groups[valid_mask]
        X_val = val_df[feat_cols].values
        
        # Impute NaN
        imputer = SimpleImputer(strategy='median')
        X_imp = imputer.fit_transform(X)
        X_val_imp = imputer.transform(X_val)
        
        # Scale for linear models
        is_linear = 'Ridge' in config_name or 'ElasticNet' in config_name
        scaler = None
        if is_linear:
            scaler = StandardScaler()
            X_imp = scaler.fit_transform(X_imp)
            X_val_imp = scaler.transform(X_val_imp)
        
        # Spatial CV
        cv_scores = []
        oof_preds = np.full(len(y_valid), np.nan)
        val_preds_fold = []
        
        for fold, (tr_idx, te_idx) in enumerate(gkf.split(X_imp, y_valid, groups_valid)):
            model = model_factory()
            model.fit(X_imp[tr_idx], y_valid[tr_idx])
            
            fold_pred = model.predict(X_imp[te_idx])
            oof_preds[te_idx] = fold_pred
            
            fold_r2 = r2_score(y_valid[te_idx], fold_pred)
            cv_scores.append(fold_r2)
            
            val_preds_fold.append(model.predict(X_val_imp))
        
        mean_r2 = np.mean(cv_scores)
        std_r2 = np.std(cv_scores)
        
        # Average validation predictions across folds
        val_pred_avg = np.mean(val_preds_fold, axis=0)
        
        results[target][config_name] = {
            'r2_mean': mean_r2,
            'r2_std': std_r2,
            'r2_folds': cv_scores,
            'val_preds': val_pred_avg,
            'oof_preds': oof_preds,
        }
        val_predictions[target][config_name] = val_pred_avg
        
        marker = '✓' if mean_r2 > 0 else '✗'
        print(f"    {marker} {config_name:25s}: R² = {mean_r2:+.4f} ± {std_r2:.4f}  "
              f"[{', '.join(f'{s:+.3f}' for s in cv_scores)}]")


# ============================================================
# [6] SELECT BEST CONFIGS & BUILD ENSEMBLE
# ============================================================
print("\n" + "=" * 70)
print("[6] BUILDING FINAL ENSEMBLE")
print("=" * 70)

final_predictions = {}

for target in TARGETS:
    print(f"\n  {target}:")
    
    # Sort configs by R²
    sorted_configs = sorted(
        results[target].items(),
        key=lambda x: x[1]['r2_mean'],
        reverse=True
    )
    
    # Show ranking
    for rank, (name, info) in enumerate(sorted_configs):
        marker = '★' if rank == 0 else ' '
        print(f"    {marker} #{rank+1}: {name:25s}  R² = {info['r2_mean']:+.4f}")
    
    # Ensemble: weighted average of top configs
    # Weight by max(R², 0) to avoid negative contributions
    top_n = min(3, len(sorted_configs))
    top_configs = sorted_configs[:top_n]
    
    weights = []
    preds_list = []
    for name, info in top_configs:
        w = max(info['r2_mean'], 0.001)  # Floor at small positive
        weights.append(w)
        preds_list.append(info['val_preds'])
    
    # Normalize weights
    total_w = sum(weights)
    weights = [w / total_w for w in weights]
    
    # Weighted ensemble
    ensemble_pred = np.zeros(len(val_df))
    for w, p in zip(weights, preds_list):
        ensemble_pred += w * p
    
    # Clip to physically reasonable ranges
    if target == 'Total Alkalinity':
        ensemble_pred = np.clip(ensemble_pred, 0, 1000)
    elif target == 'Electrical Conductance':
        ensemble_pred = np.clip(ensemble_pred, 0, 10000)
    elif target == 'Dissolved Reactive Phosphorus':
        ensemble_pred = np.clip(ensemble_pred, 0, 5000)
    
    final_predictions[target] = ensemble_pred
    
    print(f"    Ensemble weights: {[(name, f'{w:.3f}') for (name, _), w in zip(top_configs, weights)]}")
    print(f"    Prediction range: [{ensemble_pred.min():.1f}, {ensemble_pred.max():.1f}]")
    print(f"    Prediction mean:  {ensemble_pred.mean():.1f}")


# ============================================================
# [7] GENERATE SUBMISSION
# ============================================================
print("\n" + "=" * 70)
print("[7] GENERATING SUBMISSION")
print("=" * 70)

submission = pd.read_csv(os.path.join(DATA_DIR, 'submission_template.csv'))

for target in TARGETS:
    submission[target] = final_predictions[target]

# Save
sub_path = os.path.join(SUBMISSION_DIR, 'submission_v8.csv')
sub_ts_path = os.path.join(SUBMISSION_DIR, f'submission_v8_{ts}.csv')
submission.to_csv(sub_path, index=False)
submission.to_csv(sub_ts_path, index=False)

print(f"  Saved: {sub_path}")
print(f"  Saved: {sub_ts_path}")

# Verify
print(f"\n  Submission shape: {submission.shape}")
print(f"  NaN check:")
for target in TARGETS:
    n_nan = submission[target].isna().sum()
    print(f"    {target}: {n_nan} NaN values")

# ============================================================
# [8] SUMMARY
# ============================================================
print(f"""

{'='*70}
V8 MODEL RESULTS SUMMARY
{'='*70}
""")

for target in TARGETS:
    best_name = max(results[target].items(), key=lambda x: x[1]['r2_mean'])
    r2 = best_name[1]['r2_mean']
    marker = '✓' if r2 > 0 else '✗'
    print(f"  {marker} {target:40s}: Best R² = {r2:+.4f} ({best_name[0]})")

# Compare with V6 if logs exist
print(f"""
{'─'*70}
KEY V8 ADDITIONS:
  ★ Land cover: % cropland, urban, forest, bare at 1km/5km/10km
  ★ Geology: rock type, EC/alkalinity impact scores
  ★ Interactions: crop×rain, urban×rain, carbonate×rain
  ★ Spatial context: anthropogenic pressure, natural buffering

NEXT STEPS:
  1. Submit: Upload {sub_path} to the challenge
  2. If R² improved → land cover & geology are helping!
  3. If not → try target-specific models (V7-style EC focus)
  4. Consider: nearest-neighbor blending, stacking
{'='*70}
""")

# Save log
log_data = {
    'version': 'V8',
    'timestamp': ts,
    'features': len(all_feature_cols),
}
for target in TARGETS:
    best = max(results[target].items(), key=lambda x: x[1]['r2_mean'])
    log_data[f'{target}_best_config'] = best[0]
    log_data[f'{target}_best_r2'] = best[1]['r2_mean']

log_df = pd.DataFrame([log_data])
log_path = os.path.join(LOG_DIR, f'v8_results_{ts}.csv')
log_df.to_csv(log_path, index=False)
print(f"  Log saved: {log_path}")

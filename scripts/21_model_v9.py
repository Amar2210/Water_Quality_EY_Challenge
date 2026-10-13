"""
============================================================
EY Water Quality Challenge - V9 Model
============================================================
KEY INNOVATION: KNN spatial anchor + environmental adjustment

Architecture:
  1. KNN-IDW baseline: For each test point, interpolate from
     K nearest training stations (weighted by distance)
  2. Residual model: XGBoost/LightGBM predicts the RESIDUAL
     (actual - KNN prediction) using environmental features
  3. Final = KNN baseline + residual prediction
  4. Blend with pure ML predictions for robustness

This "anchor + adjust" approach should work because:
  - KNN gives a strong spatial baseline (captures regional patterns)
  - The residual model only needs to explain LOCAL deviations
    (seasonal variation, rainfall events, etc.)
  - Much easier to predict small residuals than raw values

Also includes:
  - Log-transform for DRP (right-skewed)
  - Target-specific feature tiers
  - Spatial GroupKFold CV
  - Ensemble of multiple approaches

HISTORICAL REPRODUCTION ONLY:
  KNN target features were computed before splitting into folds.
  The printed CV scores are not leakage-free spatial validation.
  This script does not automatically replace the saved best submission.
  See docs/REVIEW.md before interpreting the results.
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
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
import xgboost as xgb

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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
EY WATER QUALITY CHALLENGE - V9 MODEL
KNN Spatial Anchor + Residual Learning
{'='*70}
LightGBM: {'Available' if HAS_LGBM else 'Not installed'}
""")

# ============================================================
# [1] LOAD V9 FEATURES
# ============================================================
print("[1] Loading V9 features...")

train_path = os.path.join(EXTRACTED_DIR, 'v9_train_features.csv')
val_path = os.path.join(EXTRACTED_DIR, 'v9_val_features.csv')
feat_path = os.path.join(EXTRACTED_DIR, 'v9_feature_columns.txt')

if not os.path.exists(train_path):
    print("  ERROR: V9 features not found! Run 20_feature_engineering_v9.py first")
    sys.exit(1)

train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)
with open(feat_path, 'r') as f:
    all_feature_cols = [line.strip() for line in f if line.strip()]

# Only keep columns that actually exist
all_feature_cols = [c for c in all_feature_cols if c in train_df.columns and c in val_df.columns]

print(f"  Train: {train_df.shape}")
print(f"  Val:   {val_df.shape}")
print(f"  Features: {len(all_feature_cols)}")

# ============================================================
# [2] SPATIAL CV SETUP
# ============================================================
print("\n[2] Setting up historical spatial CV...")
print("  WARNING: Precomputed KNN features use full-training targets.")
print("  These scores are NOT leakage-free spatial CV; see docs/REVIEW.md.")

coords = train_df[['Latitude', 'Longitude']].values
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
train_df['spatial_cluster'] = kmeans.fit_predict(coords)
groups = train_df['spatial_cluster'].values

# ============================================================
# [3] DEFINE FEATURE TIERS PER TARGET
# ============================================================
print("\n[3] Defining feature tiers...")

# KNN features (the star of V9)
knn_cols = [c for c in all_feature_cols if c.startswith('knn') or c.startswith('dist_') 
            or c.startswith('spatial_anomaly') or c.startswith('log_dist') 
            or c.startswith('knn_confidence')]

# Environmental features (non-KNN)
env_cols = [c for c in all_feature_cols if c not in knn_cols]

print(f"  KNN features: {len(knn_cols)}")
print(f"  Environmental features: {len(env_cols)}")

# Target-specific KNN columns
def get_target_knn_cols(target):
    short = target.replace(' ', '_')
    return [c for c in knn_cols if short in c or 'dist_' in c or 'confidence' in c or 'log_dist' in c]

# Focused environmental features (most likely to transfer spatially)
focused_env = [c for c in env_cols if any(c.startswith(p) for p in [
    'lc_5km_', 'lc_10km_', 'lc_contrast_', 'geo_', 'lith_', 'rock_',
    'elevation_m', 'slope_deg', 'log_elevation',
    'soil_clay', 'soil_sand', 'soil_cec', 'soil_phh2o',
    'ppt', 'tmax', 'tmin', 'pet', 'vpd', 'aet',
    'unified_', 'ls_NDVI', 'ls_NDTI', 'ls_st_celsius',
    'month_sin', 'month_cos', 'is_wet_season', 'is_cool_dry',
    'anthro_pressure', 'natural_buffer', 'human_impact', 'crop_x_ppt',
    'urban_x_ppt', 'carb_x_ppt',
])]

print(f"  Focused environmental: {len(focused_env)}")

# ============================================================
# [4] APPROACH A: PURE ML WITH KNN FEATURES
# ============================================================
print("\n" + "=" * 70)
print("[4] APPROACH A: Pure ML with KNN + all features")
print("=" * 70)

gkf = GroupKFold(n_splits=N_CLUSTERS)

approach_a_results = {}
approach_a_val_preds = {}

for target in TARGETS:
    print(f"\n  {'─'*50}")
    print(f"  TARGET: {target}")
    print(f"  {'─'*50}")
    
    y = train_df[target].values
    valid_mask = ~np.isnan(y)
    y_valid = y[valid_mask]
    groups_valid = groups[valid_mask]
    
    # Use log-transform for DRP (highly skewed)
    use_log = target == 'Dissolved Reactive Phosphorus'
    if use_log:
        y_train_transformed = np.log1p(np.clip(y_valid, 0, None))
    else:
        y_train_transformed = y_valid
    
    # Feature sets to try
    target_knn = get_target_knn_cols(target)
    
    feature_sets = {
        'knn_only': target_knn,
        'knn_env_focused': target_knn + focused_env,
        'knn_env_all': target_knn + env_cols,
        'all_features': all_feature_cols,
    }
    
    best_r2 = -999
    best_config = None
    best_val_pred = None
    
    for feat_name, feat_cols in feature_sets.items():
        feat_cols = [c for c in feat_cols if c in train_df.columns]
        if not feat_cols:
            continue
            
        X = train_df[feat_cols].values[valid_mask]
        X_val = val_df[feat_cols].values
        
        imputer = SimpleImputer(strategy='median')
        X_imp = imputer.fit_transform(X)
        X_val_imp = imputer.transform(X_val)
        
        # XGBoost with regularization
        model_configs = [
            ('XGB_reg', lambda: xgb.XGBRegressor(
                n_estimators=400, max_depth=5, learning_rate=0.05,
                subsample=0.75, colsample_bytree=0.7,
                reg_alpha=2.0, reg_lambda=10.0, min_child_weight=10, gamma=0.2,
                tree_method='hist', random_state=RANDOM_STATE, verbosity=0,
            )),
            ('XGB_cons', lambda: xgb.XGBRegressor(
                n_estimators=200, max_depth=3, learning_rate=0.03,
                subsample=0.65, colsample_bytree=0.5,
                reg_alpha=5.0, reg_lambda=20.0, min_child_weight=20, gamma=0.5,
                tree_method='hist', random_state=RANDOM_STATE, verbosity=0,
            )),
        ]
        
        if HAS_LGBM:
            model_configs.append(('LGBM', lambda: lgb.LGBMRegressor(
                n_estimators=400, max_depth=5, learning_rate=0.05,
                subsample=0.75, colsample_bytree=0.7,
                reg_alpha=2.0, reg_lambda=10.0, min_child_samples=15,
                random_state=RANDOM_STATE, verbose=-1,
            )))
        
        for model_name, model_factory in model_configs:
            cv_scores = []
            val_preds_fold = []
            
            for fold, (tr_idx, te_idx) in enumerate(gkf.split(X_imp, y_train_transformed, groups_valid)):
                model = model_factory()
                model.fit(X_imp[tr_idx], y_train_transformed[tr_idx])
                
                fold_pred = model.predict(X_imp[te_idx])
                
                if use_log:
                    fold_pred_orig = np.expm1(fold_pred)
                    fold_actual = y_valid[te_idx]
                else:
                    fold_pred_orig = fold_pred
                    fold_actual = y_valid[te_idx]
                
                fold_r2 = r2_score(fold_actual, fold_pred_orig)
                cv_scores.append(fold_r2)
                
                val_fold_pred = model.predict(X_val_imp)
                if use_log:
                    val_fold_pred = np.expm1(val_fold_pred)
                val_preds_fold.append(val_fold_pred)
            
            mean_r2 = np.mean(cv_scores)
            config_key = f'{feat_name}_{model_name}'
            
            marker = '✓' if mean_r2 > 0 else '✗'
            print(f"    {marker} {config_key:35s}: R² = {mean_r2:+.4f}  "
                  f"[{', '.join(f'{s:+.3f}' for s in cv_scores)}]")
            
            if mean_r2 > best_r2:
                best_r2 = mean_r2
                best_config = config_key
                best_val_pred = np.mean(val_preds_fold, axis=0)
    
    approach_a_results[target] = {'r2': best_r2, 'config': best_config}
    approach_a_val_preds[target] = best_val_pred
    print(f"\n    ★ Best: {best_config} (R² = {best_r2:+.4f})")


# ============================================================
# [5] APPROACH B: KNN BASELINE + RESIDUAL MODEL
# ============================================================
print("\n" + "=" * 70)
print("[5] APPROACH B: KNN Baseline + Residual Model")
print("=" * 70)

approach_b_results = {}
approach_b_val_preds = {}

for target in TARGETS:
    print(f"\n  {'─'*50}")
    print(f"  TARGET: {target}")
    print(f"  {'─'*50}")
    
    y = train_df[target].values
    valid_mask = ~np.isnan(y)
    y_valid = y[valid_mask]
    groups_valid = groups[valid_mask]
    
    short = target.replace(' ', '_')
    idw_col = f'knn5_{short}_idw'
    
    if idw_col not in train_df.columns:
        print(f"    SKIP: {idw_col} not found")
        approach_b_results[target] = {'r2': -999}
        approach_b_val_preds[target] = approach_a_val_preds[target]
        continue
    
    # KNN baseline (IDW with K=5)
    knn_baseline_train = train_df[idw_col].values[valid_mask]
    knn_baseline_val = val_df[idw_col].values
    
    # Compute residuals: what the KNN baseline gets wrong
    residuals = y_valid - knn_baseline_train
    
    # How good is the pure KNN baseline?
    knn_r2 = r2_score(y_valid[~np.isnan(knn_baseline_train)], 
                       knn_baseline_train[~np.isnan(knn_baseline_train)])
    print(f"    Pure KNN-IDW baseline R²: {knn_r2:+.4f}")
    
    # Now train a model to predict the residual from environmental features
    X_env = train_df[focused_env].values[valid_mask]
    X_env_val = val_df[focused_env].values
    
    # Remove rows where KNN baseline is NaN
    has_knn = ~np.isnan(knn_baseline_train)
    X_env_clean = X_env[has_knn]
    residuals_clean = residuals[has_knn]
    groups_clean = groups_valid[has_knn]
    y_clean = y_valid[has_knn]
    knn_clean = knn_baseline_train[has_knn]
    
    imputer = SimpleImputer(strategy='median')
    X_env_imp = imputer.fit_transform(X_env_clean)
    X_env_val_imp = imputer.transform(X_env_val)
    
    # Train residual model
    best_blend_r2 = -999
    best_alpha = 0
    best_residual_pred_val = None
    
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        # alpha = 0: pure KNN, alpha = 1: full residual correction
        cv_scores = []
        val_residual_preds = []
        
        for fold, (tr_idx, te_idx) in enumerate(gkf.split(X_env_imp, residuals_clean, groups_clean)):
            if alpha > 0:
                model = xgb.XGBRegressor(
                    n_estimators=200, max_depth=3, learning_rate=0.03,
                    subsample=0.7, colsample_bytree=0.6,
                    reg_alpha=5.0, reg_lambda=20.0, min_child_weight=15,
                    tree_method='hist', random_state=RANDOM_STATE, verbosity=0,
                )
                model.fit(X_env_imp[tr_idx], residuals_clean[tr_idx])
                residual_pred = model.predict(X_env_imp[te_idx])
                val_residual_preds.append(model.predict(X_env_val_imp))
            else:
                residual_pred = np.zeros(len(te_idx))
                val_residual_preds.append(np.zeros(len(X_env_val_imp)))
            
            # Blend: KNN + alpha * residual_correction
            fold_pred = knn_clean[te_idx] + alpha * residual_pred
            fold_r2 = r2_score(y_clean[te_idx], fold_pred)
            cv_scores.append(fold_r2)
        
        mean_r2 = np.mean(cv_scores)
        marker = '✓' if mean_r2 > 0 else '✗'
        print(f"    {marker} alpha={alpha:.1f}: R² = {mean_r2:+.4f}  "
              f"[{', '.join(f'{s:+.3f}' for s in cv_scores)}]")
        
        if mean_r2 > best_blend_r2:
            best_blend_r2 = mean_r2
            best_alpha = alpha
            best_residual_pred_val = np.mean(val_residual_preds, axis=0)
    
    # Final prediction: KNN + best_alpha * residual
    final_val_pred = knn_baseline_val + best_alpha * best_residual_pred_val
    
    # Fill NaN KNN with approach A prediction
    nan_mask = np.isnan(final_val_pred)
    if nan_mask.any():
        final_val_pred[nan_mask] = approach_a_val_preds[target][nan_mask]
    
    approach_b_results[target] = {'r2': best_blend_r2, 'alpha': best_alpha}
    approach_b_val_preds[target] = final_val_pred
    print(f"\n    ★ Best alpha={best_alpha:.1f}, R² = {best_blend_r2:+.4f}")


# ============================================================
# [6] FINAL ENSEMBLE: BEST OF A AND B
# ============================================================
print("\n" + "=" * 70)
print("[6] FINAL ENSEMBLE")
print("=" * 70)

final_predictions = {}

for target in TARGETS:
    r2_a = approach_a_results[target]['r2']
    r2_b = approach_b_results[target].get('r2', -999)
    
    print(f"\n  {target}:")
    print(f"    Approach A (pure ML):            R² = {r2_a:+.4f}")
    print(f"    Approach B (KNN + residual):     R² = {r2_b:+.4f}")
    
    # If one approach is much better, lean heavily toward it
    if r2_b > r2_a + 0.05:
        # B is clearly better
        blend = 0.3 * approach_a_val_preds[target] + 0.7 * approach_b_val_preds[target]
        print(f"    → Using 70% B + 30% A (B is better)")
    elif r2_a > r2_b + 0.05:
        # A is clearly better
        blend = 0.7 * approach_a_val_preds[target] + 0.3 * approach_b_val_preds[target]
        print(f"    → Using 70% A + 30% B (A is better)")
    else:
        # Similar performance, equal blend
        blend = 0.5 * approach_a_val_preds[target] + 0.5 * approach_b_val_preds[target]
        print(f"    → Using 50/50 blend")
    
    # Clip to physical ranges
    if target == 'Total Alkalinity':
        blend = np.clip(blend, 0, 1000)
    elif target == 'Electrical Conductance':
        blend = np.clip(blend, 0, 10000)
    elif target == 'Dissolved Reactive Phosphorus':
        blend = np.clip(blend, 0, 5000)
    
    final_predictions[target] = blend
    print(f"    Prediction range: [{blend.min():.1f}, {blend.max():.1f}], mean={blend.mean():.1f}")


# ============================================================
# [7] GENERATE SUBMISSION
# ============================================================
print("\n" + "=" * 70)
print("[7] GENERATING SUBMISSION")
print("=" * 70)

submission = pd.read_csv(os.path.join(DATA_DIR, 'submission_template.csv'))
for target in TARGETS:
    submission[target] = final_predictions[target]

sub_path = os.path.join(SUBMISSION_DIR, 'submission_v9.csv')
submission.to_csv(sub_path, index=False)

print(f"  Saved: {sub_path}")
print("  Saved best_submission/submission_v9.csv is unchanged.")

# NaN check
for target in TARGETS:
    n_nan = submission[target].isna().sum()
    if n_nan > 0:
        print(f"  WARNING: {target} has {n_nan} NaN values!")

# ============================================================
# [8] SUMMARY
# ============================================================
print(f"""

{'='*70}
V9 MODEL RESULTS SUMMARY
{'='*70}
""")

for target in TARGETS:
    r2_a = approach_a_results[target]['r2']
    r2_b = approach_b_results[target].get('r2', -999)
    best = max(r2_a, r2_b)
    marker = '✓' if best > 0 else '✗'
    approach = 'ML' if r2_a > r2_b else 'KNN+Res'
    print(f"  {marker} {target:40s}: Best R² = {best:+.4f} ({approach})")

print(f"""
{'─'*70}
KEY V9 INNOVATIONS:
  ★ KNN target features: "What is water quality at nearby stations?"
  ★ IDW spatial interpolation as baseline
  ★ Residual learning: model predicts deviation from KNN baseline
  ★ KNN + environmental blend with optimal alpha
  ★ Log-transform for DRP

Submit {sub_path} and compare with V8!
{'='*70}
""")

# Save log
log_data = {
    'version': 'V9', 'timestamp': ts,
    'evaluation_status': 'precomputed_full_training_neighbor_features',
}
for target in TARGETS:
    log_data[f'{target}_approach_a_r2'] = approach_a_results[target]['r2']
    log_data[f'{target}_approach_b_r2'] = approach_b_results[target].get('r2', np.nan)
pd.DataFrame([log_data]).to_csv(os.path.join(LOG_DIR, f'v9_results_{ts}.csv'), index=False)

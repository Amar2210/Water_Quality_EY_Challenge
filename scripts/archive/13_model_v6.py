"""
============================================================
EY Water Quality Challenge - V6 Model
============================================================
WHAT'S NEW IN V6:
  - Full Landsat bands (blue, red, thermal, coastal)
  - Real NDVI, turbidity indices, surface temperature
  - Hybrid Landsat features (full + EY backup)
  - Dropped useless Sentinel-2 (97.5% NaN)
  - Same proven model architecture from V5
  - THREE feature tiers x THREE XGB configs
  - Target stacking: EC -> DRP
  - Spatial GroupKFold CV

EXPECTED IMPROVEMENT:
  - Blue band -> better turbidity prediction -> Alkalinity
  - Red band -> real NDVI -> better vegetation/nutrient proxy -> DRP
  - Surface temperature -> direct EC correlation
  - Fewer NaN features -> more stable training

OUTPUT:
  submissions/submission_v6.csv
  submissions/submission_v6_YYYYMMDD_HHMMSS.csv
============================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import warnings
import os
import time

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRACTED_DIR = os.path.join(BASE_DIR, '..', 'data', 'extracted')
DATA_DIR = os.path.join(BASE_DIR, '..', 'ey_challenge_notebooks')
SUBMISSION_DIR = os.path.join(BASE_DIR, '..', 'submissions')
os.makedirs(SUBMISSION_DIR, exist_ok=True)

TARGETS = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
RANDOM_STATE = 42
N_CLUSTERS = 5

print("=" * 70)
print("EY WATER QUALITY CHALLENGE - V6 MODEL")
print("Full Landsat bands + thermal + V4 climate/terrain/soil")
print("=" * 70)


# ============================================================
# [1] LOAD V6 FEATURES
# ============================================================
print("\n" + "=" * 70)
print("[1] LOADING V6 FEATURE-ENGINEERED DATA")
print("=" * 70)

train_df = pd.read_csv(os.path.join(EXTRACTED_DIR, 'v6_train_features.csv'))
val_df = pd.read_csv(os.path.join(EXTRACTED_DIR, 'v6_val_features.csv'))

with open(os.path.join(EXTRACTED_DIR, 'v6_feature_columns.txt'), 'r') as f:
    all_feature_cols = [line.strip() for line in f if line.strip()]

print(f"  Train: {train_df.shape}")
print(f"  Val:   {val_df.shape}")
print(f"  Total available features: {len(all_feature_cols)}")

# Coverage stats
if 'has_full_landsat' in train_df.columns:
    fl_train = train_df['has_full_landsat'].mean() * 100
    fl_val = val_df['has_full_landsat'].mean() * 100
    print(f"  Full Landsat coverage: train={fl_train:.1f}%, val={fl_val:.1f}%")
if 'has_surface_temp' in train_df.columns:
    st_train = train_df['has_surface_temp'].mean() * 100
    st_val = val_df['has_surface_temp'].mean() * 100 if 'has_surface_temp' in val_df.columns else 0
    print(f"  Surface temp coverage: train={st_train:.1f}%, val={st_val:.1f}%")


# ============================================================
# [2] DEFINE THREE FEATURE TIERS
# ============================================================
print("\n" + "=" * 70)
print("[2] DEFINING FEATURE TIERS")
print("=" * 70)

# ── TIER 1 — CORE (~35 features) ──
# Best proven features + key new Landsat bands
TIER1_FEATURES = [
    # Full Landsat bands (new)
    'ls_blue', 'ls_green', 'ls_red', 'ls_nir', 'ls_swir16', 'ls_swir22',
    'ls_st_kelvin',
    # Key Full Landsat indices
    'ls_NDVI', 'ls_NDWI', 'ls_NDMI', 'ls_MNDWI', 'ls_NDTI',
    # EY original bands (backup/complementary)
    'ey_nir', 'ey_green', 'ey_swir16', 'ey_swir22', 'ey_NDMI', 'ey_MNDWI',
    # Unified bands
    'unified_NDVI', 'unified_NDMI', 'unified_MNDWI',
    # Terrain
    'elevation_m', 'slope_deg',
    # Climate current
    'ppt', 'tmax', 'tmin', 'pet', 'q', 'vpd',
    # Soil
    'soil_phh2o', 'soil_cec', 'soil_clay', 'soil_sand', 'soil_soc',
    # Temporal
    'month_sin', 'month_cos',
    # Flags
    'has_full_landsat', 'has_surface_temp',
]

# ── TIER 2 — EXTENDED (~65 features) ──
TIER2_FEATURES = TIER1_FEATURES + [
    # Additional Landsat indices
    'ls_coastal', 'ls_blue_green_ratio', 'ls_blue_red_ratio',
    'ls_NSMI', 'ls_clarity_index', 'ls_sediment_index',
    'ls_chlor_ratio', 'ls_WRI',
    'ls_spectral_mean_6', 'ls_spectral_std_6', 'ls_vis_mean',
    # Surface temp celsius + interactions
    'ls_st_celsius',
    'ls_st_x_swir16', 'ls_st_x_NDMI',
    # 1-month lags
    'ppt_lag1', 'tmax_lag1', 'soil_lag1', 'q_lag1',
    # Key climate derived
    'aridity_index', 'temp_range', 'ppt_surge',
    'is_wet_season',
    # Key interactions
    'slope_x_ppt', 'slope_x_clay',
    'elev_x_tmax', 'elev_x_ppt',
    'ndvi_x_ppt', 'ndti_x_ppt',
    'st_x_ppt', 'st_x_elev',
    # V4 legacy ratios
    'ratio_nir_green', 'ndvi_proxy',
    'soil_sand_clay_ratio',
    # Platform
    'ls_platform_code', 'ls_days_offset',
]

# ── TIER 3 — FULL (everything) ──
TIER3_FEATURES = TIER2_FEATURES + [
    # All remaining full Landsat
    'ls_nir_red_ratio', 'ls_green_swir22_ratio',
    'ls_AWEI_nsh', 'ls_spectral_cv_6', 'ls_vis_std',
    'ls_total_brightness', 'ls_vis_brightness',
    'ls_coastal_blue_ratio', 'has_coastal',
    'ls_st_x_nir',
    # Unified
    'unified_nir', 'unified_green', 'unified_swir16', 'unified_swir22',
    # All lags
    'ppt_lag2', 'tmax_lag2', 'soil_lag2', 'q_lag2',
    'ppt_lag3', 'tmax_lag3', 'soil_lag3', 'q_lag3',
    'ppt_3m_avg', 'ppt_3m_sum', 'tmax_3m_avg', 'tmin_3m_avg',
    # More interactions
    'st_x_vpd', 'st_x_clay',
    'blue_x_slope', 'ndvi_x_tmax',
    'clarity_x_sand',
    'moisture_x_clay', 'soilph_x_ppt',
    'mndwi_x_ppt', 'ndmi_x_ppt', 'swir16_x_tmax',
    # Climate derived
    'is_dry_month', 'ppt_change_1m', 'ppt_change_2m', 'drought_3m',
    'evap_rain_ratio', 'evap_deficit_ratio',
    'vpd_change_1m', 'soil_moisture_change', 'runoff_change', 'runoff_coeff',
    'pdsi_trend', 'pdsi', 'pdsi_lag1',
    'aet', 'def', 'srad',
    # V4 legacy indices
    'ratio_swir16_nir', 'ratio_swir22_swir16', 'ratio_green_swir16',
    'ratio_nir_swir22', 'ndsi_swir', 'bsi_proxy', 'ndbi_proxy',
    'water_brightness', 'spectral_std', 'spectral_mean', 'spectral_cv',
    # Soil derived
    'soil_nitrogen', 'soil_buffer_capacity', 'soil_cn_ratio',
    # Terrain extras
    'aspect_deg', 'terrain_roughness', 'elev_range_500m', 'log_elevation', 'elev_bin',
    # Extra temporal
    'month', 'year', 'day_of_year', 'doy_sin', 'doy_cos', 'season',
    # Cloud cover from extraction
    'ls_cloud_cover',
]


def get_available(feature_list, df):
    seen = set()
    available = []
    for f in feature_list:
        if f in seen:
            continue
        seen.add(f)
        if f in df.columns:
            available.append(f)
    return available


tier1 = get_available(TIER1_FEATURES, train_df)
tier2 = get_available(TIER2_FEATURES, train_df)
tier3 = get_available(TIER3_FEATURES, train_df)

print(f"  TIER 1 (CORE):     {len(tier1)} features")
print(f"  TIER 2 (EXTENDED): {len(tier2)} features")
print(f"  TIER 3 (FULL):     {len(tier3)} features")


# ============================================================
# [3] SPATIAL CLUSTERING
# ============================================================
print("\n" + "=" * 70)
print("[3] SPATIAL CLUSTERING FOR CV")
print("=" * 70)

coords = train_df[['Latitude', 'Longitude']].values
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
train_df['spatial_cluster'] = kmeans.fit_predict(coords)

for cid in range(N_CLUSTERS):
    mask = train_df['spatial_cluster'] == cid
    center = coords[mask].mean(axis=0)
    print(f"  Cluster {cid}: {mask.sum():4d} samples ({center[0]:.1f}S, {center[1]:.1f}E)")


# ============================================================
# [4] MODEL CONFIGS
# ============================================================

XGB_CONSERVATIVE = {
    'n_estimators': 500, 'max_depth': 4, 'learning_rate': 0.05,
    'min_child_weight': 20, 'subsample': 0.7, 'colsample_bytree': 0.7,
    'reg_alpha': 1.0, 'reg_lambda': 5.0, 'random_state': RANDOM_STATE,
    'n_jobs': -1,
}
XGB_MODERATE = {
    'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.05,
    'min_child_weight': 10, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'reg_alpha': 0.5, 'reg_lambda': 3.0, 'random_state': RANDOM_STATE,
    'n_jobs': -1,
}
XGB_AGGRESSIVE = {
    'n_estimators': 800, 'max_depth': 6, 'learning_rate': 0.03,
    'min_child_weight': 5, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'reg_alpha': 0.1, 'reg_lambda': 1.0, 'random_state': RANDOM_STATE,
    'n_jobs': -1,
}


# ============================================================
# [5] SPATIAL CV EVALUATION
# ============================================================
print("\n" + "=" * 70)
print("[5] SPATIAL CV: ALL TIERS x ALL CONFIGS")
print("=" * 70)

groups = train_df['spatial_cluster']

TIER_MAP = {'T1_CORE': tier1, 'T2_EXTENDED': tier2, 'T3_FULL': tier3}
XGB_MAP = {'conservative': XGB_CONSERVATIVE, 'moderate': XGB_MODERATE, 'aggressive': XGB_AGGRESSIVE}

results = {}
best_configs = {}


def run_xgb_cv(X, y, grps, feat_names, use_log=False, params=None):
    gkf = GroupKFold(n_splits=N_CLUSTERS)
    fold_scores = []
    fold_imps = []

    for train_idx, test_idx in gkf.split(X, y, grps):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        y_tr_m = np.log1p(y_tr) if use_log else y_tr

        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr_m, verbose=False)

        preds = model.predict(X_te)
        if use_log:
            preds = np.expm1(preds)
        preds = np.maximum(preds, 0)

        fold_scores.append(r2_score(y_te, preds))
        fold_imps.append(model.feature_importances_)

    return np.mean(fold_scores), fold_scores, dict(zip(feat_names, np.mean(fold_imps, axis=0)))


def run_linear_cv(X, y, grps, feat_names, use_log=False, model_type='ridge'):
    gkf = GroupKFold(n_splits=N_CLUSTERS)
    fold_scores = []

    for train_idx, test_idx in gkf.split(X, y, grps):
        X_tr, X_te = X.iloc[train_idx].values, X.iloc[test_idx].values
        y_tr, y_te = y.iloc[train_idx].values, y.iloc[test_idx].values

        y_tr_m = np.log1p(y_tr) if use_log else y_tr

        est = Ridge(alpha=1.0) if model_type == 'ridge' else ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000)
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', est),
        ])
        pipe.fit(X_tr, y_tr_m)
        preds = pipe.predict(X_te)
        if use_log:
            preds = np.expm1(preds)
        preds = np.maximum(preds, 0)
        fold_scores.append(r2_score(y_te, preds))

    return np.mean(fold_scores), fold_scores, {}


for target in TARGETS:
    print(f"\n{'─' * 70}")
    print(f"  TARGET: {target}")
    print(f"{'─' * 70}")

    y = train_df[target]
    skew = y.skew()
    try_log = skew > 1.0
    print(f"  Skewness: {skew:.2f} -> {'log transform' if try_log else 'no log'}")

    configs = {}

    # XGBoost configs
    for tier_name, feat_list in TIER_MAP.items():
        for param_name, params in XGB_MAP.items():
            X = train_df[feat_list]

            name = f"{tier_name}_{param_name}"
            r2, folds, imps = run_xgb_cv(X, y, groups, feat_list, use_log=False, params=params)
            configs[name] = {
                'r2': r2, 'folds': folds, 'features': feat_list,
                'params': params, 'use_log': False, 'importances': imps, 'model_type': 'xgb',
            }

            if try_log:
                name_log = f"{tier_name}_{param_name}_log"
                r2_l, folds_l, imps_l = run_xgb_cv(X, y, groups, feat_list, use_log=True, params=params)
                configs[name_log] = {
                    'r2': r2_l, 'folds': folds_l, 'features': feat_list,
                    'params': params, 'use_log': True, 'importances': imps_l, 'model_type': 'xgb',
                }

    # Linear baselines on CORE tier
    for mt in ['ridge', 'elasticnet']:
        X = train_df[tier1]
        name = f"T1_CORE_{mt}"
        r2, folds, imps = run_linear_cv(X, y, groups, tier1, use_log=False, model_type=mt)
        configs[name] = {'r2': r2, 'folds': folds, 'features': tier1, 'params': {'model_type': mt},
                         'use_log': False, 'importances': imps, 'model_type': mt}
        if try_log:
            name_log = f"T1_CORE_{mt}_log"
            r2_l, folds_l, _ = run_linear_cv(X, y, groups, tier1, use_log=True, model_type=mt)
            configs[name_log] = {'r2': r2_l, 'folds': folds_l, 'features': tier1, 'params': {'model_type': mt},
                                 'use_log': True, 'importances': {}, 'model_type': mt}

    # Print sorted results
    print(f"\n  {'Config':<40s} {'R2':>8s}  Fold scores")
    print(f"  {'─'*40} {'─'*8}  {'─'*45}")
    sorted_configs = sorted(configs.items(), key=lambda x: x[1]['r2'], reverse=True)
    for i, (cname, info) in enumerate(sorted_configs):
        fold_str = ', '.join([f"{f:+.3f}" for f in info['folds']])
        marker = " <-- BEST" if i == 0 else ""
        print(f"  {cname:<40s} {info['r2']:+.4f}  [{fold_str}]{marker}")

    best_name, best_info = sorted_configs[0]
    best_configs[target] = best_info
    best_configs[target]['config_name'] = best_name
    results[target] = best_info['r2']

    # Top features
    if best_info['importances']:
        print(f"\n  Top 15 features ({best_name}):")
        sorted_imps = sorted(best_info['importances'].items(), key=lambda x: x[1], reverse=True)
        for rank, (feat, imp) in enumerate(sorted_imps[:15], 1):
            bar = '█' * int(imp * 60)
            new_flag = " [NEW]" if feat.startswith('ls_') or feat.startswith('unified_') else ""
            print(f"    {rank:2d}. {feat:35s} {imp:.4f} {bar}{new_flag}")


# ============================================================
# [6] TARGET STACKING: EC -> DRP
# ============================================================
print(f"\n{'=' * 70}")
print(f"[6] TARGET STACKING: EC -> DRP")
print(f"{'=' * 70}")

target_drp = 'Dissolved Reactive Phosphorus'
target_ec = 'Electrical Conductance'
y_drp = train_df[target_drp]
y_ec = train_df[target_ec]

ec_cfg = best_configs[target_ec]
drp_cfg = best_configs[target_drp]

use_stacking = False
if ec_cfg['model_type'] == 'xgb' and drp_cfg['model_type'] == 'xgb':
    ec_feats = ec_cfg['features']
    drp_feats = drp_cfg['features']
    drp_log = drp_cfg['use_log']
    ec_params = ec_cfg['params']
    drp_params = drp_cfg['params']

    gkf = GroupKFold(n_splits=N_CLUSTERS)
    stacked_scores = []

    for train_idx, test_idx in gkf.split(train_df, y_drp, groups):
        # Train EC
        ec_model = xgb.XGBRegressor(**ec_params)
        ec_model.fit(train_df.iloc[train_idx][ec_feats], y_ec.iloc[train_idx], verbose=False)
        ec_pred_tr = ec_model.predict(train_df.iloc[train_idx][ec_feats])
        ec_pred_te = ec_model.predict(train_df.iloc[test_idx][ec_feats])

        # Train DRP with predicted EC
        X_drp_tr = train_df.iloc[train_idx][drp_feats].copy()
        X_drp_te = train_df.iloc[test_idx][drp_feats].copy()
        X_drp_tr['predicted_EC'] = ec_pred_tr
        X_drp_te['predicted_EC'] = ec_pred_te

        y_drp_tr = np.log1p(y_drp.iloc[train_idx]) if drp_log else y_drp.iloc[train_idx]
        drp_model = xgb.XGBRegressor(**drp_params)
        drp_model.fit(X_drp_tr, y_drp_tr, verbose=False)

        preds = drp_model.predict(X_drp_te)
        if drp_log:
            preds = np.expm1(preds)
        preds = np.maximum(preds, 0)
        stacked_scores.append(r2_score(y_drp.iloc[test_idx], preds))

    stacked_r2 = np.mean(stacked_scores)
    original_r2 = results[target_drp]

    print(f"  DRP without stacking: R2 = {original_r2:+.4f}")
    print(f"  DRP with EC stacking: R2 = {stacked_r2:+.4f}")
    print(f"  Improvement: {stacked_r2 - original_r2:+.4f}")

    use_stacking = stacked_r2 > original_r2
    if use_stacking:
        results[target_drp] = stacked_r2
        print(f"  -> STACKING HELPS! Using it.")
    else:
        print(f"  -> Stacking did not help. Skipping.")
else:
    print(f"  Skipping (best models not both XGBoost)")


# ============================================================
# [7] RESULTS SUMMARY
# ============================================================
print(f"\n{'=' * 70}")
print(f"[7] V6 SPATIAL CV RESULTS")
print(f"{'=' * 70}")

for target in TARGETS:
    r2 = results[target]
    cfg = best_configs[target]
    cname = cfg.get('config_name', '')
    log_flag = " (log)" if cfg['use_log'] else ""
    stack_flag = " + stacking" if target == target_drp and use_stacking else ""
    print(f"  {target:40s}: R2 = {r2:+.4f}  ({cname}{log_flag}{stack_flag})")

mean_r2 = np.mean(list(results.values()))
print(f"  {'─' * 60}")
print(f"  {'MEAN R2':40s}: {mean_r2:+.4f}")

print(f"\n  Version history (Spatial CV Mean R2):")
print(f"    V0.1 (benchmark):      ~-0.40")
print(f"    V4   (research-backed): ~-0.35")
print(f"    V5   (V4 + Sentinel-2): ~-0.35 (S2 was 97.5% NaN)")
print(f"    V6   (full Landsat):    {mean_r2:+.4f}")


# ============================================================
# [8] TRAIN FINAL MODELS + SUBMISSION
# ============================================================
print(f"\n{'=' * 70}")
print(f"[8] TRAINING FINAL MODELS + GENERATING SUBMISSION")
print(f"{'=' * 70}")

submission = pd.read_csv(os.path.join(DATA_DIR, 'submission_template.csv'))
final_models = {}

for target in TARGETS:
    cfg = best_configs[target]
    features = cfg['features']
    use_log = cfg['use_log']
    model_type = cfg['model_type']

    print(f"\n  {target}: {cfg.get('config_name', '')} ({len(features)} features, log={use_log})")

    X_train = train_df[features]
    y_train = train_df[target]
    y_model = np.log1p(y_train) if use_log else y_train

    if model_type == 'xgb':
        model = xgb.XGBRegressor(**cfg['params'])
        model.fit(X_train, y_model, verbose=False)
    else:
        est = Ridge(alpha=1.0) if model_type == 'ridge' else ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000)
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', est),
        ])
        model.fit(X_train.values, y_model.values)

    final_models[target] = {'model': model, 'features': features, 'use_log': use_log, 'model_type': model_type}

    if target != target_drp or not use_stacking:
        X_val = val_df[features]
        preds = model.predict(X_val) if model_type == 'xgb' else model.predict(X_val.values)
        if use_log:
            preds = np.expm1(preds)
        preds = np.maximum(preds, 0)
        submission[target] = preds
        print(f"    Predictions: [{preds.min():.1f}, {preds.max():.1f}], mean={preds.mean():.1f}")

# Stacked DRP
if use_stacking:
    print(f"\n  DRP with target stacking...")
    ec_info = final_models[target_ec]
    ec_model = ec_info['model']
    ec_feats = ec_info['features']

    ec_preds_train = ec_model.predict(train_df[ec_feats])
    ec_preds_val = ec_model.predict(val_df[ec_feats])

    drp_feats = drp_cfg['features']
    X_drp_train = train_df[drp_feats].copy()
    X_drp_train['predicted_EC'] = ec_preds_train
    y_drp_model = np.log1p(train_df[target_drp]) if drp_cfg['use_log'] else train_df[target_drp]

    stacked_model = xgb.XGBRegressor(**drp_cfg['params'])
    stacked_model.fit(X_drp_train, y_drp_model, verbose=False)

    X_drp_val = val_df[drp_feats].copy()
    X_drp_val['predicted_EC'] = ec_preds_val
    preds_drp = stacked_model.predict(X_drp_val)
    if drp_cfg['use_log']:
        preds_drp = np.expm1(preds_drp)
    preds_drp = np.maximum(preds_drp, 0)
    submission[target_drp] = preds_drp
    print(f"    DRP predictions: [{preds_drp.min():.1f}, {preds_drp.max():.1f}], mean={preds_drp.mean():.1f}")

# Save
ts = time.strftime("%Y%m%d_%H%M%S")
submission.to_csv(os.path.join(SUBMISSION_DIR, f'submission_v6_{ts}.csv'), index=False)
submission.to_csv(os.path.join(SUBMISSION_DIR, 'submission_v6.csv'), index=False)

print(f"\n  Saved: submissions/submission_v6.csv")
print(f"  Saved: submissions/submission_v6_{ts}.csv")
print(f"\n  Preview:")
print(submission.head(5).to_string())


# ============================================================
# [9] ENSEMBLE WITH PREVIOUS BEST
# ============================================================
print(f"\n{'=' * 70}")
print(f"[9] ENSEMBLE WITH PREVIOUS SUBMISSIONS")
print(f"{'=' * 70}")

ensemble_candidates = []
for vname in ['submission_v1.csv', 'submission_v2.csv', 'submission_v4.csv', 'submission_v5.csv']:
    path = os.path.join(SUBMISSION_DIR, vname)
    if os.path.exists(path):
        ensemble_candidates.append((vname, pd.read_csv(path)))
        print(f"  Found: {vname}")

if ensemble_candidates:
    all_subs = [('V6', submission)] + [(n.replace('submission_', '').replace('.csv', ''), d)
                                        for n, d in ensemble_candidates]
    ensemble = submission.copy()
    for target in TARGETS:
        preds_list = [s[target].values for _, s in all_subs if target in s.columns and s[target].notna().all()]
        if preds_list:
            ensemble[target] = np.maximum(np.mean(preds_list, axis=0), 0)

    ensemble.to_csv(os.path.join(SUBMISSION_DIR, f'submission_ensemble_v6_{ts}.csv'), index=False)
    ensemble.to_csv(os.path.join(SUBMISSION_DIR, 'submission_ensemble_v6.csv'), index=False)
    print(f"  Saved: submissions/submission_ensemble_v6.csv")
else:
    print(f"  No previous submissions found.")


# ============================================================
# [10] FINAL SUMMARY
# ============================================================
print(f"""
{'='*70}
V6 FINAL SUMMARY
{'='*70}

SPATIAL CV:""")
for t in TARGETS:
    print(f"  {t:40s}: R2 = {results[t]:+.4f}")
print(f"  {'─'*55}")
print(f"  {'MEAN':40s}: {mean_r2:+.4f}")

print(f"""
KEY V6 ADDITIONS:
  - Blue band: Water clarity, turbidity (NSMI, clarity index)
  - Red band: Real NDVI, chlorophyll ratio, sediment index
  - Surface temperature: Direct EC correlation (~32% coverage)
  - Coastal aerosol: Atmospheric scattering (L8 only, ~35%)
  - Hybrid Landsat: Full extraction + EY backup = max coverage
  - New thermal x climate interactions for EC prediction
  - New blue x terrain interactions for turbidity

SUBMIT:
  1. submissions/submission_v6.csv
  2. submissions/submission_ensemble_v6.csv (if exists)
  
NEXT STEPS:
  1. Submit to leaderboard and compare vs V4/V5
  2. If improved: Write the experiment log document
  3. Try Optuna hyperparameter tuning on winning config
  4. Try LightGBM/CatBoost as alternatives
{'='*70}
""")

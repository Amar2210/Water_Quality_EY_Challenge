"""
============================================================
EY Water Quality Challenge - V5 Model
============================================================
WHAT'S NEW IN V5:
  - Sentinel-2 features integrated alongside V4 features
  - THREE feature tiers: CORE (~30), EXTENDED (~50), FULL (~80+)
  - THREE XGB configs: Conservative, Moderate, Aggressive
  - Ridge and ElasticNet as linear baselines
  - Target stacking: EC -> DRP (inside each CV fold)
  - Log transform for skewed targets (DRP, optionally EC)
  - Comprehensive comparison across all configs

ARCHITECTURE:
  For each target:
    1. Test all (tier x config) combinations via spatial CV
    2. Pick best combination per target
    3. Try target stacking for DRP
    4. Train final model on full data
    5. Generate submission

OUTPUT:
  submissions/submission_v5.csv
  submissions/submission_v5_YYYYMMDD_HHMMSS.csv
============================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import warnings
import os
import time

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXTRACTED_DIR = os.path.join(BASE_DIR, '..', 'data', 'extracted')
DATA_DIR = os.path.join(BASE_DIR, '..', 'Jupyter Notebook Package')
SUBMISSION_DIR = os.path.join(BASE_DIR, '..', 'submissions')
os.makedirs(SUBMISSION_DIR, exist_ok=True)

TARGETS = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
RANDOM_STATE = 42
N_CLUSTERS = 5

print("=" * 70)
print("EY WATER QUALITY CHALLENGE - V5 MODEL")
print("Strategy: V4 features + Sentinel-2 + 3 tiers x 3 configs + stacking")
print("=" * 70)


# ============================================================
# [1] LOAD DATA
# ============================================================
print("\n" + "=" * 70)
print("[1] LOADING V5 FEATURE-ENGINEERED DATA")
print("=" * 70)

train_df = pd.read_csv(os.path.join(EXTRACTED_DIR, 'v5_train_features.csv'))
val_df = pd.read_csv(os.path.join(EXTRACTED_DIR, 'v5_val_features.csv'))

# Load feature column list
feature_list_path = os.path.join(EXTRACTED_DIR, 'v5_feature_columns.txt')
with open(feature_list_path, 'r') as f:
    all_feature_cols = [line.strip() for line in f if line.strip()]

print(f"  Train: {train_df.shape}")
print(f"  Val:   {val_df.shape}")
print(f"  Total available features: {len(all_feature_cols)}")

# Sentinel-2 coverage
if 'has_sentinel2' in train_df.columns:
    s2_train_pct = 100 * train_df['has_sentinel2'].mean()
    s2_val_pct = 100 * val_df['has_sentinel2'].mean() if 'has_sentinel2' in val_df.columns else 0
    print(f"  S2 coverage: train={s2_train_pct:.1f}%, val={s2_val_pct:.1f}%")


# ============================================================
# [2] DEFINE THREE FEATURE TIERS
# ============================================================
print("\n" + "=" * 70)
print("[2] DEFINING FEATURE TIERS")
print("=" * 70)

# ── TIER 1 — CORE (~30 features) ──
# Best features from V4 research + top S2 indices
TIER1_FEATURES = [
    # Landsat spectral
    'nir', 'green', 'swir16', 'swir22',
    'NDMI', 'MNDWI',

    # Terrain/Geology
    'elevation_m', 'slope_deg',

    # Climate - current
    'ppt', 'tmax', 'tmin', 'pet', 'q', 'vpd',

    # Soil properties
    'soil_phh2o', 'soil_cec', 'soil_clay', 'soil_sand', 'soil_soc',

    # Temporal
    'month_sin', 'month_cos',

    # Top Sentinel-2 indices (physically meaningful for water quality)
    'has_sentinel2',
    's2_ndvi', 's2_ndwi', 's2_mndwi', 's2_ndre',
    's2_chlorophyll_index',
    's2_turbidity_index',
    's2_ndti',
]

# ── TIER 2 — EXTENDED (~50 features) ──
# CORE + all S2 bands + 1-month lags + key interactions
TIER2_FEATURES = TIER1_FEATURES + [
    # S2 raw bands
    's2_blue', 's2_green', 's2_red', 's2_re1', 's2_re2', 's2_re3',
    's2_nir', 's2_nir8a', 's2_swir1', 's2_swir2',

    # 1-month climate lags
    'ppt_lag1', 'tmax_lag1', 'soil_lag1', 'q_lag1',

    # Key V4 derived features
    'aridity_index', 'temp_range', 'ppt_surge',
    'terrain_roughness', 'soil_sand_clay_ratio',
    'ratio_nir_green', 'ndvi_proxy',
    'is_wet_season',

    # Key S2 derived
    's2_bre_ratio', 's2_re_slope', 's2_brightness',
    's2_fai',
]

# ── TIER 3 — FULL (~80+ features) ──
# Everything including cross-sensor, 2-3 month lags, all indices
TIER3_FEATURES = TIER2_FEATURES + [
    # S2 extra derived
    's2_nir_re_diff', 's2_swir_nir_ratio', 's2_vis_nir_ratio',
    's2_green_red_ratio', 's2_swir_diff', 's2_re_nd',
    's2_wv',
    's2_g_r', 's2_re_ratio', 's2_swir_ratio',

    # Cross-sensor features
    's2_ndvi_x_ppt', 's2_turbidity_x_slope', 's2_chlorophyll_x_temp',
    's2_ndwi_x_ppt', 's2_brightness_x_elev', 's2_mndwi_x_soilmoist',
    's2_ndre_x_nitrogen', 's2_fai_x_temp',
    'landsat_s2_water_diff', 'landsat_s2_mndwi_diff',

    # 2-3 month lags
    'ppt_lag2', 'tmax_lag2', 'soil_lag2', 'q_lag2',
    'ppt_lag3', 'tmax_lag3', 'soil_lag3', 'q_lag3',
    'ppt_3m_avg', 'ppt_3m_sum', 'tmax_3m_avg', 'tmin_3m_avg',

    # Extra V4 derived
    'soil_nitrogen', 'soil_buffer_capacity', 'soil_cn_ratio',
    'aspect_deg', 'elev_range_500m', 'log_elevation', 'elev_bin',
    'ratio_swir16_nir', 'ratio_swir22_swir16', 'ratio_green_swir16',
    'ratio_nir_swir22', 'ndsi_swir', 'bsi_proxy', 'ndbi_proxy',
    'water_brightness', 'spectral_std', 'spectral_mean', 'spectral_cv',
    'is_dry_month', 'ppt_change_1m', 'ppt_change_2m', 'drought_3m',
    'evap_rain_ratio', 'evap_deficit_ratio',
    'vpd_change_1m', 'soil_moisture_change', 'runoff_change', 'runoff_coeff',
    'pdsi_trend', 'pdsi', 'pdsi_lag1',
    'aet', 'def', 'srad',

    # V4 interactions
    'slope_x_ppt', 'slope_x_clay',
    'elev_x_tmax', 'elev_x_ppt',
    'moisture_x_clay', 'soilph_x_ppt',
    'mndwi_x_ppt', 'ndmi_x_ppt', 'swir16_x_tmax',

    # Extra temporal
    'month', 'year', 'day_of_year', 'doy_sin', 'doy_cos', 'season',
]


def get_available(feature_list, df):
    """Return features that actually exist in the dataframe, deduplicated."""
    seen = set()
    available = []
    missing = []
    for f in feature_list:
        if f in seen:
            continue
        seen.add(f)
        if f in df.columns:
            available.append(f)
        else:
            missing.append(f)
    return available, missing


tier1_avail, tier1_miss = get_available(TIER1_FEATURES, train_df)
tier2_avail, tier2_miss = get_available(TIER2_FEATURES, train_df)
tier3_avail, tier3_miss = get_available(TIER3_FEATURES, train_df)

print(f"\n  TIER 1 (CORE):     {len(tier1_avail)}/{len(set(TIER1_FEATURES))} available")
if tier1_miss:
    print(f"    Missing: {tier1_miss}")
print(f"  TIER 2 (EXTENDED): {len(tier2_avail)}/{len(set(TIER2_FEATURES))} available")
if tier2_miss:
    print(f"    Missing: {tier2_miss}")
print(f"  TIER 3 (FULL):     {len(tier3_avail)}/{len(set(TIER3_FEATURES))} available")
if tier3_miss:
    print(f"    Missing: {tier3_miss[:15]}{'...' if len(tier3_miss) > 15 else ''}")


# ============================================================
# [3] SPATIAL CLUSTERING
# ============================================================
print("\n" + "=" * 70)
print("[3] SPATIAL CLUSTERING FOR CV")
print("=" * 70)

coords = train_df[['Latitude', 'Longitude']].values
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
train_df['spatial_cluster'] = kmeans.fit_predict(coords)

cluster_counts = train_df['spatial_cluster'].value_counts().sort_index()
print(f"  {N_CLUSTERS} clusters created:")
for cluster_id, count in cluster_counts.items():
    center = coords[train_df['spatial_cluster'] == cluster_id].mean(axis=0)
    print(f"    Cluster {cluster_id}: {count:4d} samples (center: {center[0]:.2f}S, {center[1]:.2f}E)")


# ============================================================
# [4] MODEL CONFIGURATIONS
# ============================================================
print("\n" + "=" * 70)
print("[4] MODEL CONFIGURATIONS")
print("=" * 70)

XGB_CONSERVATIVE = {
    'n_estimators': 500, 'max_depth': 4, 'learning_rate': 0.05,
    'min_child_weight': 20, 'subsample': 0.7, 'colsample_bytree': 0.7,
    'reg_alpha': 1.0, 'reg_lambda': 5.0, 'random_state': RANDOM_STATE,
    'n_jobs': -1, 'early_stopping_rounds': 50,
}
XGB_MODERATE = {
    'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.05,
    'min_child_weight': 10, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'reg_alpha': 0.5, 'reg_lambda': 3.0, 'random_state': RANDOM_STATE,
    'n_jobs': -1, 'early_stopping_rounds': 50,
}
XGB_AGGRESSIVE = {
    'n_estimators': 800, 'max_depth': 6, 'learning_rate': 0.03,
    'min_child_weight': 5, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'reg_alpha': 0.1, 'reg_lambda': 1.0, 'random_state': RANDOM_STATE,
    'n_jobs': -1, 'early_stopping_rounds': 50,
}

print("  XGB_CONSERVATIVE: depth=4, min_child=20, alpha=1.0, lambda=5.0")
print("  XGB_MODERATE:     depth=5, min_child=10, alpha=0.5, lambda=3.0")
print("  XGB_AGGRESSIVE:   depth=6, min_child=5,  alpha=0.1, lambda=1.0")
print("  Ridge:            alpha=1.0 (with imputation + scaling)")
print("  ElasticNet:       alpha=1.0, l1_ratio=0.5 (with imputation + scaling)")


# ============================================================
# [5] SPATIAL CV EVALUATION FUNCTIONS
# ============================================================

def evaluate_xgb_cv(X, y, groups, feature_names, use_log=False, xgb_params=None):
    """Run spatial GroupKFold CV with XGBoost."""
    if xgb_params is None:
        xgb_params = XGB_CONSERVATIVE

    gkf = GroupKFold(n_splits=N_CLUSTERS)
    fold_scores = []
    fold_importances = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        if use_log:
            y_tr_model = np.log1p(y_tr)
        else:
            y_tr_model = y_tr

        params = xgb_params.copy()
        early_stop = params.pop('early_stopping_rounds', 50)

        model = xgb.XGBRegressor(**params)
        model.fit(
            X_tr, y_tr_model,
            eval_set=[(X_te, np.log1p(y_te) if use_log else y_te)],
            verbose=False,
        )

        preds = model.predict(X_te)
        if use_log:
            preds = np.expm1(preds)
        preds = np.maximum(preds, 0)

        r2 = r2_score(y_te, preds)
        fold_scores.append(r2)
        fold_importances.append(model.feature_importances_)

    mean_r2 = np.mean(fold_scores)
    mean_importances = np.mean(fold_importances, axis=0)
    return mean_r2, fold_scores, dict(zip(feature_names, mean_importances))


def evaluate_linear_cv(X, y, groups, feature_names, use_log=False, model_type='ridge'):
    """Run spatial GroupKFold CV with Ridge or ElasticNet (with imputation + scaling)."""
    gkf = GroupKFold(n_splits=N_CLUSTERS)
    fold_scores = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_te = X.iloc[train_idx].values, X.iloc[test_idx].values
        y_tr, y_te = y.iloc[train_idx].values, y.iloc[test_idx].values

        if use_log:
            y_tr_model = np.log1p(y_tr)
        else:
            y_tr_model = y_tr

        # Build pipeline: impute NaN -> scale -> linear model
        if model_type == 'ridge':
            estimator = Ridge(alpha=1.0, random_state=RANDOM_STATE)
        else:
            estimator = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=5000)

        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', estimator),
        ])

        pipe.fit(X_tr, y_tr_model)
        preds = pipe.predict(X_te)

        if use_log:
            preds = np.expm1(preds)
        preds = np.maximum(preds, 0)

        r2 = r2_score(y_te, preds)
        fold_scores.append(r2)

    mean_r2 = np.mean(fold_scores)
    # Linear models don't have tree feature importances; use absolute coefficients
    # (we train one more time on full data for importances — approximate)
    pipe_full = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=1.0) if model_type == 'ridge' else
         ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000)),
    ])
    X_full_imp = SimpleImputer(strategy='median').fit_transform(X.values)
    pipe_full.fit(X.values, np.log1p(y.values) if use_log else y.values)
    coefs = np.abs(pipe_full.named_steps['model'].coef_)
    importances = dict(zip(feature_names, coefs / (coefs.sum() + 1e-8)))

    return mean_r2, fold_scores, importances


# ============================================================
# [6] EVALUATE ALL TIERS x ALL CONFIGS PER TARGET
# ============================================================
print("\n" + "=" * 70)
print("[6] SPATIAL CV EVALUATION: ALL TIERS x ALL CONFIGS")
print("=" * 70)

results = {}        # target -> best R2
best_configs = {}   # target -> best config info
all_results = {}    # target -> {config_name: info}

groups = train_df['spatial_cluster']

TIER_MAP = {
    'T1_CORE': tier1_avail,
    'T2_EXTENDED': tier2_avail,
    'T3_FULL': tier3_avail,
}

XGB_MAP = {
    'conservative': XGB_CONSERVATIVE,
    'moderate': XGB_MODERATE,
    'aggressive': XGB_AGGRESSIVE,
}

for target in TARGETS:
    print(f"\n{'=' * 70}")
    print(f"  TARGET: {target}")
    print(f"{'=' * 70}")

    y = train_df[target]
    skewness = y.skew()
    try_log = skewness > 1.0
    print(f"  Skewness: {skewness:.2f} -> {'will try log transform' if try_log else 'no log needed'}")

    configs = {}

    # ── XGBoost configs ──
    for tier_name, feat_list in TIER_MAP.items():
        for param_name, params in XGB_MAP.items():
            X = train_df[feat_list]

            # Without log
            config_name = f"{tier_name}_{param_name}"
            r2, folds, imps = evaluate_xgb_cv(
                X, y, groups, feat_list,
                use_log=False, xgb_params=params
            )
            configs[config_name] = {
                'r2': r2, 'folds': folds, 'features': feat_list,
                'params': params, 'use_log': False, 'importances': imps,
                'model_type': 'xgb',
            }

            # With log (skewed targets only)
            if try_log:
                config_name_log = f"{tier_name}_{param_name}_log"
                r2_log, folds_log, imps_log = evaluate_xgb_cv(
                    X, y, groups, feat_list,
                    use_log=True, xgb_params=params
                )
                configs[config_name_log] = {
                    'r2': r2_log, 'folds': folds_log, 'features': feat_list,
                    'params': params, 'use_log': True, 'importances': imps_log,
                    'model_type': 'xgb',
                }

    # ── Linear models (Ridge, ElasticNet) — only on CORE tier to keep it fast ──
    for model_type in ['ridge', 'elasticnet']:
        X = train_df[tier1_avail]
        config_name = f"T1_CORE_{model_type}"
        r2, folds, imps = evaluate_linear_cv(
            X, y, groups, tier1_avail,
            use_log=False, model_type=model_type
        )
        configs[config_name] = {
            'r2': r2, 'folds': folds, 'features': tier1_avail,
            'params': {'model_type': model_type}, 'use_log': False,
            'importances': imps, 'model_type': model_type,
        }

        if try_log:
            config_name_log = f"T1_CORE_{model_type}_log"
            r2_log, folds_log, imps_log = evaluate_linear_cv(
                X, y, groups, tier1_avail,
                use_log=True, model_type=model_type
            )
            configs[config_name_log] = {
                'r2': r2_log, 'folds': folds_log, 'features': tier1_avail,
                'params': {'model_type': model_type}, 'use_log': True,
                'importances': imps_log, 'model_type': model_type,
            }

    # ── Print all results sorted ──
    print(f"\n  {'Config':<40s} {'R2':>8s}  {'Fold scores'}")
    print(f"  {'─'*40} {'─'*8}  {'─'*45}")
    sorted_configs = sorted(configs.items(), key=lambda x: x[1]['r2'], reverse=True)
    for i, (config_name, info) in enumerate(sorted_configs):
        fold_str = ', '.join([f"{f:+.3f}" for f in info['folds']])
        marker = " <-- BEST" if i == 0 else ""
        print(f"  {config_name:<40s} {info['r2']:+.4f}  [{fold_str}]{marker}")

    # ── Select best ──
    best_name, best_info = sorted_configs[0]
    best_configs[target] = best_info
    best_configs[target]['config_name'] = best_name
    results[target] = best_info['r2']
    all_results[target] = configs

    # ── Feature importances for best config ──
    print(f"\n  Best config: {best_name}")
    print(f"  Top 10 features:")
    sorted_imps = sorted(best_info['importances'].items(), key=lambda x: x[1], reverse=True)
    for rank, (feat, imp) in enumerate(sorted_imps[:10], 1):
        bar = '#' * int(imp * 80)
        s2_flag = " [S2]" if feat.startswith('s2_') or feat == 'has_sentinel2' else ""
        print(f"    {rank:2d}. {feat:30s} {imp:.4f} {bar}{s2_flag}")


# ============================================================
# [7] TARGET STACKING: EC -> DRP
# ============================================================
print(f"\n{'=' * 70}")
print(f"[7] TARGET STACKING: EC predictions -> DRP feature")
print(f"    Research: conductivity was #1 predictor of phosphate (r=0.63)")
print(f"{'=' * 70}")

target_drp = 'Dissolved Reactive Phosphorus'
target_ec = 'Electrical Conductance'
y_drp = train_df[target_drp]
y_ec = train_df[target_ec]

ec_config = best_configs[target_ec]
drp_config = best_configs[target_drp]

ec_features = ec_config['features']
drp_features = drp_config['features']
drp_use_log = drp_config['use_log']
ec_params = ec_config['params'] if ec_config['model_type'] == 'xgb' else XGB_MODERATE
drp_params = drp_config['params'] if drp_config['model_type'] == 'xgb' else XGB_MODERATE

# Only do stacking if both best models are XGBoost (stacking makes less sense for linear)
if ec_config['model_type'] == 'xgb' and drp_config['model_type'] == 'xgb':
    gkf = GroupKFold(n_splits=N_CLUSTERS)
    stacked_fold_scores = []

    print(f"\n  EC model features:  {len(ec_features)}")
    print(f"  DRP model features: {len(drp_features)} + 1 (predicted_EC)")
    print(f"  DRP log transform:  {drp_use_log}")

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(train_df, y_drp, groups)):
        # Step 1: Train EC model on this fold's train data
        X_ec_tr = train_df.iloc[train_idx][ec_features]
        X_ec_te = train_df.iloc[test_idx][ec_features]
        y_ec_tr = y_ec.iloc[train_idx]

        params_ec = ec_params.copy()
        params_ec.pop('early_stopping_rounds', None)
        ec_model = xgb.XGBRegressor(**params_ec)
        ec_model.fit(X_ec_tr, y_ec_tr, verbose=False)

        # Predict EC for both folds (train predictions refit, test predictions OOF)
        ec_pred_train = ec_model.predict(X_ec_tr)
        ec_pred_test = ec_model.predict(X_ec_te)

        # Step 2: Add predicted EC as feature for DRP
        X_drp_tr = train_df.iloc[train_idx][drp_features].copy()
        X_drp_te = train_df.iloc[test_idx][drp_features].copy()
        X_drp_tr['predicted_EC'] = ec_pred_train
        X_drp_te['predicted_EC'] = ec_pred_test

        y_drp_tr = y_drp.iloc[train_idx]
        y_drp_te = y_drp.iloc[test_idx]

        y_drp_tr_model = np.log1p(y_drp_tr) if drp_use_log else y_drp_tr

        params_drp = drp_params.copy()
        params_drp.pop('early_stopping_rounds', None)
        drp_model = xgb.XGBRegressor(**params_drp)
        drp_model.fit(X_drp_tr, y_drp_tr_model, verbose=False)

        preds = drp_model.predict(X_drp_te)
        if drp_use_log:
            preds = np.expm1(preds)
        preds = np.maximum(preds, 0)

        r2 = r2_score(y_drp_te, preds)
        stacked_fold_scores.append(r2)

    stacked_r2 = np.mean(stacked_fold_scores)
    original_drp_r2 = results[target_drp]

    print(f"\n  DRP without stacking: R2 = {original_drp_r2:+.4f}")
    print(f"  DRP with EC stacking: R2 = {stacked_r2:+.4f}")
    print(f"  Improvement:          {stacked_r2 - original_drp_r2:+.4f}")

    use_stacking = stacked_r2 > original_drp_r2
    if use_stacking:
        print(f"  -> STACKING HELPS! Will use for final model.")
        results[target_drp] = stacked_r2
    else:
        print(f"  -> Stacking did not help. Using original DRP model.")
else:
    use_stacking = False
    print(f"\n  Skipping stacking (best DRP or EC model is not XGBoost).")
    print(f"  Best EC model:  {ec_config.get('config_name', 'unknown')}")
    print(f"  Best DRP model: {drp_config.get('config_name', 'unknown')}")


# ============================================================
# [8] SUMMARY OF SPATIAL CV
# ============================================================
print(f"\n{'=' * 70}")
print(f"[8] V5 SPATIAL CV RESULTS SUMMARY")
print(f"{'=' * 70}")

for target in TARGETS:
    r2 = results[target]
    cfg = best_configs[target]
    feat_count = len(cfg['features'])
    log_flag = " (log)" if cfg['use_log'] else ""
    model_flag = f" [{cfg['model_type']}]"
    stack_flag = " + EC stacking" if (target == target_drp and use_stacking) else ""
    config_name = cfg.get('config_name', 'unknown')
    status = "+" if r2 > 0 else "-"
    print(f"  {status} {target:40s}: R2 = {r2:+.4f}  ({config_name}{log_flag}{stack_flag})")

mean_r2 = np.mean(list(results.values()))
print(f"  {'=' * 65}")
print(f"    {'MEAN R2':40s}: {mean_r2:+.4f}")

# Sentinel-2 feature analysis
print(f"\n  Sentinel-2 feature analysis (in top-20 of best configs):")
for target in TARGETS:
    cfg = best_configs[target]
    sorted_imps = sorted(cfg['importances'].items(), key=lambda x: x[1], reverse=True)
    top20 = [f for f, _ in sorted_imps[:20]]
    s2_in_top20 = [f for f in top20 if f.startswith('s2_') or f == 'has_sentinel2']
    print(f"  {target}:")
    if s2_in_top20:
        for f in s2_in_top20:
            print(f"    - {f}")
    else:
        print(f"    (none in top 20)")


# Version comparison
print(f"\n  Version comparison (Spatial CV Mean R2):")
print(f"    V0.1 (benchmark):               ~-0.40")
print(f"    V1   (all features):             ~-0.40")
print(f"    V2   (invariant only):           ~-0.42")
print(f"    V4   (research-backed):          ~-0.35")
print(f"    V5   (V4 + Sentinel-2):          {mean_r2:+.4f}")


# ============================================================
# [9] TRAIN FINAL MODELS + GENERATE SUBMISSION
# ============================================================
print(f"\n{'=' * 70}")
print(f"[9] TRAINING FINAL MODELS ON ALL DATA + SUBMISSION")
print(f"{'=' * 70}")

submission = pd.read_csv(os.path.join(DATA_DIR, 'submission_template.csv'))
final_models = {}

for target in TARGETS:
    cfg = best_configs[target]
    features = cfg['features']
    use_log = cfg['use_log']
    model_type = cfg['model_type']
    config_name = cfg.get('config_name', '')

    print(f"\n  Training final model for: {target}")
    print(f"    Config: {config_name}, features: {len(features)}, log: {use_log}, type: {model_type}")

    X_train_full = train_df[features]
    y_train_full = train_df[target]

    if use_log:
        y_train_model = np.log1p(y_train_full)
    else:
        y_train_model = y_train_full

    if model_type == 'xgb':
        params = cfg['params'].copy()
        params.pop('early_stopping_rounds', None)
        model = xgb.XGBRegressor(**params)
        model.fit(X_train_full, y_train_model, verbose=False)
    elif model_type in ('ridge', 'elasticnet'):
        if model_type == 'ridge':
            estimator = Ridge(alpha=1.0, random_state=RANDOM_STATE)
        else:
            estimator = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=5000)
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', estimator),
        ])
        model.fit(X_train_full.values, y_train_model.values)

    final_models[target] = {
        'model': model,
        'features': features,
        'use_log': use_log,
        'model_type': model_type,
    }

    # Predict for validation (non-stacked targets)
    if target != target_drp or not use_stacking:
        X_val = val_df[features]
        if model_type == 'xgb':
            preds = model.predict(X_val)
        else:
            preds = model.predict(X_val.values)
        if use_log:
            preds = np.expm1(preds)
        preds = np.maximum(preds, 0)
        submission[target] = preds
        print(f"    Predictions: range [{preds.min():.2f}, {preds.max():.2f}], mean {preds.mean():.2f}")


# Handle DRP with stacking
if use_stacking:
    print(f"\n  Training DRP with target stacking...")

    # EC predictions for train + val
    ec_info = final_models[target_ec]
    ec_model = ec_info['model']
    ec_features = ec_info['features']

    if ec_info['model_type'] == 'xgb':
        ec_preds_train = ec_model.predict(train_df[ec_features])
        ec_preds_val = ec_model.predict(val_df[ec_features])
    else:
        ec_preds_train = ec_model.predict(train_df[ec_features].values)
        ec_preds_val = ec_model.predict(val_df[ec_features].values)

    # DRP model with stacked EC
    drp_feats = drp_config['features']
    drp_use_log_final = drp_config['use_log']

    X_drp_train = train_df[drp_feats].copy()
    X_drp_train['predicted_EC'] = ec_preds_train
    y_drp_train = train_df[target_drp]
    if drp_use_log_final:
        y_drp_train_model = np.log1p(y_drp_train)
    else:
        y_drp_train_model = y_drp_train

    params_drp_final = drp_params.copy()
    params_drp_final.pop('early_stopping_rounds', None)
    drp_stacked_model = xgb.XGBRegressor(**params_drp_final)
    drp_stacked_model.fit(X_drp_train, y_drp_train_model, verbose=False)

    X_drp_val = val_df[drp_feats].copy()
    X_drp_val['predicted_EC'] = ec_preds_val
    preds_drp = drp_stacked_model.predict(X_drp_val)
    if drp_use_log_final:
        preds_drp = np.expm1(preds_drp)
    preds_drp = np.maximum(preds_drp, 0)
    submission[target_drp] = preds_drp
    print(f"    DRP predictions (stacked): range [{preds_drp.min():.2f}, {preds_drp.max():.2f}], mean {preds_drp.mean():.2f}")

# Save submission
ts = time.strftime("%Y%m%d_%H%M%S")
submission.to_csv(os.path.join(SUBMISSION_DIR, f'submission_v5_{ts}.csv'), index=False)
submission.to_csv(os.path.join(SUBMISSION_DIR, 'submission_v5.csv'), index=False)

print(f"\n  Saved: submissions/submission_v5.csv")
print(f"  Saved: submissions/submission_v5_{ts}.csv")
print(f"\n  Preview:")
print(submission.head(5).to_string())


# ============================================================
# [10] ENSEMBLE WITH PREVIOUS BEST (OPTIONAL)
# ============================================================
print(f"\n{'=' * 70}")
print(f"[10] ENSEMBLE WITH PREVIOUS SUBMISSIONS")
print(f"{'=' * 70}")

# Check which previous submissions exist
ensemble_candidates = []
for v_name in ['submission_v1.csv', 'submission_v2.csv', 'submission_v4.csv']:
    path = os.path.join(SUBMISSION_DIR, v_name)
    if os.path.exists(path):
        ensemble_candidates.append((v_name, pd.read_csv(path)))
        print(f"  Found: {v_name}")

if ensemble_candidates:
    # Simple average ensemble of V5 + all previous versions
    v5_sub = submission.copy()
    all_subs = [(f'V5', v5_sub)] + [(name.replace('submission_', '').replace('.csv', ''), df)
                                      for name, df in ensemble_candidates]

    print(f"\n  Building ensemble from: {[name for name, _ in all_subs]}")

    ensemble_sub = submission.copy()
    for target in TARGETS:
        target_preds = []
        for name, sub_df in all_subs:
            if target in sub_df.columns and sub_df[target].notna().all():
                target_preds.append(sub_df[target].values)

        if target_preds:
            ensemble_sub[target] = np.mean(target_preds, axis=0)
            ensemble_sub[target] = np.maximum(ensemble_sub[target], 0)

    ensemble_sub.to_csv(os.path.join(SUBMISSION_DIR, f'submission_ensemble_v5_{ts}.csv'), index=False)
    ensemble_sub.to_csv(os.path.join(SUBMISSION_DIR, 'submission_ensemble_v5.csv'), index=False)
    print(f"  Saved: submissions/submission_ensemble_v5.csv")
    print(f"\n  Ensemble preview:")
    print(ensemble_sub.head(5).to_string())
else:
    print(f"  No previous submissions found for ensemble.")


# ============================================================
# [11] FINAL ANALYSIS
# ============================================================
print(f"""
{'=' * 70}
V5 FINAL ANALYSIS
{'=' * 70}

SPATIAL CV RESULTS:""")
for target in TARGETS:
    r2 = results[target]
    cfg = best_configs[target]
    print(f"  {target:40s}: R2 = {r2:+.4f}  ({cfg.get('config_name', '')})")
print(f"  {'─' * 60}")
print(f"  {'MEAN R2':40s}: {mean_r2:+.4f}")

print(f"""
NOTES:
  - Spatial CV R2 may be negative (expected for this challenge)
  - Leaderboard scores have historically been MUCH better than spatial CV
  - V4 had negative CV but positive leaderboard scores
  - Submit both submission_v5.csv and submission_ensemble_v5.csv
  - The ensemble often performs best on the leaderboard

WHAT V5 ADDS OVER V4:
  - Sentinel-2 bands at 10m resolution (vs 30m Landsat)
  - Red edge bands for chlorophyll/eutrophication detection
  - Cross-sensor interactions (S2 x climate, S2 x terrain)
  - 3 XGB config tiers for better config search
  - Linear model baselines (Ridge, ElasticNet)
  - More rigorous per-target config selection

NEXT STEPS:
  1. Submit submission_v5.csv to leaderboard
  2. Submit submission_ensemble_v5.csv to leaderboard
  3. Compare leaderboard score vs V4
  4. If V5 > V4: Sentinel-2 features are helping!
  5. Try: Optuna hyperparameter search on winning tier/config
  6. Try: LightGBM / CatBoost as XGBoost alternatives
{'=' * 70}
""")

"""
============================================================
EY Water Quality Challenge — V7: EC-Focused Model
============================================================
Research-backed approach focusing on Electrical Conductance.

Key V7 innovations:
1. Cool dry season flag (May-Oct) — optimal remote sensing window
2. El Niño drought flag (Oct 2014+) — concentrated ions = stronger signal
3. Season×spectral and drought×spectral interactions
4. SWIR/NIR prioritization (EC-specific spectral science)
5. Surface temperature as key input (EC is temp-dependent)
6. Feature discipline: ~40 focused features, not 173
7. LightGBM added alongside XGBoost, Ridge, ElasticNet
8. RMSE + MAE reported alongside R²
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
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb

try:
    import lightgbm as lgb
    HAS_LGBM = True
    print("  LightGBM available ✓")
except ImportError:
    HAS_LGBM = False
    print("  LightGBM not installed — will skip LightGBM configs")
    print("  Install with: pip install lightgbm")

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================
# PATHS
# ============================================================
BASE_DIR = r"C:\Users\amarbir\Self_Learning\EY_Challenge"
DATA_DIR = os.path.join(BASE_DIR, "data", "extracted")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
SUBMISSION_DIR = os.path.join(BASE_DIR, "submissions")
LOG_DIR = os.path.join(BASE_DIR, "experiment_logs")

TARGET = "Electrical Conductance"
TARGET_SHORT = "EC"

ts = datetime.now().strftime("%Y%m%d_%H%M%S")

print(f"""
{'='*70}
EY WATER QUALITY — V7: EC-FOCUSED MODEL
{'='*70}
Target: {TARGET}
Date: {ts}
{'='*70}
""")

# ============================================================
# [1] LOAD DATA
# ============================================================
print(f"[1] Loading V6 feature data...")

train_df = pd.read_csv(os.path.join(DATA_DIR, "v6_train_features.csv"))
val_df = pd.read_csv(os.path.join(DATA_DIR, "v6_val_features.csv"))

print(f"  Train: {len(train_df)} rows")
print(f"  Val:   {len(val_df)} rows")

# Parse dates
train_df['Sample Date'] = pd.to_datetime(train_df['Sample Date'], format='%d-%m-%Y', dayfirst=True, errors='coerce')
val_df['Sample Date'] = pd.to_datetime(val_df['Sample Date'], format='%d-%m-%Y', dayfirst=True, errors='coerce')

# If date parsing failed, try alternative format
if train_df['Sample Date'].isna().sum() > len(train_df) * 0.5:
    train_df['Sample Date'] = pd.to_datetime(train_df['Sample Date'], errors='coerce')
    val_df['Sample Date'] = pd.to_datetime(val_df['Sample Date'], errors='coerce')

print(f"  Date range (train): {train_df['Sample Date'].min()} to {train_df['Sample Date'].max()}")
print(f"  EC stats: mean={train_df[TARGET].mean():.1f}, median={train_df[TARGET].median():.1f}, "
      f"std={train_df[TARGET].std():.1f}, skew={train_df[TARGET].skew():.2f}")

# ============================================================
# [2] V7 FEATURE ENGINEERING — Research-Backed
# ============================================================
print(f"\n{'='*70}")
print(f"[2] V7 FEATURE ENGINEERING")
print(f"{'='*70}")

def add_v7_features(df):
    """Add V7-specific research-backed features for EC prediction."""
    
    # ---- COOL DRY SEASON FLAG (May-October) ----
    # Research: optimal remote sensing window for Southern Africa
    # EC models perform better during dry seasons (higher, more stable ionic concentrations)
    df['is_cool_dry_season'] = df['month'].isin([5, 6, 7, 8, 9, 10]).astype(int)
    
    # ---- EL NIÑO DROUGHT FLAG (October 2014 onwards) ----
    # Research: declining water levels → concentrated dissolved ions → stronger spectral signal
    # This provides "high-end" extremes of EC and TA
    if df['Sample Date'].notna().any():
        drought_start = pd.Timestamp('2014-10-01')
        df['is_drought_period'] = (df['Sample Date'] >= drought_start).astype(int)
    else:
        # Fallback: use year column
        df['is_drought_period'] = ((df['year'] >= 2015) | 
                                    ((df['year'] == 2014) & (df['month'] >= 10))).astype(int)
    
    # ---- SEASON × SPECTRAL INTERACTIONS ----
    # Rationale: During dry season, SWIR/NIR signal from minerals is amplified
    # because ionic concentrations are higher and turbidity interference is lower
    df['cool_dry_x_swir16'] = df['is_cool_dry_season'] * df['unified_swir16'].fillna(0)
    df['cool_dry_x_nir'] = df['is_cool_dry_season'] * df['unified_nir'].fillna(0)
    df['cool_dry_x_ndmi'] = df['is_cool_dry_season'] * df['unified_NDMI'].fillna(0)
    df['cool_dry_x_swir_ratio'] = df['is_cool_dry_season'] * df['ratio_swir16_nir'].fillna(0)
    
    # ---- DROUGHT × SPECTRAL INTERACTIONS ----
    # Rationale: Drought concentrates ions → stronger SWIR response
    df['drought_x_swir16'] = df['is_drought_period'] * df['unified_swir16'].fillna(0)
    df['drought_x_swir22'] = df['is_drought_period'] * df['unified_swir22'].fillna(0)
    df['drought_x_ndmi'] = df['is_drought_period'] * df['unified_NDMI'].fillna(0)
    
    # ---- DROUGHT × ARIDITY ----
    # Double concentration effect: drought period + already arid location
    df['drought_x_aridity'] = df['is_drought_period'] * df['aridity_index'].fillna(0)
    
    # ---- TEMPERATURE × EC PROXIES ----
    # Research: EC is temperature-dependent. Higher temp → higher EC reading
    # Surface temp × SWIR captures thermal-mineral coupling
    df['st_celsius_safe'] = df['ls_st_celsius'].fillna(df['tmax'].fillna(25))  # fallback to air temp
    df['st_x_vpd'] = df['st_celsius_safe'] * df['vpd'].fillna(0)
    df['st_x_aridity'] = df['st_celsius_safe'] * df['aridity_index'].fillna(0)
    
    # ---- SWIR BAND AGGREGATES ----
    # Research: SWIR region captures mineral-induced spectral changes
    swir16 = df['unified_swir16'].fillna(0)
    swir22 = df['unified_swir22'].fillna(0)
    df['swir_mean'] = (swir16 + swir22) / 2
    df['swir_diff'] = swir16 - swir22  # mineral vs moisture discrimination
    
    # ---- SEASONAL CLIMATE WINDOW ----
    # 1-month window = immediate conditions
    # 3-month window = seasonal context (already have ppt_3m_sum, tmax_3m_avg)
    # Ratio of current month vs 3-month average → anomaly detection
    df['ppt_anomaly'] = np.where(
        df['ppt_3m_avg'].fillna(0) > 0,
        df['ppt'].fillna(0) / df['ppt_3m_avg'].fillna(1).clip(lower=0.1),
        1.0
    )
    df['temp_anomaly'] = df['tmax'].fillna(0) - df['tmax_3m_avg'].fillna(df['tmax'].fillna(0))
    
    return df


# Apply to both train and val
for label, df in [("Train", train_df), ("Val", val_df)]:
    add_v7_features(df)
    n_dry = df['is_cool_dry_season'].sum()
    n_drought = df['is_drought_period'].sum()
    print(f"  {label}: cool_dry_season={n_dry}/{len(df)} ({100*n_dry/len(df):.1f}%), "
          f"drought_period={n_drought}/{len(df)} ({100*n_drought/len(df):.1f}%)")


# ============================================================
# [3] DEFINE EC-FOCUSED FEATURE SETS
# ============================================================
print(f"\n{'='*70}")
print(f"[3] DEFINING EC-FOCUSED FEATURE SETS")
print(f"{'='*70}")

# TIER 1: Core EC features (~25 features) — physically justified, minimal
EC_CORE = [
    # Spectral — SWIR/NIR priority (research: most sensitive to dissolved minerals)
    'unified_swir16', 'unified_swir22', 'unified_nir', 'unified_green',
    'ratio_swir16_nir', 'ratio_swir22_swir16', 'ratio_nir_swir22',
    'unified_NDMI', 'unified_MNDWI',
    'swir_mean', 'swir_diff',
    
    # Surface temperature (research: EC is temperature-dependent)
    'ls_st_celsius', 'has_surface_temp',
    
    # Terrain/Soil (top performers V0-V6)
    'elevation_m', 'slope_deg', 'terrain_roughness', 'elev_range_500m',
    'soil_soc', 'soil_sand', 'soil_clay', 'soil_cec', 'soil_phh2o',
    'soil_buffer_capacity', 'soil_sand_clay_ratio',
    
    # Climate (immediate + seasonal)
    'ppt', 'tmax', 'vpd', 'aridity_index',
    'ppt_3m_sum', 'tmax_3m_avg',
    
    # Temporal
    'month_sin', 'month_cos',
]

# TIER 2: EC Extended (~40 features) — adds seasonal/drought signals
EC_EXTENDED = EC_CORE + [
    # V7 seasonal/drought flags
    'is_cool_dry_season', 'is_drought_period',
    
    # V7 season × spectral interactions
    'cool_dry_x_swir16', 'cool_dry_x_nir', 'cool_dry_x_ndmi',
    
    # V7 drought × spectral interactions
    'drought_x_swir16', 'drought_x_ndmi', 'drought_x_aridity',
    
    # Temperature interactions
    'st_x_vpd', 'st_x_aridity',
    'swir16_x_tmax',
    
    # Terrain × climate (erosion, weathering)
    'slope_x_clay', 'elev_x_ppt',
    
    # Climate dynamics
    'ppt_anomaly', 'temp_anomaly',
    'soil',  # soil moisture
    'aet', 'def',
    
    # Additional soil
    'soil_nitrogen',
]

# TIER 3: EC Full (~55 features) — everything that might help, including more spectral
EC_FULL = EC_EXTENDED + [
    # More spectral indices
    'ls_NSMI', 'unified_NDVI',
    'ratio_green_swir16',
    
    # More climate lags
    'ppt_lag1', 'tmax_lag1', 'vpd_lag1',
    'ppt_lag2', 'tmax_lag2',
    
    # More terrain
    'log_elevation', 'aspect_deg',
    
    # More interactions from V6
    'st_x_swir16', 'st_x_elev', 'st_x_clay',
    'moisture_x_clay', 'soilph_x_ppt',
    
    # Drought additional
    'drought_x_swir22',
    'cool_dry_x_swir_ratio',
    
    # Spectral variability
    'spectral_std', 'spectral_mean',
]

FEATURE_TIERS = {
    'EC_CORE': EC_CORE,
    'EC_EXTENDED': EC_EXTENDED,
    'EC_FULL': EC_FULL,
}

# Filter to only features that exist in the data
for tier_name, features in FEATURE_TIERS.items():
    available = [f for f in features if f in train_df.columns]
    missing = [f for f in features if f not in train_df.columns]
    FEATURE_TIERS[tier_name] = available
    print(f"  {tier_name}: {len(available)} available, {len(missing)} missing")
    if missing:
        print(f"    Missing: {missing[:10]}{'...' if len(missing) > 10 else ''}")


# ============================================================
# [4] SPATIAL CROSS-VALIDATION SETUP
# ============================================================
print(f"\n{'='*70}")
print(f"[4] SPATIAL CV SETUP")
print(f"{'='*70}")

N_CLUSTERS = 8
N_FOLDS = 8

coords = train_df[['Latitude', 'Longitude']].values
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=20)
train_df['spatial_cluster'] = kmeans.fit_predict(coords)

# Show cluster distribution
for c in range(N_CLUSTERS):
    n = (train_df['spatial_cluster'] == c).sum()
    print(f"  Cluster {c}: {n} samples ({100*n/len(train_df):.1f}%)")


# ============================================================
# [5] MODEL CONFIGURATIONS
# ============================================================
print(f"\n{'='*70}")
print(f"[5] MODEL CONFIGURATIONS")
print(f"{'='*70}")

MODEL_CONFIGS = {}

# --- XGBoost configs ---
MODEL_CONFIGS['xgb_aggressive'] = {
    'type': 'xgboost',
    'params': {
        'n_estimators': 500, 'max_depth': 4, 'learning_rate': 0.05,
        'subsample': 0.7, 'colsample_bytree': 0.7,
        'min_child_weight': 10, 'reg_alpha': 1.0, 'reg_lambda': 5.0,
        'random_state': 42, 'verbosity': 0,
    }
}

MODEL_CONFIGS['xgb_moderate'] = {
    'type': 'xgboost',
    'params': {
        'n_estimators': 300, 'max_depth': 3, 'learning_rate': 0.03,
        'subsample': 0.6, 'colsample_bytree': 0.6,
        'min_child_weight': 20, 'reg_alpha': 2.0, 'reg_lambda': 10.0,
        'random_state': 42, 'verbosity': 0,
    }
}

MODEL_CONFIGS['xgb_conservative'] = {
    'type': 'xgboost',
    'params': {
        'n_estimators': 200, 'max_depth': 2, 'learning_rate': 0.02,
        'subsample': 0.5, 'colsample_bytree': 0.5,
        'min_child_weight': 30, 'reg_alpha': 5.0, 'reg_lambda': 20.0,
        'random_state': 42, 'verbosity': 0,
    }
}

# --- LightGBM configs ---
if HAS_LGBM:
    MODEL_CONFIGS['lgbm_default'] = {
        'type': 'lightgbm',
        'params': {
            'n_estimators': 500, 'max_depth': 4, 'learning_rate': 0.05,
            'subsample': 0.7, 'colsample_bytree': 0.7,
            'min_child_samples': 20, 'reg_alpha': 1.0, 'reg_lambda': 5.0,
            'num_leaves': 15, 'random_state': 42, 'verbosity': -1,
        }
    }
    MODEL_CONFIGS['lgbm_conservative'] = {
        'type': 'lightgbm',
        'params': {
            'n_estimators': 300, 'max_depth': 3, 'learning_rate': 0.02,
            'subsample': 0.5, 'colsample_bytree': 0.5,
            'min_child_samples': 40, 'reg_alpha': 3.0, 'reg_lambda': 15.0,
            'num_leaves': 8, 'random_state': 42, 'verbosity': -1,
        }
    }

# --- Linear models (only on CORE features — they don't handle 50+ well) ---
MODEL_CONFIGS['ridge'] = {
    'type': 'ridge',
    'params': {'alpha': 10.0},
    'tiers': ['EC_CORE'],  # restrict to core features only
}

MODEL_CONFIGS['elasticnet'] = {
    'type': 'elasticnet',
    'params': {'alpha': 1.0, 'l1_ratio': 0.5, 'max_iter': 5000},
    'tiers': ['EC_CORE'],
}

print(f"  Total configs: {len(MODEL_CONFIGS)}")
for name, cfg in MODEL_CONFIGS.items():
    tier_note = f" (tiers: {cfg.get('tiers', 'all')})" if 'tiers' in cfg else ""
    print(f"    {name}: {cfg['type']}{tier_note}")


# ============================================================
# [6] TRAINING LOOP — EC ONLY
# ============================================================
print(f"\n{'='*70}")
print(f"[6] TRAINING — EC ONLY")
print(f"{'='*70}")

y_train_raw = train_df[TARGET].values
y_train_log = np.log1p(y_train_raw)  # log(1 + EC) for log-transform option

groups = train_df['spatial_cluster'].values
gkf = GroupKFold(n_splits=N_FOLDS)

results = {}

for tier_name, feature_cols in FEATURE_TIERS.items():
    for config_name, config in MODEL_CONFIGS.items():
        
        # Check if this config is restricted to certain tiers
        allowed_tiers = config.get('tiers', list(FEATURE_TIERS.keys()))
        if tier_name not in allowed_tiers:
            continue
        
        for use_log in [False, True]:
            log_label = "_log" if use_log else "_raw"
            run_name = f"{tier_name}_{config_name}{log_label}"
            
            y_target = y_train_log if use_log else y_train_raw
            
            X = train_df[feature_cols].values
            
            fold_r2s = []
            fold_rmses = []
            fold_maes = []
            oof_preds = np.full(len(train_df), np.nan)
            
            for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X, y_target, groups)):
                X_tr, X_va = X[tr_idx], X[va_idx]
                y_tr, y_va = y_target[tr_idx], y_target[va_idx]
                
                # For evaluation, always compare in original scale
                y_va_original = y_train_raw[va_idx]
                
                # Handle NaN in features
                col_means = np.nanmean(X_tr, axis=0)
                col_means = np.where(np.isnan(col_means), 0, col_means)
                
                X_tr_clean = X_tr.copy()
                X_va_clean = X_va.copy()
                for j in range(X_tr.shape[1]):
                    X_tr_clean[np.isnan(X_tr_clean[:, j]), j] = col_means[j]
                    X_va_clean[np.isnan(X_va_clean[:, j]), j] = col_means[j]
                
                # Scale for linear models
                if config['type'] in ['ridge', 'elasticnet']:
                    scaler = StandardScaler()
                    X_tr_clean = scaler.fit_transform(X_tr_clean)
                    X_va_clean = scaler.transform(X_va_clean)
                
                # Train model
                if config['type'] == 'xgboost':
                    model = xgb.XGBRegressor(**config['params'])
                    model.fit(X_tr_clean, y_tr, eval_set=[(X_va_clean, y_va)],
                              verbose=False)
                elif config['type'] == 'lightgbm':
                    model = lgb.LGBMRegressor(**config['params'])
                    model.fit(X_tr_clean, y_tr,
                              eval_set=[(X_va_clean, y_va)],
                              callbacks=[lgb.log_evaluation(period=0)])
                elif config['type'] == 'ridge':
                    model = Ridge(**config['params'])
                    model.fit(X_tr_clean, y_tr)
                elif config['type'] == 'elasticnet':
                    model = ElasticNet(**config['params'])
                    model.fit(X_tr_clean, y_tr)
                
                # Predict
                preds = model.predict(X_va_clean)
                
                # Convert back from log if needed
                if use_log:
                    preds = np.expm1(preds)
                
                # Clip to non-negative (EC can't be negative)
                preds = np.maximum(preds, 0)
                
                # Store OOF predictions
                oof_preds[va_idx] = preds
                
                # Metrics (original scale)
                fold_r2 = r2_score(y_va_original, preds)
                fold_rmse = np.sqrt(mean_squared_error(y_va_original, preds))
                fold_mae = mean_absolute_error(y_va_original, preds)
                
                fold_r2s.append(fold_r2)
                fold_rmses.append(fold_rmse)
                fold_maes.append(fold_mae)
            
            # Aggregate metrics
            mean_r2 = np.mean(fold_r2s)
            std_r2 = np.std(fold_r2s)
            mean_rmse = np.mean(fold_rmses)
            mean_mae = np.mean(fold_maes)
            
            results[run_name] = {
                'tier': tier_name,
                'config': config_name,
                'log_transform': use_log,
                'n_features': len(feature_cols),
                'mean_r2': mean_r2,
                'std_r2': std_r2,
                'mean_rmse': mean_rmse,
                'mean_mae': mean_mae,
                'fold_r2s': fold_r2s,
                'fold_rmses': fold_rmses,
                'fold_maes': fold_maes,
                'oof_preds': oof_preds,
                'feature_cols': feature_cols,
                'model_config': config,
            }
            
            # Progress print
            r2_str = f"{mean_r2:+.4f}" if mean_r2 < 0 else f"+{mean_r2:.4f}"
            print(f"  {run_name:50s}  R²={r2_str} ±{std_r2:.3f}  "
                  f"RMSE={mean_rmse:.1f}  MAE={mean_mae:.1f}")


# ============================================================
# [7] RESULTS SUMMARY — RANKED
# ============================================================
print(f"\n{'='*70}")
print(f"[7] RESULTS SUMMARY — EC MODELS RANKED BY R²")
print(f"{'='*70}")

ranked = sorted(results.items(), key=lambda x: x[1]['mean_r2'], reverse=True)

print(f"\n{'Rank':<5} {'Config':<50} {'R²':>8} {'±std':>7} {'RMSE':>8} {'MAE':>8} {'Feats':>6}")
print(f"{'─'*93}")

for rank, (name, res) in enumerate(ranked, 1):
    r2_str = f"{res['mean_r2']:+.4f}"
    print(f"{rank:<5} {name:<50} {r2_str:>8} ±{res['std_r2']:.3f} "
          f"{res['mean_rmse']:>8.1f} {res['mean_mae']:>8.1f} {res['n_features']:>6}")

best_name, best_res = ranked[0]
print(f"\n★ BEST: {best_name}")
print(f"  R² = {best_res['mean_r2']:+.4f} ± {best_res['std_r2']:.3f}")
print(f"  RMSE = {best_res['mean_rmse']:.1f}")
print(f"  MAE = {best_res['mean_mae']:.1f}")
print(f"  Features: {best_res['n_features']}")

# Show per-fold detail for best
print(f"\n  Per-fold R² breakdown:")
for i, (r2, rmse, mae) in enumerate(zip(best_res['fold_r2s'], best_res['fold_rmses'], best_res['fold_maes'])):
    print(f"    Fold {i}: R²={r2:+.4f}  RMSE={rmse:.1f}  MAE={mae:.1f}")


# ============================================================
# [8] FEATURE IMPORTANCE — BEST MODEL
# ============================================================
print(f"\n{'='*70}")
print(f"[8] FEATURE IMPORTANCE — BEST CONFIG")
print(f"{'='*70}")

best_cfg = best_res['model_config']
best_features = best_res['feature_cols']
use_log_best = best_res['log_transform']

y_full = y_train_log if use_log_best else y_train_raw
X_full = train_df[best_features].values

# Impute NaN
col_means = np.nanmean(X_full, axis=0)
col_means = np.where(np.isnan(col_means), 0, col_means)
X_full_clean = X_full.copy()
for j in range(X_full.shape[1]):
    X_full_clean[np.isnan(X_full_clean[:, j]), j] = col_means[j]

# Train on full data for importance + final predictions
if best_cfg['type'] == 'xgboost':
    final_model = xgb.XGBRegressor(**best_cfg['params'])
    final_model.fit(X_full_clean, y_full, verbose=False)
    importances = final_model.feature_importances_
elif best_cfg['type'] == 'lightgbm':
    final_model = lgb.LGBMRegressor(**best_cfg['params'])
    final_model.fit(X_full_clean, y_full, callbacks=[lgb.log_evaluation(period=0)])
    importances = final_model.feature_importances_ / final_model.feature_importances_.sum()
elif best_cfg['type'] in ['ridge', 'elasticnet']:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full_clean)
    if best_cfg['type'] == 'ridge':
        final_model = Ridge(**best_cfg['params'])
    else:
        final_model = ElasticNet(**best_cfg['params'])
    final_model.fit(X_scaled, y_full)
    importances = np.abs(final_model.coef_)
    importances = importances / importances.sum()

# Show top 20
sorted_idx = np.argsort(importances)[::-1]
print(f"\n  Top 20 features for EC ({best_name}):")
for rank, idx in enumerate(sorted_idx[:20], 1):
    feat = best_features[idx]
    imp = importances[idx]
    # Flag V7-new features
    v7_flag = " ★NEW" if feat in [
        'is_cool_dry_season', 'is_drought_period',
        'cool_dry_x_swir16', 'cool_dry_x_nir', 'cool_dry_x_ndmi', 'cool_dry_x_swir_ratio',
        'drought_x_swir16', 'drought_x_swir22', 'drought_x_ndmi', 'drought_x_aridity',
        'st_x_vpd', 'st_x_aridity', 'swir_mean', 'swir_diff',
        'ppt_anomaly', 'temp_anomaly'
    ] else ""
    print(f"    {rank:2d}. {feat:35s}  {imp:.4f}{v7_flag}")


# ============================================================
# [9] GENERATE SUBMISSION (EC column only)
# ============================================================
print(f"\n{'='*70}")
print(f"[9] GENERATE EC PREDICTIONS FOR VALIDATION")
print(f"{'='*70}")

X_val = val_df[best_features].values
X_val_clean = X_val.copy()
for j in range(X_val.shape[1]):
    X_val_clean[np.isnan(X_val_clean[:, j]), j] = col_means[j]

if best_cfg['type'] in ['ridge', 'elasticnet']:
    X_val_clean = scaler.transform(X_val_clean)

val_preds = final_model.predict(X_val_clean)

if use_log_best:
    val_preds = np.expm1(val_preds)

val_preds = np.maximum(val_preds, 0)  # EC can't be negative

print(f"  EC predictions: mean={np.mean(val_preds):.1f}, "
      f"median={np.median(val_preds):.1f}, "
      f"min={np.min(val_preds):.1f}, max={np.max(val_preds):.1f}")
print(f"  Training EC:    mean={train_df[TARGET].mean():.1f}, "
      f"median={train_df[TARGET].median():.1f}")

# Load submission template
template_path = os.path.join(BASE_DIR, "Jupyter Notebook Package", "submission_template.csv")
submission = pd.read_csv(template_path)
submission['Electrical Conductance'] = val_preds

# Save partial submission (EC only — TA and DRP will be filled later)
ec_submission_path = os.path.join(SUBMISSION_DIR, f'submission_v7_ec_{ts}.csv')
submission.to_csv(ec_submission_path, index=False)
print(f"\n  Saved: {ec_submission_path}")

# Also save as latest
submission.to_csv(os.path.join(SUBMISSION_DIR, 'submission_v7_ec.csv'), index=False)
print(f"  Saved: submissions/submission_v7_ec.csv")


# ============================================================
# [10] COMPARE WITH PREVIOUS VERSIONS
# ============================================================
print(f"\n{'='*70}")
print(f"[10] COMPARISON WITH PREVIOUS VERSIONS")
print(f"{'='*70}")

# V6 EC result from experiment log: -0.4960
print(f"""
  Version  | EC Spatial CV R²  | Notes
  ─────────|───────────────────|──────────────────────────
  V0       | ~ -0.60 (all)     | Baseline XGBoost
  V1       | ~ -0.43           | Expanded climate + soil
  V2       | ~ -0.30 (all)     | Removed coordinates
  V4       | ~ -0.35 (all)     | Research features
  V6       | -0.4960           | Full Landsat + 173 features
  V7 (EC)  | {best_res['mean_r2']:+.4f}           | ← THIS RUN (focused {best_res['n_features']} features)
""")


# ============================================================
# [11] ALSO TRAIN TOP-3 MODELS FOR ENSEMBLE
# ============================================================
print(f"\n{'='*70}")
print(f"[11] TOP-3 ENSEMBLE FOR EC")
print(f"{'='*70}")

top3 = ranked[:3]
ensemble_preds = []

for name, res in top3:
    cfg = res['model_config']
    feats = res['feature_cols']
    use_log = res['log_transform']
    
    y_t = y_train_log if use_log else y_train_raw
    X_t = train_df[feats].values
    
    # Impute
    cm = np.nanmean(X_t, axis=0)
    cm = np.where(np.isnan(cm), 0, cm)
    X_t_c = X_t.copy()
    for j in range(X_t.shape[1]):
        X_t_c[np.isnan(X_t_c[:, j]), j] = cm[j]
    
    X_v = val_df[feats].values.copy()
    for j in range(X_v.shape[1]):
        X_v[np.isnan(X_v[:, j]), j] = cm[j]
    
    if cfg['type'] in ['ridge', 'elasticnet']:
        sc = StandardScaler()
        X_t_c = sc.fit_transform(X_t_c)
        X_v = sc.transform(X_v)
    
    if cfg['type'] == 'xgboost':
        m = xgb.XGBRegressor(**cfg['params'])
        m.fit(X_t_c, y_t, verbose=False)
    elif cfg['type'] == 'lightgbm':
        m = lgb.LGBMRegressor(**cfg['params'])
        m.fit(X_t_c, y_t, callbacks=[lgb.log_evaluation(period=0)])
    elif cfg['type'] == 'ridge':
        m = Ridge(**cfg['params'])
        m.fit(X_t_c, y_t)
    elif cfg['type'] == 'elasticnet':
        m = ElasticNet(**cfg['params'])
        m.fit(X_t_c, y_t)
    
    preds = m.predict(X_v)
    if use_log:
        preds = np.expm1(preds)
    preds = np.maximum(preds, 0)
    ensemble_preds.append(preds)
    
    print(f"  {name}: mean_pred={np.mean(preds):.1f}, R²={res['mean_r2']:+.4f}")

# Average ensemble
ec_ensemble = np.mean(ensemble_preds, axis=0)
print(f"\n  Ensemble (top-3 avg): mean_pred={np.mean(ec_ensemble):.1f}, "
      f"median={np.median(ec_ensemble):.1f}")

# Save ensemble submission
submission_ens = pd.read_csv(template_path)
submission_ens['Electrical Conductance'] = ec_ensemble
submission_ens.to_csv(os.path.join(SUBMISSION_DIR, f'submission_v7_ec_ensemble_{ts}.csv'), index=False)
submission_ens.to_csv(os.path.join(SUBMISSION_DIR, 'submission_v7_ec_ensemble.csv'), index=False)
print(f"  Saved: submissions/submission_v7_ec_ensemble.csv")


# ============================================================
# [12] SAVE DETAILED RESULTS LOG
# ============================================================
print(f"\n{'='*70}")
print(f"[12] SAVING DETAILED RESULTS")
print(f"{'='*70}")

results_path = os.path.join(LOG_DIR, f'v7_ec_results_{ts}.txt')
with open(results_path, 'w', encoding='utf-8') as f:
    f.write(f"V7 EC Results — {ts}\n")
    f.write(f"{'='*80}\n\n")
    
    f.write(f"{'Rank':<5} {'Config':<50} {'R²':>8} {'±std':>7} {'RMSE':>8} {'MAE':>8} {'Feats':>6}\n")
    f.write(f"{'─'*93}\n")
    
    for rank, (name, res) in enumerate(ranked, 1):
        r2_str = f"{res['mean_r2']:+.4f}"
        f.write(f"{rank:<5} {name:<50} {r2_str:>8} ±{res['std_r2']:.3f} "
                f"{res['mean_rmse']:>8.1f} {res['mean_mae']:>8.1f} {res['n_features']:>6}\n")
    
    f.write(f"\n\nBest: {best_name}\n")
    f.write(f"R² = {best_res['mean_r2']:+.4f} ± {best_res['std_r2']:.3f}\n")
    f.write(f"RMSE = {best_res['mean_rmse']:.1f}\n")
    f.write(f"MAE = {best_res['mean_mae']:.1f}\n")
    
    f.write(f"\n\nTop 20 Feature Importances:\n")
    for rank, idx in enumerate(sorted_idx[:20], 1):
        f.write(f"  {rank:2d}. {best_features[idx]:35s}  {importances[idx]:.4f}\n")
    
    f.write(f"\n\nPer-fold details:\n")
    for i, (r2, rmse, mae) in enumerate(zip(best_res['fold_r2s'], best_res['fold_rmses'], best_res['fold_maes'])):
        f.write(f"  Fold {i}: R²={r2:+.4f}  RMSE={rmse:.1f}  MAE={mae:.1f}\n")

print(f"  Saved: {results_path}")


# ============================================================
# [13] FINAL SUMMARY
# ============================================================
print(f"""
{'='*70}
V7 EC MODEL — FINAL SUMMARY
{'='*70}

TARGET: Electrical Conductance (EC)

BEST CONFIG: {best_name}
  R²   = {best_res['mean_r2']:+.4f} ± {best_res['std_r2']:.3f}
  RMSE  = {best_res['mean_rmse']:.1f} mS/m
  MAE   = {best_res['mean_mae']:.1f} mS/m
  Features: {best_res['n_features']}

V7 KEY INNOVATIONS:
  - Cool dry season flag (May-Oct) → optimal EC sensing window
  - El Niño drought flag (Oct 2014+) → concentrated ion signal
  - Season × SWIR/NIR interactions
  - Drought × SWIR interactions
  - Feature discipline: {best_res['n_features']} focused features (vs V6's 173)
  - SWIR/NIR/Temperature prioritization (research-backed)
  - LightGBM added alongside XGBoost
  - RMSE + MAE metrics added

SUBMISSIONS:
  1. submissions/submission_v7_ec.csv (best single model)
  2. submissions/submission_v7_ec_ensemble.csv (top-3 ensemble)

NOTE: These submissions only have EC filled in.
      TA and DRP columns need to be filled separately.
      TA model will be built next as V7b.

NEXT STEPS:
  1. Run this script: python scripts/14_model_v7_ec.py
  2. Check results — does focused feature set beat V6's 173-feature approach?
  3. Do V7 seasonal/drought features appear in top importances?
  4. Build V7b for Total Alkalinity
  5. Build V7c for DRP (using EC predictions as feature)
  6. Combine into full submission and submit to leaderboard
{'='*70}
""")

"""
============================================================
EY Water Quality Challenge - V7b: EC + Land Use + Geology + KNN
============================================================
Builds on V7's findings:
  - V7 showed CORE 32 features beat 69+ features
  - Terrain/soil dominate feature importance
  - More features = more overfitting in spatial CV

V7b adds:
  1. ESA WorldCover land use (% cropland, urban, bare at buffer radii)
  2. Geology/lithology data (Macrostrat + soil proxies)
  3. KNN blending (nearest-neighbor predictions from similar training locations)
  4. Focused feature set: V7 CORE + best land use + geology features

KEY HYPOTHESIS:
  Land use and geology are the MISSING SPATIAL features.
  Two locations with identical terrain/soil but different land use
  (urban vs forest) will have very different EC.
  Geology determines the BASELINE mineral content of water.
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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
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
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "extracted")
SUBMISSION_DIR = os.path.join(BASE_DIR, "submissions")
LOG_DIR = os.path.join(BASE_DIR, "experiment_logs")

TARGET = "Electrical Conductance"
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

print(f"""
{'='*70}
EY WATER QUALITY - V7b: EC + LAND USE + GEOLOGY + KNN
{'='*70}
Target: {TARGET}
Date: {ts}
{'='*70}
""")

# ============================================================
# [1] LOAD ALL DATA
# ============================================================
print(f"[1] Loading data...")

# V6 features (base)
train_df = pd.read_csv(os.path.join(DATA_DIR, "v6_train_features.csv"))
val_df = pd.read_csv(os.path.join(DATA_DIR, "v6_val_features.csv"))

train_df['Sample Date'] = pd.to_datetime(train_df['Sample Date'], format='%d-%m-%Y', dayfirst=True, errors='coerce')
val_df['Sample Date'] = pd.to_datetime(val_df['Sample Date'], format='%d-%m-%Y', dayfirst=True, errors='coerce')
if train_df['Sample Date'].isna().sum() > len(train_df) * 0.5:
    train_df['Sample Date'] = pd.to_datetime(train_df['Sample Date'], errors='coerce')
    val_df['Sample Date'] = pd.to_datetime(val_df['Sample Date'], errors='coerce')

print(f"  Train: {len(train_df)} rows, Val: {len(val_df)} rows")

# WorldCover land use
wc_path = os.path.join(DATA_DIR, "worldcover_data.csv")
has_worldcover = os.path.exists(wc_path)
if has_worldcover:
    wc_df = pd.read_csv(wc_path)
    print(f"  WorldCover: {len(wc_df)} locations loaded")
    
    # Merge with train/val on Latitude/Longitude
    wc_cols = [c for c in wc_df.columns if c.startswith('lc_')]
    merge_cols = ['Latitude', 'Longitude'] + wc_cols
    
    # Round for merge safety
    wc_df['Latitude'] = wc_df['Latitude'].round(6)
    wc_df['Longitude'] = wc_df['Longitude'].round(6)
    train_df['lat_round'] = train_df['Latitude'].round(6)
    train_df['lon_round'] = train_df['Longitude'].round(6)
    val_df['lat_round'] = val_df['Latitude'].round(6)
    val_df['lon_round'] = val_df['Longitude'].round(6)
    wc_df_merge = wc_df.rename(columns={'Latitude': 'lat_round', 'Longitude': 'lon_round'})
    
    train_df = train_df.merge(wc_df_merge[['lat_round', 'lon_round'] + wc_cols], 
                               on=['lat_round', 'lon_round'], how='left')
    val_df = val_df.merge(wc_df_merge[['lat_round', 'lon_round'] + wc_cols],
                           on=['lat_round', 'lon_round'], how='left')
    
    # Check coverage
    sample_col = 'lc_5km_pct_built_up'
    if sample_col in train_df.columns:
        train_cov = train_df[sample_col].notna().sum()
        val_cov = val_df[sample_col].notna().sum()
        print(f"  WorldCover merge: train={train_cov}/{len(train_df)}, val={val_cov}/{len(val_df)}")
    else:
        print(f"  WARNING: WorldCover columns not found after merge")
        has_worldcover = False
else:
    print(f"  WorldCover: NOT FOUND at {wc_path}")
    print(f"  Run scripts/15_extract_worldcover.py first")

# Geology data
geo_path = os.path.join(DATA_DIR, "geology_data.csv")
has_geology = os.path.exists(geo_path)
if has_geology:
    geo_df = pd.read_csv(geo_path)
    print(f"  Geology: {len(geo_df)} locations loaded")
    
    geo_cols = [c for c in geo_df.columns if c.startswith('geo_') and c not in ['geo_source', 'geo_lith_raw', 'geo_formation', 'geo_rock_type', 'geo_lith_category']]
    
    geo_df['lat_round'] = geo_df['Latitude'].round(6)
    geo_df['lon_round'] = geo_df['Longitude'].round(6)
    
    train_df = train_df.merge(geo_df[['lat_round', 'lon_round'] + geo_cols].drop_duplicates(subset=['lat_round', 'lon_round']),
                               on=['lat_round', 'lon_round'], how='left')
    val_df = val_df.merge(geo_df[['lat_round', 'lon_round'] + geo_cols].drop_duplicates(subset=['lat_round', 'lon_round']),
                           on=['lat_round', 'lon_round'], how='left')
    
    geo_sample = 'geo_ec_impact'
    if geo_sample in train_df.columns:
        train_cov = train_df[geo_sample].notna().sum()
        val_cov = val_df[geo_sample].notna().sum()
        print(f"  Geology merge: train={train_cov}/{len(train_df)}, val={val_cov}/{len(val_df)}")
    else:
        print(f"  WARNING: Geology columns not found after merge")
        has_geology = False
else:
    print(f"  Geology: NOT FOUND at {geo_path}")
    print(f"  Run scripts/16_extract_geology.py first")


# ============================================================
# [2] DEFINE FEATURE TIERS
# ============================================================
print(f"\n{'='*70}")
print(f"[2] DEFINING FEATURE TIERS")
print(f"{'='*70}")

# V7 CORE winners (the 32 features that beat everything in V7)
V7_CORE = [
    'unified_swir16', 'unified_swir22', 'unified_nir', 'unified_green',
    'ratio_swir16_nir', 'ratio_swir22_swir16', 'ratio_nir_swir22',
    'unified_NDMI', 'unified_MNDWI',
    'ls_st_celsius', 'has_surface_temp',
    'elevation_m', 'slope_deg', 'terrain_roughness', 'elev_range_500m',
    'soil_soc', 'soil_sand', 'soil_clay', 'soil_cec', 'soil_phh2o',
    'soil_buffer_capacity', 'soil_sand_clay_ratio',
    'ppt', 'tmax', 'vpd', 'aridity_index',
    'ppt_3m_sum', 'tmax_3m_avg',
    'month_sin', 'month_cos',
]

# Land use features (best picks from WorldCover)
LAND_USE_FEATURES = []
if has_worldcover:
    # Use 5km buffer as primary (good balance of local vs regional context)
    # + 1km for immediate surroundings + 10km for broader landscape
    for radius in ['1km', '5km', '10km']:
        for cls in ['built_up', 'cropland', 'bare_sparse', 'tree_cover', 'grassland', 'water']:
            col = f'lc_{radius}_pct_{cls}'
            if col in train_df.columns:
                LAND_USE_FEATURES.append(col)
    
    # Derived features
    for radius in ['1km', '5km', '10km']:
        for feat in ['human_impact', 'natural_veg', 'urban_ratio']:
            col = f'lc_{radius}_{feat}'
            if col in train_df.columns:
                LAND_USE_FEATURES.append(col)

# Geology features
GEOLOGY_FEATURES = []
if has_geology:
    for col in train_df.columns:
        if col.startswith('geo_') and col not in ['geo_source', 'geo_lith_raw', 'geo_formation', 'geo_rock_type', 'geo_lith_category']:
            if train_df[col].notna().sum() > len(train_df) * 0.1:  # at least 10% coverage
                GEOLOGY_FEATURES.append(col)

print(f"  V7 CORE features: {len(V7_CORE)}")
print(f"  Land use features: {len(LAND_USE_FEATURES)}")
print(f"  Geology features: {len(GEOLOGY_FEATURES)}")

# Build tiers
FEATURE_TIERS = {}

# Tier A: V7 Core only (baseline — should match V7 results)
FEATURE_TIERS['A_V7_CORE'] = [f for f in V7_CORE if f in train_df.columns]

# Tier B: V7 Core + Land Use
FEATURE_TIERS['B_CORE_LANDUSE'] = FEATURE_TIERS['A_V7_CORE'] + LAND_USE_FEATURES

# Tier C: V7 Core + Geology
FEATURE_TIERS['C_CORE_GEOLOGY'] = FEATURE_TIERS['A_V7_CORE'] + GEOLOGY_FEATURES

# Tier D: V7 Core + Land Use + Geology (everything)
FEATURE_TIERS['D_CORE_ALL'] = FEATURE_TIERS['A_V7_CORE'] + LAND_USE_FEATURES + GEOLOGY_FEATURES

# Tier E: Focused — V7 Core + only the most impactful land use (5km buffer)
focused_lc = [f for f in LAND_USE_FEATURES if '5km' in f]
FEATURE_TIERS['E_FOCUSED'] = FEATURE_TIERS['A_V7_CORE'] + focused_lc + GEOLOGY_FEATURES

for tier_name, features in FEATURE_TIERS.items():
    avail = [f for f in features if f in train_df.columns]
    FEATURE_TIERS[tier_name] = avail
    print(f"  {tier_name}: {len(avail)} features")


# ============================================================
# [3] SPATIAL CV SETUP
# ============================================================
print(f"\n[3] Spatial CV setup...")
N_CLUSTERS = 8
coords = train_df[['Latitude', 'Longitude']].values
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=20)
train_df['spatial_cluster'] = kmeans.fit_predict(coords)

y_train_raw = train_df[TARGET].values
groups = train_df['spatial_cluster'].values
gkf = GroupKFold(n_splits=N_CLUSTERS)


# ============================================================
# [4] MODEL CONFIGURATIONS (focused on V7 winners)
# ============================================================
MODEL_CONFIGS = {
    'xgb_conservative': {
        'type': 'xgboost',
        'params': {
            'n_estimators': 200, 'max_depth': 2, 'learning_rate': 0.02,
            'subsample': 0.5, 'colsample_bytree': 0.5,
            'min_child_weight': 30, 'reg_alpha': 5.0, 'reg_lambda': 20.0,
            'random_state': 42, 'verbosity': 0,
        }
    },
    'xgb_moderate': {
        'type': 'xgboost',
        'params': {
            'n_estimators': 300, 'max_depth': 3, 'learning_rate': 0.03,
            'subsample': 0.6, 'colsample_bytree': 0.6,
            'min_child_weight': 20, 'reg_alpha': 2.0, 'reg_lambda': 10.0,
            'random_state': 42, 'verbosity': 0,
        }
    },
    'ridge': {
        'type': 'ridge',
        'params': {'alpha': 10.0},
    },
}

if HAS_LGBM:
    MODEL_CONFIGS['lgbm_conservative'] = {
        'type': 'lightgbm',
        'params': {
            'n_estimators': 300, 'max_depth': 3, 'learning_rate': 0.02,
            'subsample': 0.5, 'colsample_bytree': 0.5,
            'min_child_samples': 40, 'reg_alpha': 3.0, 'reg_lambda': 15.0,
            'num_leaves': 8, 'random_state': 42, 'verbosity': -1,
        }
    }


# ============================================================
# [5] TRAINING LOOP
# ============================================================
print(f"\n{'='*70}")
print(f"[5] TRAINING — EC MODELS")
print(f"{'='*70}")

results = {}

for tier_name, feature_cols in FEATURE_TIERS.items():
    if not feature_cols:
        continue
    
    for config_name, config in MODEL_CONFIGS.items():
        run_name = f"{tier_name}_{config_name}"
        
        X = train_df[feature_cols].values
        fold_r2s, fold_rmses, fold_maes = [], [], []
        oof_preds = np.full(len(train_df), np.nan)
        
        for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X, y_train_raw, groups)):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y_train_raw[tr_idx], y_train_raw[va_idx]
            
            # Impute NaN
            col_means = np.nanmean(X_tr, axis=0)
            col_means = np.where(np.isnan(col_means), 0, col_means)
            X_tr_c = X_tr.copy()
            X_va_c = X_va.copy()
            for j in range(X_tr.shape[1]):
                X_tr_c[np.isnan(X_tr_c[:, j]), j] = col_means[j]
                X_va_c[np.isnan(X_va_c[:, j]), j] = col_means[j]
            
            # Scale for linear
            if config['type'] == 'ridge':
                scaler = StandardScaler()
                X_tr_c = scaler.fit_transform(X_tr_c)
                X_va_c = scaler.transform(X_va_c)
            
            # Train
            if config['type'] == 'xgboost':
                model = xgb.XGBRegressor(**config['params'])
                model.fit(X_tr_c, y_tr, eval_set=[(X_va_c, y_va)], verbose=False)
            elif config['type'] == 'lightgbm':
                model = lgb.LGBMRegressor(**config['params'])
                model.fit(X_tr_c, y_tr, eval_set=[(X_va_c, y_va)],
                          callbacks=[lgb.log_evaluation(period=0)])
            elif config['type'] == 'ridge':
                model = Ridge(**config['params'])
                model.fit(X_tr_c, y_tr)
            
            preds = np.maximum(model.predict(X_va_c), 0)
            oof_preds[va_idx] = preds
            
            fold_r2s.append(r2_score(y_va, preds))
            fold_rmses.append(np.sqrt(mean_squared_error(y_va, preds)))
            fold_maes.append(mean_absolute_error(y_va, preds))
        
        mean_r2 = np.mean(fold_r2s)
        results[run_name] = {
            'tier': tier_name, 'config': config_name,
            'n_features': len(feature_cols),
            'mean_r2': mean_r2, 'std_r2': np.std(fold_r2s),
            'mean_rmse': np.mean(fold_rmses), 'mean_mae': np.mean(fold_maes),
            'fold_r2s': fold_r2s, 'oof_preds': oof_preds,
            'feature_cols': feature_cols, 'model_config': config,
        }
        
        print(f"  {run_name:45s}  R2={mean_r2:+.4f} +/-{np.std(fold_r2s):.3f}  "
              f"RMSE={np.mean(fold_rmses):.1f}  MAE={np.mean(fold_maes):.1f}  ({len(feature_cols)} feats)")


# ============================================================
# [6] KNN BLENDING
# ============================================================
print(f"\n{'='*70}")
print(f"[6] KNN BLENDING")
print(f"{'='*70}")
print(f"  For each validation point, find K most similar training locations")
print(f"  by feature similarity, and blend their EC values with model predictions")

# Use the V7 CORE features for KNN distance computation
knn_features = FEATURE_TIERS.get('A_V7_CORE', V7_CORE)
knn_features = [f for f in knn_features if f in train_df.columns]

X_train_knn = train_df[knn_features].values.copy()
X_val_knn = val_df[knn_features].values.copy()

# Impute NaN
km = np.nanmean(X_train_knn, axis=0)
km = np.where(np.isnan(km), 0, km)
for j in range(X_train_knn.shape[1]):
    X_train_knn[np.isnan(X_train_knn[:, j]), j] = km[j]
    X_val_knn[np.isnan(X_val_knn[:, j]), j] = km[j]

# Scale features for distance computation
knn_scaler = StandardScaler()
X_train_knn_scaled = knn_scaler.fit_transform(X_train_knn)
X_val_knn_scaled = knn_scaler.transform(X_val_knn)

# Test different K values with spatial CV
for K in [5, 10, 20, 50]:
    knn_model = KNeighborsRegressor(n_neighbors=K, weights='distance', metric='euclidean')
    
    fold_r2s = []
    oof_knn = np.full(len(train_df), np.nan)
    
    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X_train_knn_scaled, y_train_raw, groups)):
        X_tr = X_train_knn_scaled[tr_idx]
        X_va = X_train_knn_scaled[va_idx]
        y_tr = y_train_raw[tr_idx]
        y_va = y_train_raw[va_idx]
        
        knn_model.fit(X_tr, y_tr)
        preds = np.maximum(knn_model.predict(X_va), 0)
        oof_knn[va_idx] = preds
        fold_r2s.append(r2_score(y_va, preds))
    
    mean_r2 = np.mean(fold_r2s)
    rmse = np.sqrt(mean_squared_error(y_train_raw[~np.isnan(oof_knn)], oof_knn[~np.isnan(oof_knn)]))
    mae = mean_absolute_error(y_train_raw[~np.isnan(oof_knn)], oof_knn[~np.isnan(oof_knn)])
    
    results[f'KNN_K{K}'] = {
        'tier': 'KNN', 'config': f'K={K}',
        'n_features': len(knn_features),
        'mean_r2': mean_r2, 'std_r2': np.std(fold_r2s),
        'mean_rmse': rmse, 'mean_mae': mae,
        'fold_r2s': fold_r2s, 'oof_preds': oof_knn,
        'feature_cols': knn_features, 'model_config': {'type': 'knn', 'K': K},
    }
    
    print(f"  KNN K={K:3d}:  R2={mean_r2:+.4f} +/-{np.std(fold_r2s):.3f}  RMSE={rmse:.1f}  MAE={mae:.1f}")


# ============================================================
# [7] HYBRID: MODEL + KNN BLEND
# ============================================================
print(f"\n{'='*70}")
print(f"[7] HYBRID BLENDING: Best Model + KNN")
print(f"{'='*70}")

# Get best model and best KNN
ranked_models = sorted(
    [(k, v) for k, v in results.items() if not k.startswith('KNN')],
    key=lambda x: x[1]['mean_r2'], reverse=True
)
ranked_knn = sorted(
    [(k, v) for k, v in results.items() if k.startswith('KNN')],
    key=lambda x: x[1]['mean_r2'], reverse=True
)

if ranked_models and ranked_knn:
    best_model_name, best_model = ranked_models[0]
    best_knn_name, best_knn = ranked_knn[0]
    
    model_oof = best_model['oof_preds']
    knn_oof = best_knn['oof_preds']
    
    # Test different blend weights
    print(f"  Blending: {best_model_name} + {best_knn_name}")
    
    for model_weight in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        knn_weight = 1.0 - model_weight
        
        # Only blend where both have predictions
        mask = ~np.isnan(model_oof) & ~np.isnan(knn_oof)
        blended = model_weight * model_oof[mask] + knn_weight * knn_oof[mask]
        blended = np.maximum(blended, 0)
        
        r2 = r2_score(y_train_raw[mask], blended)
        rmse = np.sqrt(mean_squared_error(y_train_raw[mask], blended))
        mae = mean_absolute_error(y_train_raw[mask], blended)
        
        label = f"BLEND_{int(model_weight*100)}model_{int(knn_weight*100)}knn"
        results[label] = {
            'tier': 'BLEND', 'config': f'{model_weight:.1f}/{knn_weight:.1f}',
            'n_features': best_model['n_features'],
            'mean_r2': r2, 'std_r2': 0,
            'mean_rmse': rmse, 'mean_mae': mae,
            'fold_r2s': [r2], 'oof_preds': None,
            'feature_cols': best_model['feature_cols'],
            'model_config': {'type': 'blend'},
        }
        
        print(f"  {label:40s}  R2={r2:+.4f}  RMSE={rmse:.1f}  MAE={mae:.1f}")


# ============================================================
# [8] FINAL RANKINGS
# ============================================================
print(f"\n{'='*70}")
print(f"[8] ALL RESULTS RANKED BY R2")
print(f"{'='*70}")

ranked_all = sorted(results.items(), key=lambda x: x[1]['mean_r2'], reverse=True)

print(f"\n{'Rank':<5} {'Config':<48} {'R2':>8} {'RMSE':>8} {'MAE':>8} {'Feats':>6}")
print(f"{'-'*90}")

for rank, (name, res) in enumerate(ranked_all[:25], 1):
    print(f"{rank:<5} {name:<48} {res['mean_r2']:+.4f} {res['mean_rmse']:>8.1f} "
          f"{res['mean_mae']:>8.1f} {res['n_features']:>6}")

best_name, best_res = ranked_all[0]
print(f"\n  BEST OVERALL: {best_name}")
print(f"  R2 = {best_res['mean_r2']:+.4f}")
print(f"  RMSE = {best_res['mean_rmse']:.1f}")
print(f"  MAE = {best_res['mean_mae']:.1f}")


# ============================================================
# [9] FEATURE IMPORTANCE — BEST NON-KNN MODEL
# ============================================================
print(f"\n{'='*70}")
print(f"[9] FEATURE IMPORTANCE")
print(f"{'='*70}")

best_tree_name, best_tree = ranked_models[0]
best_cfg = best_tree['model_config']
best_feats = best_tree['feature_cols']

X_full = train_df[best_feats].values
col_means = np.nanmean(X_full, axis=0)
col_means = np.where(np.isnan(col_means), 0, col_means)
X_full_c = X_full.copy()
for j in range(X_full.shape[1]):
    X_full_c[np.isnan(X_full_c[:, j]), j] = col_means[j]

if best_cfg['type'] in ['ridge']:
    scaler = StandardScaler()
    X_full_c = scaler.fit_transform(X_full_c)
    final = Ridge(**best_cfg['params'])
    final.fit(X_full_c, y_train_raw)
    importances = np.abs(final.coef_) / np.abs(final.coef_).sum()
elif best_cfg['type'] == 'xgboost':
    final = xgb.XGBRegressor(**best_cfg['params'])
    final.fit(X_full_c, y_train_raw, verbose=False)
    importances = final.feature_importances_
elif best_cfg['type'] == 'lightgbm':
    final = lgb.LGBMRegressor(**best_cfg['params'])
    final.fit(X_full_c, y_train_raw, callbacks=[lgb.log_evaluation(period=0)])
    importances = final.feature_importances_ / final.feature_importances_.sum()

sorted_idx = np.argsort(importances)[::-1]
print(f"\n  Top 25 features ({best_tree_name}):")
for rank, idx in enumerate(sorted_idx[:25], 1):
    feat = best_feats[idx]
    imp = importances[idx]
    tag = ""
    if feat.startswith('lc_'): tag = " [LAND USE]"
    elif feat.startswith('geo_'): tag = " [GEOLOGY]"
    print(f"    {rank:2d}. {feat:40s} {imp:.4f}{tag}")


# ============================================================
# [10] GENERATE SUBMISSIONS
# ============================================================
print(f"\n{'='*70}")
print(f"[10] GENERATING SUBMISSIONS")
print(f"{'='*70}")

template_path = os.path.join(BASE_DIR, "ey_challenge_notebooks", "submission_template.csv")
submission = pd.read_csv(template_path)

# Best model predictions
X_val = val_df[best_feats].values
X_val_c = X_val.copy()
for j in range(X_val.shape[1]):
    X_val_c[np.isnan(X_val_c[:, j]), j] = col_means[j]

if best_cfg['type'] == 'ridge':
    X_val_c = scaler.transform(X_val_c)

model_preds = np.maximum(final.predict(X_val_c), 0)

# KNN predictions
if ranked_knn:
    best_knn_K = ranked_knn[0][1]['model_config']['K']
    knn_final = KNeighborsRegressor(n_neighbors=best_knn_K, weights='distance')
    knn_final.fit(X_train_knn_scaled, y_train_raw)
    knn_preds = np.maximum(knn_final.predict(X_val_knn_scaled), 0)
    
    print(f"  Model preds: mean={np.mean(model_preds):.1f}, median={np.median(model_preds):.1f}")
    print(f"  KNN preds:   mean={np.mean(knn_preds):.1f}, median={np.median(knn_preds):.1f}")
    
    # Save model-only submission
    submission['Electrical Conductance'] = model_preds
    sub_path = os.path.join(SUBMISSION_DIR, f'submission_v7b_ec_model_{ts}.csv')
    submission.to_csv(sub_path, index=False)
    
    # Save KNN-only submission
    submission['Electrical Conductance'] = knn_preds
    sub_path = os.path.join(SUBMISSION_DIR, f'submission_v7b_ec_knn_{ts}.csv')
    submission.to_csv(sub_path, index=False)
    
    # Save blended submission (50/50 as default, adjust based on results)
    # Find best blend weight from results
    best_blend = sorted(
        [(k, v) for k, v in results.items() if k.startswith('BLEND')],
        key=lambda x: x[1]['mean_r2'], reverse=True
    )
    if best_blend:
        blend_name = best_blend[0][0]
        # Parse weight from name
        parts = blend_name.split('_')
        model_w = int(parts[1].replace('model', '')) / 100.0
    else:
        model_w = 0.5
    
    knn_w = 1.0 - model_w
    blended_preds = model_w * model_preds + knn_w * knn_preds
    blended_preds = np.maximum(blended_preds, 0)
    
    submission['Electrical Conductance'] = blended_preds
    sub_path = os.path.join(SUBMISSION_DIR, f'submission_v7b_ec_blend_{ts}.csv')
    submission.to_csv(sub_path, index=False)
    submission.to_csv(os.path.join(SUBMISSION_DIR, 'submission_v7b_ec.csv'), index=False)
    
    print(f"\n  Blended ({model_w:.0%} model + {knn_w:.0%} KNN):")
    print(f"    mean={np.mean(blended_preds):.1f}, median={np.median(blended_preds):.1f}")
    print(f"    Saved: submissions/submission_v7b_ec.csv")
else:
    submission['Electrical Conductance'] = model_preds
    submission.to_csv(os.path.join(SUBMISSION_DIR, 'submission_v7b_ec.csv'), index=False)
    print(f"  Saved: submissions/submission_v7b_ec.csv (model only, no KNN)")


# ============================================================
# [11] SAVE RESULTS LOG
# ============================================================
results_path = os.path.join(LOG_DIR, f'v7b_ec_results_{ts}.txt')
with open(results_path, 'w', encoding='utf-8') as f:
    f.write(f"V7b EC Results - {ts}\n{'='*80}\n\n")
    f.write(f"Data sources:\n")
    f.write(f"  WorldCover: {'YES' if has_worldcover else 'NO'}\n")
    f.write(f"  Geology: {'YES' if has_geology else 'NO'}\n\n")
    
    for rank, (name, res) in enumerate(ranked_all, 1):
        f.write(f"{rank:3d}. {name:48s} R2={res['mean_r2']:+.4f} RMSE={res['mean_rmse']:.1f} MAE={res['mean_mae']:.1f}\n")

print(f"\n  Results saved: {results_path}")


# ============================================================
# FINAL SUMMARY
# ============================================================
print(f"""
{'='*70}
V7b FINAL SUMMARY
{'='*70}

BEST OVERALL: {ranked_all[0][0]}
  R2   = {ranked_all[0][1]['mean_r2']:+.4f}
  RMSE = {ranked_all[0][1]['mean_rmse']:.1f}
  MAE  = {ranked_all[0][1]['mean_mae']:.1f}

BEST MODEL: {ranked_models[0][0]} (R2={ranked_models[0][1]['mean_r2']:+.4f})
BEST KNN:   {ranked_knn[0][0] if ranked_knn else 'N/A'} (R2={ranked_knn[0][1]['mean_r2']:+.4f if ranked_knn else 'N/A'})

DID LAND USE HELP?  Compare A_V7_CORE vs B_CORE_LANDUSE
DID GEOLOGY HELP?   Compare A_V7_CORE vs C_CORE_GEOLOGY
DID KNN HELP?       Compare best model vs KNN vs blend

SUBMISSIONS:
  submissions/submission_v7b_ec.csv (best blend)

NEXT: Build TA model (V7b_ta), then DRP, then combine for full submission
{'='*70}
""")

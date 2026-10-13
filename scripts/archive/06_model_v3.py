"""
============================================================
EY Water Quality Challenge - V3 Model
============================================================
KEY CHANGES FROM V2:
  V1: Raw Lat/Lon → model memorized coordinates (R²=0.006)
  V2: No coordinates → lost all location signal (R²=-0.087)
  
  V3 Strategy: NEAREST-NEIGHBOR TARGET ENCODING
  Give the model "what does water quality look like around here?"
  without giving it raw coordinates.

  For each sample, find K geographically closest training points
  and use their target statistics as features:
    - knn_{target}_mean, median, std
    - knn_{target}_dist_weighted_mean
  For K = 3, 5, 10 → 3 targets × 4 stats × 3 K values = 36 features

  Plus distance-based features:
    - dist_to_nearest_train, dist_to_centroid
    - n_train_within_50km, n_train_within_100km

  CRITICAL: Leakage prevention during CV — exclude fold samples
  from neighbor pool.

  Combined with ALL V2 geography-invariant features.
============================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import BallTree
import xgboost as xgb
import warnings
import os
import time

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, '..', 'ey_challenge_notebooks')
EXTRACTED_DIR = os.path.join(BASE_DIR, '..', 'data', 'extracted')
SUBMISSION_DIR = os.path.join(BASE_DIR, '..', 'submissions')
os.makedirs(SUBMISSION_DIR, exist_ok=True)

TARGETS = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
RANDOM_STATE = 42
EARTH_RADIUS_KM = 6371.0

print("=" * 70)
print("EY WATER QUALITY CHALLENGE - V3 MODEL")
print("Strategy: Nearest-Neighbor Target Encoding + V2 features")
print("=" * 70)

# ============================================================
# [1] LOAD & MERGE (same as V1/V2)
# ============================================================
print("\n[1] Loading and merging all data sources...")

wq_train = pd.read_csv(os.path.join(DATA_DIR, 'water_quality_training_dataset.csv'))
submission_template = pd.read_csv(os.path.join(DATA_DIR, 'submission_template.csv'))
landsat_train = pd.read_csv(os.path.join(DATA_DIR, 'landsat_features_training.csv'))
landsat_val = pd.read_csv(os.path.join(DATA_DIR, 'landsat_features_validation.csv'))
terra_exp_train = pd.read_csv(os.path.join(EXTRACTED_DIR, 'terraclimate_expanded_training.csv'))
terra_exp_val = pd.read_csv(os.path.join(EXTRACTED_DIR, 'terraclimate_expanded_validation.csv'))
elevation = pd.read_csv(os.path.join(EXTRACTED_DIR, 'elevation_data.csv'))
soil = pd.read_csv(os.path.join(EXTRACTED_DIR, 'soil_data.csv'))

join_keys = ['Latitude', 'Longitude', 'Sample Date']
loc_keys = ['Latitude', 'Longitude']

train_df = wq_train.copy()
train_df = train_df.merge(landsat_train, on=join_keys, how='left')
train_df = train_df.merge(terra_exp_train, on=join_keys, how='left')
train_df = train_df.merge(elevation, on=loc_keys, how='left')
train_df = train_df.merge(soil, on=loc_keys, how='left')

val_df = submission_template[['Latitude', 'Longitude', 'Sample Date']].copy()
val_df = val_df.merge(landsat_val, on=join_keys, how='left')
val_df = val_df.merge(terra_exp_val, on=join_keys, how='left')
val_df = val_df.merge(elevation, on=loc_keys, how='left')
val_df = val_df.merge(soil, on=loc_keys, how='left')

print(f"  Train: {train_df.shape}, Val: {val_df.shape}")

# ============================================================
# [2] V2 FEATURE ENGINEERING (reuse from V2)
# ============================================================
print("\n[2] V2 feature engineering (geography-invariant)...")

def engineer_features_v2(df):
    """
    Create features that are TRANSFERABLE across geographies.
    NO Latitude, Longitude, or coordinate-based interactions.
    Reused from 05_model_v2.py.
    """
    df = df.copy()
    
    # -- Temporal --
    df['parsed_date'] = pd.to_datetime(df['Sample Date'], dayfirst=True, errors='coerce')
    df['month'] = df['parsed_date'].dt.month
    df['year'] = df['parsed_date'].dt.year
    df['day_of_year'] = df['parsed_date'].dt.dayofyear
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    season_map = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
                  6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    df['season'] = df['month'].map(season_map)
    
    eps = 1e-10
    
    # -- Landsat spectral indices --
    if 'nir' in df.columns:
        df['nir_green_ratio'] = df['nir'] / (df['green'] + eps)
        df['pseudo_ndvi'] = (df['nir'] - df['green']) / (df['nir'] + df['green'] + eps)
        df['brightness'] = (df['nir'] + df['green'] + df.get('swir16', 0) + df.get('swir22', 0)) / 4
    
    if 'swir16' in df.columns:
        df['swir_ratio'] = df['swir16'] / (df['swir22'] + eps)
        df['green_swir22_ratio'] = df['green'] / (df['swir22'] + eps)
        df['nir_swir16_ratio'] = df['nir'] / (df['swir16'] + eps)
        df['bsi'] = ((df['swir16'] + df['green']) - (df['nir'] + df['swir22'])) / \
                    ((df['swir16'] + df['green']) + (df['nir'] + df['swir22']) + eps)
    
    # -- Climate-derived --
    if 'tmax' in df.columns and 'tmin' in df.columns:
        df['temp_range'] = df['tmax'] - df['tmin']
        df['temp_mean'] = (df['tmax'] + df['tmin']) / 2
    
    if 'pet' in df.columns and 'ppt' in df.columns:
        df['aridity_index'] = df['pet'] / df['ppt'].replace(0, np.nan)
        df['moisture_surplus'] = df['ppt'] - df['pet']
    
    if 'aet' in df.columns and 'pet' in df.columns:
        df['aet_pet_ratio'] = df['aet'] / (df['pet'] + eps)
    
    if 'ppt' in df.columns and 'ppt_lag1' in df.columns:
        df['ppt_change_1m'] = df['ppt'] - df['ppt_lag1']
    if 'ppt_lag1' in df.columns and 'ppt_lag2' in df.columns:
        df['ppt_change_2m'] = df['ppt_lag1'] - df['ppt_lag2']
    
    if 'q' in df.columns and 'ppt' in df.columns:
        df['runoff_ratio'] = df['q'] / (df['ppt'] + eps)
    
    if 'def' in df.columns and 'ppt' in df.columns:
        df['deficit_ppt_ratio'] = df['def'] / (df['ppt'] + eps)
    
    if 'vpd' in df.columns and 'tmax' in df.columns:
        df['vpd_temp_ratio'] = df['vpd'] / (df['tmax'] + eps)
    
    # -- Soil-derived --
    if 'soil_clay' in df.columns and 'soil_sand' in df.columns:
        df['clay_sand_ratio'] = df['soil_clay'] / (df['soil_sand'] + eps)
        df['clay_fraction'] = df['soil_clay'] / (df['soil_clay'] + df['soil_sand'] + eps)
    
    if 'soil_soc' in df.columns:
        df['soc_clay_ratio'] = df['soil_soc'] / (df['soil_clay'] + eps)
    
    if 'soil_phh2o' in df.columns:
        df['soil_ph_centered'] = df['soil_phh2o'] - 70
    
    # -- Elevation x Climate interactions --
    if 'elevation_m' in df.columns:
        df['elev_x_ppt'] = df['elevation_m'] * df.get('ppt', 0)
        df['elev_x_temp'] = df['elevation_m'] * df.get('temp_mean', df.get('tmax', 0))
        df['elev_x_runoff'] = df['elevation_m'] * df.get('q', 0)
    
    # -- Soil x Climate interactions --
    if 'soil_clay' in df.columns and 'ppt' in df.columns:
        df['clay_x_ppt'] = df['soil_clay'] * df['ppt']
    if 'soil_soc' in df.columns and 'ppt' in df.columns:
        df['soc_x_ppt'] = df['soil_soc'] * df['ppt']
    
    # -- Spectral x Climate interactions --
    if 'NDMI' in df.columns and 'ppt' in df.columns:
        df['ndmi_x_ppt'] = df['NDMI'] * df['ppt']
    if 'MNDWI' in df.columns and 'soil' in df.columns:
        df['mndwi_x_soilmoist'] = df['MNDWI'] * df['soil']
    
    # -- Location fingerprint from climate normals --
    if 'ppt' in df.columns and 'ppt_3m_avg' in df.columns:
        df['ppt_vs_avg'] = df['ppt'] / (df['ppt_3m_avg'] + eps)
    if 'tmax' in df.columns and 'tmax_3m_avg' in df.columns:
        df['tmax_vs_avg'] = df['tmax'] / (df['tmax_3m_avg'] + eps)
    
    # -- DROP coordinate columns and date --
    df = df.drop(columns=['parsed_date', 'Sample Date'], errors='ignore')
    # NOTE: We keep Latitude/Longitude here because V3 needs them for KNN.
    # They get dropped from the final feature list later.
    
    return df

train_featured = engineer_features_v2(train_df)
val_featured = engineer_features_v2(val_df)

print(f"  V2 features created: {train_featured.shape[1]} columns (incl. coords & targets)")

# ============================================================
# [3] KNN TARGET ENCODING — the NEW part
# ============================================================
print("\n[3] Building KNN target encoding features...")

train_coords_rad = np.deg2rad(train_df[['Latitude', 'Longitude']].values)
val_coords_rad = np.deg2rad(val_df[['Latitude', 'Longitude']].values)

K_VALUES = [3, 5, 10]
MAX_K = max(K_VALUES)


def compute_knn_features(query_coords_rad, pool_coords_rad, pool_targets_df,
                         k_values, exclude_self_idx=None):
    """
    For each query point, find K nearest neighbors in pool and compute
    target statistics.
    
    Args:
        query_coords_rad: (N, 2) array of query lat/lon in radians
        pool_coords_rad: (M, 2) array of pool lat/lon in radians
        pool_targets_df: DataFrame with target columns for pool points
        k_values: list of K values to compute features for
        exclude_self_idx: if not None, array of indices to exclude 
                          (for leakage prevention). Length N, value is the
                          index in pool to exclude for each query, or -1 if none.
    
    Returns:
        DataFrame with KNN features for each query point
    """
    max_k = max(k_values)
    # BallTree needs (lat, lon) in radians for haversine
    tree = BallTree(pool_coords_rad, metric='haversine')
    
    # Query for max_k + 1 neighbors (in case we need to exclude self)
    n_query = max_k + 1 if exclude_self_idx is not None else max_k
    n_query = min(n_query, len(pool_coords_rad))
    
    distances, indices = tree.query(query_coords_rad, k=n_query)
    # Convert haversine distances to km
    distances_km = distances * EARTH_RADIUS_KM
    
    result = {}
    
    for i in range(len(query_coords_rad)):
        nbr_dists = distances_km[i]
        nbr_idxs = indices[i]
        
        # Exclude self if needed
        if exclude_self_idx is not None and exclude_self_idx[i] >= 0:
            mask = nbr_idxs != exclude_self_idx[i]
            nbr_dists = nbr_dists[mask]
            nbr_idxs = nbr_idxs[mask]
        
        for k in k_values:
            k_dists = nbr_dists[:k]
            k_idxs = nbr_idxs[:k]
            
            for target in TARGETS:
                target_vals = pool_targets_df[target].iloc[k_idxs].values
                valid_mask = ~np.isnan(target_vals)
                
                if valid_mask.sum() == 0:
                    mean_val = np.nan
                    median_val = np.nan
                    std_val = np.nan
                    dwm_val = np.nan
                else:
                    valid_vals = target_vals[valid_mask]
                    valid_dists = k_dists[valid_mask]
                    
                    mean_val = np.mean(valid_vals)
                    median_val = np.median(valid_vals)
                    std_val = np.std(valid_vals) if len(valid_vals) > 1 else 0.0
                    
                    # Inverse-distance weighted mean
                    weights = 1.0 / (valid_dists + 1e-6)  # avoid div by 0
                    dwm_val = np.average(valid_vals, weights=weights)
                
                tname = target.replace(' ', '_')
                key_prefix = f"knn_k{k}_{tname}"
                result.setdefault(f"{key_prefix}_mean", []).append(mean_val)
                result.setdefault(f"{key_prefix}_median", []).append(median_val)
                result.setdefault(f"{key_prefix}_std", []).append(std_val)
                result.setdefault(f"{key_prefix}_dwm", []).append(dwm_val)
    
    return pd.DataFrame(result)


def compute_distance_features(query_coords_rad, pool_coords_rad):
    """
    Compute distance-based features for query points relative to pool.
    """
    tree = BallTree(pool_coords_rad, metric='haversine')
    
    # Distance to nearest training point
    dists, _ = tree.query(query_coords_rad, k=1)
    dist_to_nearest = dists[:, 0] * EARTH_RADIUS_KM
    
    # Distance to centroid
    centroid_rad = pool_coords_rad.mean(axis=0, keepdims=True)
    # Haversine for centroid
    dlat = query_coords_rad[:, 0] - centroid_rad[0, 0]
    dlon = query_coords_rad[:, 1] - centroid_rad[0, 1]
    a = np.sin(dlat/2)**2 + np.cos(query_coords_rad[:, 0]) * np.cos(centroid_rad[0, 0]) * np.sin(dlon/2)**2
    dist_to_centroid = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    
    # Count within radius
    r_50km = 50.0 / EARTH_RADIUS_KM  # in radians
    r_100km = 100.0 / EARTH_RADIUS_KM
    
    n_within_50 = tree.query_radius(query_coords_rad, r=r_50km, count_only=True)
    n_within_100 = tree.query_radius(query_coords_rad, r=r_100km, count_only=True)
    
    return pd.DataFrame({
        'dist_to_nearest_train_km': dist_to_nearest,
        'dist_to_centroid_km': dist_to_centroid,
        'n_train_within_50km': n_within_50,
        'n_train_within_100km': n_within_100,
    })


# --- Compute KNN features for VALIDATION set (uses all training data) ---
print("  Computing KNN features for validation set...")
val_knn = compute_knn_features(val_coords_rad, train_coords_rad,
                                wq_train[TARGETS], K_VALUES)
val_dist = compute_distance_features(val_coords_rad, train_coords_rad)

val_featured = pd.concat([val_featured.reset_index(drop=True), 
                           val_knn.reset_index(drop=True),
                           val_dist.reset_index(drop=True)], axis=1)

# --- Compute KNN features for TRAINING set (exclude self) ---
print("  Computing KNN features for training set (excluding self)...")
# For final model retraining: exclude only the point itself
self_indices = np.arange(len(train_coords_rad))
train_knn = compute_knn_features(train_coords_rad, train_coords_rad,
                                  wq_train[TARGETS], K_VALUES,
                                  exclude_self_idx=self_indices)
train_dist = compute_distance_features(train_coords_rad, train_coords_rad)

train_featured = pd.concat([train_featured.reset_index(drop=True),
                             train_knn.reset_index(drop=True),
                             train_dist.reset_index(drop=True)], axis=1)

knn_feature_names = list(train_knn.columns) + list(train_dist.columns)
print(f"  KNN features created: {len(knn_feature_names)}")
for f in knn_feature_names[:8]:
    print(f"    {f}")
print(f"    ... and {len(knn_feature_names) - 8} more")

# ============================================================
# [4] PREPARE FEATURE MATRIX
# ============================================================
print("\n[4] Preparing final feature matrix...")

# Remove targets, coordinates, and non-feature columns from feature list
drop_cols = TARGETS + ['Latitude', 'Longitude']
feature_cols = [c for c in train_featured.columns if c not in drop_cols]

# Verify no coordinate leakage
coord_leaks = [c for c in feature_cols if c.lower() in ('latitude', 'longitude')]
if coord_leaks:
    print(f"  WARNING: Coordinate columns still present: {coord_leaks}")
    feature_cols = [c for c in feature_cols if c not in coord_leaks]
else:
    print(f"  No raw coordinate leakage")

print(f"  Total features: {len(feature_cols)}")
print(f"    V2 features: {len(feature_cols) - len(knn_feature_names)}")
print(f"    KNN features: {len(knn_feature_names)}")

# Handle inf
train_featured = train_featured.replace([np.inf, -np.inf], np.nan)
val_featured = val_featured.replace([np.inf, -np.inf], np.nan)

# ============================================================
# [5] SPATIAL CV SETUP
# ============================================================
print("\n[5] Spatial cross-validation setup...")

train_coords = train_df[['Latitude', 'Longitude']].values
kmeans = KMeans(n_clusters=10, random_state=RANDOM_STATE, n_init=10)
spatial_groups = kmeans.fit_predict(train_coords)
n_groups = len(np.unique(spatial_groups))
print(f"  Spatial clusters: {n_groups}")

# ============================================================
# [6] KNN-AWARE CV — recompute KNN features per fold
# ============================================================
print("\n[6] Training models with leakage-free KNN features...")

X_train_base = train_featured[feature_cols].copy()
X_val = val_featured[feature_cols].copy()

# XGBoost configs
xgb_configs = {
    'conservative': {
        'n_estimators': 300,
        'max_depth': 3,
        'learning_rate': 0.03,
        'subsample': 0.6,
        'colsample_bytree': 0.5,
        'min_child_weight': 30,
        'reg_alpha': 5.0,
        'reg_lambda': 10.0,
        'gamma': 0.5,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbosity': 0,
    },
    'moderate': {
        'n_estimators': 500,
        'max_depth': 4,
        'learning_rate': 0.02,
        'subsample': 0.7,
        'colsample_bytree': 0.6,
        'min_child_weight': 15,
        'reg_alpha': 2.0,
        'reg_lambda': 5.0,
        'gamma': 0.1,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbosity': 0,
    },
    'moderate_deep': {
        'n_estimators': 800,
        'max_depth': 5,
        'learning_rate': 0.02,
        'subsample': 0.7,
        'colsample_bytree': 0.6,
        'min_child_weight': 10,
        'reg_alpha': 1.5,
        'reg_lambda': 4.0,
        'gamma': 0.05,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbosity': 0,
    },
}

best_models = {}
best_scores = {}

for target in TARGETS:
    print(f"\n  {'=' * 60}")
    print(f"  TARGET: {target}")
    print(f"  {'=' * 60}")
    
    y_raw = train_featured[target].copy()
    valid_mask = y_raw.notna()
    y_valid = y_raw[valid_mask]
    X_base_valid = X_train_base[valid_mask]
    groups_valid = spatial_groups[valid_mask.values]
    coords_valid_rad = train_coords_rad[valid_mask.values]
    targets_valid = wq_train[TARGETS][valid_mask.values].reset_index(drop=True)
    
    best_r2 = -999
    best_config_name = ""
    best_model_type = ""
    best_log = False
    
    n_splits = min(5, n_groups)
    
    # ── A. XGBoost models with proper KNN leakage prevention ──
    for config_name, params in xgb_configs.items():
        for use_log in [False, True]:
            label = f"XGB {config_name} {'+ log' if use_log else '+ raw'}"
            
            if use_log and (y_valid < 0).any():
                continue
            
            y = np.log1p(y_valid) if use_log else y_valid
            
            gkf = GroupKFold(n_splits=n_splits)
            fold_scores = []
            
            for fold_i, (train_idx, val_idx) in enumerate(gkf.split(X_base_valid, y, groups=groups_valid)):
                # --- Recompute KNN features with fold-aware leakage prevention ---
                # Pool = only training fold points
                pool_coords = coords_valid_rad[train_idx]
                pool_targets = targets_valid.iloc[train_idx].reset_index(drop=True)
                
                # Query coords for train fold (exclude self from pool)
                tr_query_coords = coords_valid_rad[train_idx]
                tr_self_idx = np.arange(len(train_idx))  # self index within pool
                tr_knn = compute_knn_features(tr_query_coords, pool_coords,
                                               pool_targets, K_VALUES,
                                               exclude_self_idx=tr_self_idx)
                
                # Query coords for val fold (use full train pool, no exclusion)
                va_query_coords = coords_valid_rad[val_idx]
                va_knn = compute_knn_features(va_query_coords, pool_coords,
                                               pool_targets, K_VALUES)
                
                # Build X with updated KNN features
                X_tr_base = X_base_valid.iloc[train_idx].reset_index(drop=True)
                X_va_base = X_base_valid.iloc[val_idx].reset_index(drop=True)
                
                # Replace KNN columns with fold-aware values
                knn_cols = [c for c in knn_feature_names if c in X_tr_base.columns]
                non_knn_cols = [c for c in X_tr_base.columns if c not in knn_cols]
                
                X_tr = pd.concat([X_tr_base[non_knn_cols].reset_index(drop=True),
                                   tr_knn.reset_index(drop=True)], axis=1)
                X_va = pd.concat([X_va_base[non_knn_cols].reset_index(drop=True),
                                   va_knn.reset_index(drop=True)], axis=1)
                
                y_tr = y.iloc[train_idx].reset_index(drop=True)
                y_va = y.iloc[val_idx].reset_index(drop=True)
                
                model = xgb.XGBRegressor(**params)
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                preds = model.predict(X_va)
                
                if use_log:
                    r2 = r2_score(np.expm1(y_va), np.expm1(preds))
                else:
                    r2 = r2_score(y_va, preds)
                fold_scores.append(r2)
            
            mean_r2 = np.mean(fold_scores)
            std_r2 = np.std(fold_scores)
            print(f"    {label:40s} -> R2 = {mean_r2:.4f} (+/-{std_r2:.4f})")
            
            if mean_r2 > best_r2:
                best_r2 = mean_r2
                best_config_name = label
                best_model_type = 'xgb'
                best_log = use_log
    
    # ── B. Ridge Regression ──
    for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        for use_log in [False, True]:
            label = f"Ridge a={alpha} {'+ log' if use_log else '+ raw'}"
            
            if use_log and (y_valid < 0).any():
                continue
            
            y = np.log1p(y_valid) if use_log else y_valid
            
            gkf = GroupKFold(n_splits=n_splits)
            fold_scores = []
            
            for fold_i, (train_idx, val_idx) in enumerate(gkf.split(X_base_valid, y, groups=groups_valid)):
                pool_coords = coords_valid_rad[train_idx]
                pool_targets = targets_valid.iloc[train_idx].reset_index(drop=True)
                
                tr_query_coords = coords_valid_rad[train_idx]
                tr_self_idx = np.arange(len(train_idx))
                tr_knn = compute_knn_features(tr_query_coords, pool_coords,
                                               pool_targets, K_VALUES,
                                               exclude_self_idx=tr_self_idx)
                
                va_query_coords = coords_valid_rad[val_idx]
                va_knn = compute_knn_features(va_query_coords, pool_coords,
                                               pool_targets, K_VALUES)
                
                X_tr_base = X_base_valid.iloc[train_idx].reset_index(drop=True)
                X_va_base = X_base_valid.iloc[val_idx].reset_index(drop=True)
                
                knn_cols = [c for c in knn_feature_names if c in X_tr_base.columns]
                non_knn_cols = [c for c in X_tr_base.columns if c not in knn_cols]
                
                X_tr = pd.concat([X_tr_base[non_knn_cols].reset_index(drop=True),
                                   tr_knn.reset_index(drop=True)], axis=1)
                X_va = pd.concat([X_va_base[non_knn_cols].reset_index(drop=True),
                                   va_knn.reset_index(drop=True)], axis=1)
                
                y_tr = y.iloc[train_idx].reset_index(drop=True)
                y_va = y.iloc[val_idx].reset_index(drop=True)
                
                # Impute + scale for Ridge
                train_medians = X_tr.median()
                X_tr_imp = X_tr.fillna(train_medians)
                X_va_imp = X_va.fillna(train_medians)
                
                scaler = StandardScaler()
                X_tr_sc = scaler.fit_transform(X_tr_imp)
                X_va_sc = scaler.transform(X_va_imp)
                
                model = Ridge(alpha=alpha, random_state=RANDOM_STATE)
                model.fit(X_tr_sc, y_tr)
                preds = model.predict(X_va_sc)
                
                if use_log:
                    r2 = r2_score(np.expm1(y_va), np.expm1(preds))
                else:
                    r2 = r2_score(y_va, preds)
                fold_scores.append(r2)
            
            mean_r2 = np.mean(fold_scores)
            std_r2 = np.std(fold_scores)
            print(f"    {label:40s} -> R2 = {mean_r2:.4f} (+/-{std_r2:.4f})")
            
            if mean_r2 > best_r2:
                best_r2 = mean_r2
                best_config_name = label
                best_model_type = 'ridge'
                best_log = use_log
    
    # ── Retrain best on all data ──
    print(f"\n    * BEST: {best_config_name} -> R2 = {best_r2:.4f}")
    
    y_full = np.log1p(y_valid) if best_log else y_valid
    
    if best_model_type == 'xgb':
        config_key = best_config_name.split(' ')[1]  # e.g., 'conservative'
        params = xgb_configs[config_key]
        best_model_obj = xgb.XGBRegressor(**params)
        best_model_obj.fit(X_base_valid, y_full)
        
        # Feature importance
        imps = best_model_obj.feature_importances_
        feat_imp = sorted(zip(feature_cols, imps), key=lambda x: x[1], reverse=True)
        print(f"\n    Top 10 features:")
        knn_in_top10 = 0
        for rank, (fn, fi) in enumerate(feat_imp[:10], 1):
            marker = " [KNN]" if fn in knn_feature_names else ""
            bar = chr(9608) * int(fi * 100)
            print(f"      {rank:2d}. {fn:35s} {fi:.4f} {bar}{marker}")
            if fn in knn_feature_names:
                knn_in_top10 += 1
        print(f"    KNN features in top 10: {knn_in_top10}/10")
    
    elif best_model_type == 'ridge':
        alpha = float(best_config_name.split('a=')[1].split(' ')[0])
        
        X_imp = X_base_valid.fillna(X_base_valid.median())
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X_imp)
        
        best_model_obj = Ridge(alpha=alpha, random_state=RANDOM_STATE)
        best_model_obj.fit(X_sc, y_full)
        best_model_obj._v3_scaler = scaler
        best_model_obj._v3_medians = X_base_valid.median()
        
        coefs = np.abs(best_model_obj.coef_)
        feat_imp = sorted(zip(feature_cols, coefs), key=lambda x: x[1], reverse=True)
        print(f"\n    Top 10 features (|coefficient|):")
        knn_in_top10 = 0
        for rank, (fn, fi) in enumerate(feat_imp[:10], 1):
            marker = " [KNN]" if fn in knn_feature_names else ""
            print(f"      {rank:2d}. {fn:35s} {fi:.4f}{marker}")
            if fn in knn_feature_names:
                knn_in_top10 += 1
        print(f"    KNN features in top 10: {knn_in_top10}/10")
    
    best_models[target] = {
        'model': best_model_obj,
        'use_log': best_log,
        'type': best_model_type,
    }
    best_scores[target] = best_r2

# ============================================================
# [7] V3 SPATIAL CV SUMMARY
# ============================================================
print(f"\n{'=' * 70}")
print(f"V3 MODEL - SPATIAL CV RESULTS (KNN Target Encoding + V2 features)")
print(f"{'=' * 70}")
for t, s in best_scores.items():
    status = "+" if s > 0 else "-"
    print(f"  {status} {t:40s}: R2 = {s:.4f}")
v3_mean = np.mean(list(best_scores.values()))
print(f"  {'-' * 55}")
print(f"    {'MEAN R2':40s}: {v3_mean:.4f}")

# ============================================================
# [8] COMPARISON: V1 vs V2 vs V3
# ============================================================
print(f"\n{'=' * 70}")
print(f"COMPARISON: V1 vs V2 vs V3 (Spatial CV Mean R2)")
print(f"{'=' * 70}")
v1_mean = -0.4009
v2_mean = -0.4166  # From V2 run
print(f"  V1 (raw coords + XGB):              {v1_mean:+.4f}")
print(f"  V2 (no coords, invariant features):  {v2_mean:+.4f}")
print(f"  V3 (KNN target encoding + V2):       {v3_mean:+.4f}")
print(f"  {'-' * 55}")
print(f"  V3 vs V1 improvement:                {v3_mean - v1_mean:+.4f}")
print(f"  V3 vs V2 improvement:                {v3_mean - v2_mean:+.4f}")

# ============================================================
# [9] GENERATE SUBMISSIONS
# ============================================================
print(f"\n[9] Generating submissions...")

# --- Pure V3 submission ---
submission = submission_template.copy()
for target in TARGETS:
    m = best_models[target]
    
    if m['type'] == 'xgb':
        preds = m['model'].predict(X_val)
    else:
        X_val_imp = X_val.fillna(m['model']._v3_medians)
        X_val_sc = m['model']._v3_scaler.transform(X_val_imp)
        preds = m['model'].predict(X_val_sc)
    
    if m['use_log']:
        preds = np.expm1(preds)
    
    preds = np.maximum(preds, 0)
    submission[target] = preds
    print(f"  {target}:")
    print(f"    Range: [{preds.min():.2f}, {preds.max():.2f}], Mean: {preds.mean():.2f}")

ts = time.strftime("%Y%m%d_%H%M%S")
submission.to_csv(os.path.join(SUBMISSION_DIR, f'submission_v3_{ts}.csv'), index=False)
submission.to_csv(os.path.join(SUBMISSION_DIR, 'submission_v3.csv'), index=False)
print(f"\n  Saved: submissions/submission_v3.csv")

# --- Ensemble V1 + V2 + V3 ---
print(f"\n  Building ensemble with V1 and V2...")
try:
    v1_sub = pd.read_csv(os.path.join(SUBMISSION_DIR, 'submission_v1.csv'))
    v2_sub = pd.read_csv(os.path.join(SUBMISSION_DIR, 'submission_v2.csv'))
    
    ensemble = submission_template.copy()
    for target in TARGETS:
        ensemble[target] = (0.5 * submission[target] + 
                            0.3 * v1_sub[target] + 
                            0.2 * v2_sub[target])
        ensemble[target] = np.maximum(ensemble[target], 0)
    
    ensemble.to_csv(os.path.join(SUBMISSION_DIR, f'submission_v3_ensemble_{ts}.csv'), index=False)
    ensemble.to_csv(os.path.join(SUBMISSION_DIR, 'submission_v3_ensemble.csv'), index=False)
    print(f"  Saved: submissions/submission_v3_ensemble.csv")
    print(f"  Blend: 0.5*V3 + 0.3*V1 + 0.2*V2")
    
    print(f"\n  Ensemble prediction stats:")
    for target in TARGETS:
        vals = ensemble[target]
        print(f"    {target}: [{vals.min():.2f}, {vals.max():.2f}], Mean: {vals.mean():.2f}")
except FileNotFoundError as e:
    print(f"  WARNING: Could not load V1/V2 submissions for ensemble: {e}")
    print(f"  Skipping ensemble generation.")

# ============================================================
# [10] PREVIEW
# ============================================================
print(f"\n  V3 submission preview:")
print(submission.head(5).to_string())

print(f"""
{'=' * 70}
V3 ANALYSIS
{'=' * 70}
Key innovation: KNN Target Encoding
  - For each point, uses target stats of K nearest training neighbors
  - Gives model "what does water quality look like around here?"
  - No raw coordinates -> no coordinate memorization
  - Leakage-free: fold samples excluded during CV

If V3 CV > V2 CV: KNN encoding captures useful spatial signal
If V3 CV < V2 CV: Location signal may not generalize to new region
Either way: submit both V3 and V3_ensemble to compare on leaderboard
{'=' * 70}
""")

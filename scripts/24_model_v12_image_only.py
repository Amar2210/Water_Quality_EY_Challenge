"""
============================================================
EY Water Quality Challenge - V12 Model
============================================================
SATELLITE IMAGE-ONLY MODEL WITH KNN BACKBONE

Returns to V9's proven KNN-IDW approach but drastically simplifies:
  - ONLY satellite-image-derived features (Landsat spectral,
    SoilGrids maps, SRTM DEM, ESA CCI land cover)
  - No climate/weather, no geology API, no temporal features
  - Minimal feature count per target
  - Sequential target prediction: EC → DRP (with EC stacking) → TA

Architecture per target:
  1. Geographic KNN-IDW (V9/V11 spatial baseline)
  2. Feature-space KNN-IDW (environmental similarity)
  3. Blend geo + feat-space KNN (tune w_geo per target)
  4. ML residual correction (conservative XGBoost)
  5. For DRP: add EC_predicted as stacked feature

Two approaches evaluated per target:
  A: XGBoost/Ridge/ElasticNet on IMAGE_FEATURES + KNN target features
  B: Blended KNN + residual XGBoost on IMAGE_FEATURES only

Best approach per target selected for final submission.
============================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.spatial import cKDTree
from sklearn.model_selection import GroupKFold
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
import xgboost as xgb

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXTRACTED_DIR = os.path.join(BASE_DIR, '..', 'data', 'extracted')
DATA_DIR = os.path.join(BASE_DIR, '..', 'Jupyter Notebook Package')
SUBMISSION_DIR = os.path.join(BASE_DIR, '..', 'submissions')
LOG_DIR = os.path.join(BASE_DIR, '..', 'experiment_logs')
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

TARGETS = ['Electrical Conductance', 'Dissolved Reactive Phosphorus', 'Total Alkalinity']
RANDOM_STATE = 42
N_CLUSTERS = 5
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

print(f"""
{'='*70}
EY WATER QUALITY CHALLENGE - V12 MODEL
Satellite Image-Only with KNN Backbone
{'='*70}
Prediction order: EC → DRP (with EC stacking) → TA
""")

# ============================================================
# [1] LOAD DATA
# ============================================================
print("[1] Loading data...")

train_path = os.path.join(EXTRACTED_DIR, 'v9_train_features.csv')
val_path = os.path.join(EXTRACTED_DIR, 'v9_val_features.csv')

if not os.path.exists(train_path):
    print("  ERROR: V9 features not found! Run 20_feature_engineering_v9.py first")
    sys.exit(1)

train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)

# Load original training data for target values
wq_train = pd.read_csv(os.path.join(DATA_DIR, 'water_quality_training_dataset.csv'))

# Load submission template
submission_template = pd.read_csv(os.path.join(DATA_DIR, 'submission_template.csv'))

print(f"  Train samples: {len(train_df)}")
print(f"  Val samples:   {len(val_df)}")
print(f"  Submission rows: {len(submission_template)}")

# ============================================================
# [2] DEFINE ALL FEATURE SETS
# ============================================================
print("\n[2] Defining feature sets (image-only)...")

# --- SIMILARITY FEATURES (20) — for feature-space KD-tree ---
SIMILARITY_FEATURES = [
    # Terrain (4)
    'elevation_m', 'slope_deg', 'terrain_roughness', 'elev_range_500m',
    # Soil (6)
    'soil_clay', 'soil_sand', 'soil_soc', 'soil_phh2o', 'soil_cec', 'soil_nitrogen',
    # Land cover 5km (10)
    'lc_5km_pct_cropland', 'lc_5km_pct_forest', 'lc_5km_pct_grassland',
    'lc_5km_pct_urban', 'lc_5km_pct_bare', 'lc_5km_pct_water',
    'lc_5km_pct_wetland', 'lc_5km_pct_shrubland',
    'lc_5km_human_impact', 'lc_5km_natural_veg',
]

# --- EC IMAGE FEATURES (38) ---
EC_IMAGE_FEATURES = [
    # Spectral (10)
    'unified_swir16', 'unified_swir22', 'unified_nir', 'unified_green',
    'unified_NDMI', 'unified_MNDWI', 'unified_NDVI', 'ls_st_celsius', 'ls_NSMI', 'ls_NDTI',
    # Ratios (4)
    'ratio_swir16_nir', 'ratio_swir22_swir16', 'ratio_nir_green', 'ratio_nir_swir22',
    # Thermal (3)
    'ls_st_x_swir16', 'ls_st_x_nir', 'ls_st_x_NDMI',
    # Stats (3)
    'ls_spectral_mean_6', 'ls_spectral_std_6', 'ls_total_brightness',
    # Terrain (4)
    'elevation_m', 'slope_deg', 'terrain_roughness', 'elev_range_500m',
    # Soil (6)
    'soil_clay', 'soil_sand', 'soil_soc', 'soil_phh2o', 'soil_cec', 'soil_nitrogen',
    # Land cover (8)
    'lc_5km_pct_cropland', 'lc_5km_pct_urban', 'lc_5km_pct_bare',
    'lc_5km_pct_water', 'lc_5km_human_impact', 'lc_5km_natural_veg',
    'lc_10km_pct_cropland', 'lc_10km_pct_urban',
]

# --- DRP IMAGE FEATURES (35) ---
DRP_IMAGE_FEATURES = [
    # Spectral (10)
    'unified_nir', 'unified_green', 'unified_NDVI', 'unified_NDMI',
    'ls_chlor_ratio', 'ls_NDVI', 'ls_NDWI', 'ls_clarity_index', 'ls_sediment_index', 'ls_WRI',
    # Ratios (3)
    'ratio_nir_green', 'ratio_swir16_nir', 'ratio_swir22_swir16',
    # Stats (3)
    'ls_spectral_mean_6', 'ls_spectral_std_6', 'ls_vis_mean',
    # Thermal (2)
    'ls_st_celsius', 'ls_st_x_nir',
    # Terrain (4)
    'elevation_m', 'slope_deg', 'terrain_roughness', 'elev_range_500m',
    # Soil (5)
    'soil_clay', 'soil_sand', 'soil_soc', 'soil_nitrogen', 'soil_phh2o',
    # Land cover (8)
    'lc_5km_pct_cropland', 'lc_5km_pct_urban', 'lc_5km_pct_water',
    'lc_5km_pct_wetland', 'lc_5km_human_impact', 'lc_5km_natural_veg',
    'lc_10km_pct_cropland', 'lc_10km_pct_forest',
]

# --- TA IMAGE FEATURES (30) ---
TA_IMAGE_FEATURES = [
    # Spectral (7)
    'unified_swir16', 'unified_swir22', 'unified_nir', 'unified_green',
    'unified_NDMI', 'unified_MNDWI', 'ls_st_celsius',
    # Ratios (3)
    'ratio_swir16_nir', 'ratio_swir22_swir16', 'ratio_nir_green',
    # Stats (2)
    'ls_spectral_mean_6', 'ls_total_brightness',
    # Thermal (2)
    'ls_st_x_swir16', 'ls_st_x_NDMI',
    # Terrain (4)
    'elevation_m', 'slope_deg', 'terrain_roughness', 'elev_range_500m',
    # Soil (6)
    'soil_clay', 'soil_sand', 'soil_soc', 'soil_phh2o', 'soil_cec', 'soil_nitrogen',
    # Land cover (6)
    'lc_5km_pct_cropland', 'lc_5km_pct_forest', 'lc_5km_pct_urban',
    'lc_5km_pct_bare', 'lc_5km_human_impact', 'lc_5km_natural_veg',
]

IMAGE_FEATURES_MAP = {
    'Electrical Conductance': EC_IMAGE_FEATURES,
    'Dissolved Reactive Phosphorus': DRP_IMAGE_FEATURES,
    'Total Alkalinity': TA_IMAGE_FEATURES,
}

# Filter to features that actually exist in data
for name, feat_list in [('SIMILARITY', SIMILARITY_FEATURES)]:
    available = [f for f in feat_list if f in train_df.columns]
    missing = [f for f in feat_list if f not in train_df.columns]
    if missing:
        print(f"  WARNING: Missing {name} features: {missing}")
SIMILARITY_FEATURES = [f for f in SIMILARITY_FEATURES if f in train_df.columns]

for target, feat_list in IMAGE_FEATURES_MAP.items():
    available = [f for f in feat_list if f in train_df.columns]
    missing = [f for f in feat_list if f not in train_df.columns]
    short = target.split()[0]
    if missing:
        print(f"  WARNING: Missing {short} features: {missing}")
    IMAGE_FEATURES_MAP[target] = available

print(f"  Similarity features: {len(SIMILARITY_FEATURES)}")
for target in TARGETS:
    print(f"  {target}: {len(IMAGE_FEATURES_MAP[target])} image features")

# ============================================================
# [3] BUILD PER-LOCATION DATABASE
# ============================================================
print("\n[3] Building per-location database...")

# Round coordinates for grouping
train_df['lat_r'] = train_df['Latitude'].round(6)
train_df['lon_r'] = train_df['Longitude'].round(6)
val_df['lat_r'] = val_df['Latitude'].round(6)
val_df['lon_r'] = val_df['Longitude'].round(6)

# Per-location target statistics
wq_train['lat_r'] = wq_train['Latitude'].round(6)
wq_train['lon_r'] = wq_train['Longitude'].round(6)

loc_targets = wq_train.groupby(['lat_r', 'lon_r']).agg({
    'Total Alkalinity': 'mean',
    'Electrical Conductance': 'mean',
    'Dissolved Reactive Phosphorus': 'mean',
}).reset_index()

# Per-location environmental features (median of static features per location)
loc_env = train_df.groupby(['lat_r', 'lon_r'])[SIMILARITY_FEATURES].median().reset_index()

# Merge targets with environmental features
loc_db = loc_env.merge(loc_targets, on=['lat_r', 'lon_r'], how='inner')
print(f"  Training locations: {len(loc_db)}")

# Validation location features
val_loc_env = val_df.groupby(['lat_r', 'lon_r'])[SIMILARITY_FEATURES].median().reset_index()
print(f"  Validation locations: {len(val_loc_env)}")

# Check NaN rates
nan_rates = loc_db[SIMILARITY_FEATURES].isna().mean()
high_nan = nan_rates[nan_rates > 0.5]
if len(high_nan) > 0:
    print(f"  WARNING: High NaN features (>50%):")
    for f, r in high_nan.items():
        print(f"    {f}: {r:.1%}")

# ============================================================
# [4] COORDINATE CONVERSION + HELPER FUNCTIONS
# ============================================================
print("\n[4] Building coordinate systems + helpers...")

MEAN_LAT = loc_db['lat_r'].mean()
KM_PER_LAT = 111.32
KM_PER_LON = 111.32 * np.cos(np.radians(MEAN_LAT))

print(f"  Mean latitude: {MEAN_LAT:.2f}")
print(f"  1 deg lat ~ {KM_PER_LAT:.1f} km, 1 deg lon ~ {KM_PER_LON:.1f} km")


def build_geo_tree(locs):
    """Build KD-tree from geographic coordinates (in km)."""
    coords_km = np.column_stack([
        locs['lat_r'].values * KM_PER_LAT,
        locs['lon_r'].values * KM_PER_LON
    ])
    return cKDTree(coords_km), coords_km


def build_feature_tree(locs, features=None):
    """Build KD-tree from environmental feature space."""
    if features is None:
        features = SIMILARITY_FEATURES
    feat_vals = locs[features].values.copy()
    imp = SimpleImputer(strategy='median')
    feat_vals = imp.fit_transform(feat_vals)
    scaler = StandardScaler()
    feat_scaled = scaler.fit_transform(feat_vals)
    return cKDTree(feat_scaled), feat_scaled, scaler, imp


def idw_predict(tree, tree_data, target_vals, query_point, k=10, power=1.5):
    """IDW prediction: weighted average of K nearest neighbors."""
    max_k = min(k, len(tree_data))
    if max_k == 0:
        return np.nan, np.nan

    dists, idxs = tree.query(query_point, k=max_k)

    if max_k == 1:
        dists = np.array([dists])
        idxs = np.array([idxs])

    valid = ~np.isnan(target_vals[idxs])
    if valid.sum() == 0:
        return np.nan, dists[0] if len(dists) > 0 else np.nan

    v_dists = dists[valid]
    v_vals = target_vals[idxs][valid]
    v_dists = np.maximum(v_dists, 1e-6)

    weights = 1.0 / (v_dists ** power)
    prediction = np.sum(v_vals * weights) / np.sum(weights)
    return prediction, dists[0]


# ============================================================
# [5] SPATIAL CV SETUP
# ============================================================
print("\n[5] Setting up spatial CV (5 clusters, GroupKFold)...")

coords = train_df[['Latitude', 'Longitude']].values
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
train_df['spatial_cluster'] = kmeans.fit_predict(coords)

loc_cluster = train_df.groupby(['lat_r', 'lon_r'])['spatial_cluster'].first().reset_index()
loc_db = loc_db.merge(loc_cluster, on=['lat_r', 'lon_r'], how='left')

for c in range(N_CLUSTERS):
    n_locs = (loc_db['spatial_cluster'] == c).sum()
    n_samples = (train_df['spatial_cluster'] == c).sum()
    print(f"  Cluster {c}: {n_locs} locations, {n_samples} samples")

groups = train_df['spatial_cluster'].values
gkf = GroupKFold(n_splits=N_CLUSTERS)

# ============================================================
# KNN PARAMETERS
# ============================================================
GEO_K = 30
GEO_POWER = {
    'Total Alkalinity': 1.5,
    'Electrical Conductance': 1.5,
    'Dissolved Reactive Phosphorus': 1.0,
}
FEAT_K_VALUES = [5, 7, 10, 15, 20]
FEAT_POWER_VALUES = [1.0, 1.5, 2.0]
BLEND_WEIGHTS = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]

# Per-target configs
TARGET_CONFIG = {
    'Electrical Conductance': {
        'alpha_search': [0.0, 0.1, 0.15, 0.2, 0.3],
        'use_log': False,
        'use_ridge': True,
        'ridge_alpha': 50.0,
    },
    'Dissolved Reactive Phosphorus': {
        'alpha_search': [0.0, 0.1, 0.2, 0.3],
        'use_log': True,
        'use_ridge': False,
        'ridge_alpha': 50.0,
    },
    'Total Alkalinity': {
        'alpha_search': [0.0, 0.05, 0.1, 0.15],
        'use_log': False,
        'use_ridge': True,
        'ridge_alpha': 50.0,
    },
}

# ============================================================
# HELPER: Get KNN target columns for a target
# ============================================================
def get_knn_target_cols(target):
    """Get pre-computed KNN target features from V9 data."""
    short = target.replace(' ', '_')
    cols = [c for c in train_df.columns
            if (short in c and (c.startswith('knn') or c.startswith('spatial_anomaly')))
            or c.startswith('dist_') or c.startswith('log_dist') or c.startswith('knn_confidence')]
    return [c for c in cols if c in train_df.columns and c in val_df.columns]


# ============================================================
# HELPER: Run blended KNN CV for a target
# ============================================================
def run_blended_knn_cv(target, loc_db_fold_func=None):
    """
    Run full blended KNN pipeline for a target:
      1. Tune feature-space KNN (K, power)
      2. Geographic KNN baseline
      3. Blend geo + feat-space
    Returns best config + OOF predictions.
    """
    y_all = train_df[target].values

    # ── Step A: Tune feature-space KNN ──
    print(f"\n    [A] Tuning feature-space KNN...")
    best_feat_r2 = -999
    best_feat_k = 10
    best_feat_p = 1.5

    for feat_k in FEAT_K_VALUES:
        for feat_p in FEAT_POWER_VALUES:
            fold_scores = []
            for fold, (tr_idx, te_idx) in enumerate(gkf.split(
                    np.arange(len(train_df)), y_all, groups)):
                te_clusters = set(groups[te_idx])
                fold_train_locs = loc_db[~loc_db['spatial_cluster'].isin(te_clusters)].reset_index(drop=True)
                fold_test_locs = loc_db[loc_db['spatial_cluster'].isin(te_clusters)].reset_index(drop=True)
                if len(fold_train_locs) == 0 or len(fold_test_locs) == 0:
                    continue

                feat_tree, feat_data, scaler, imp = build_feature_tree(fold_train_locs)
                train_target_vals = fold_train_locs[target].values

                test_feat_raw = fold_test_locs[SIMILARITY_FEATURES].values.copy()
                test_feat_imp = imp.transform(test_feat_raw)
                test_feat_scaled = scaler.transform(test_feat_imp)

                loc_preds = {}
                for i in range(len(fold_test_locs)):
                    pred, _ = idw_predict(feat_tree, feat_data, train_target_vals,
                                          test_feat_scaled[i], k=feat_k, power=feat_p)
                    loc_preds[(fold_test_locs.iloc[i]['lat_r'],
                               fold_test_locs.iloc[i]['lon_r'])] = pred

                te_preds, te_actuals = [], []
                for idx in te_idx:
                    if np.isnan(y_all[idx]):
                        continue
                    key = (train_df.iloc[idx]['lat_r'], train_df.iloc[idx]['lon_r'])
                    if key in loc_preds and not np.isnan(loc_preds[key]):
                        te_preds.append(loc_preds[key])
                        te_actuals.append(y_all[idx])
                if len(te_preds) > 1:
                    fold_scores.append(r2_score(te_actuals, te_preds))

            if fold_scores:
                mean_r2 = np.mean(fold_scores)
                if mean_r2 > best_feat_r2:
                    best_feat_r2 = mean_r2
                    best_feat_k = feat_k
                    best_feat_p = feat_p

    print(f"        Best: K={best_feat_k}, p={best_feat_p}, R2={best_feat_r2:+.4f}")

    # ── Step B: Geographic KNN ──
    print(f"    [B] Geographic KNN (K={GEO_K}, p={GEO_POWER[target]})...")
    geo_fold_scores = []
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(
            np.arange(len(train_df)), y_all, groups)):
        te_clusters = set(groups[te_idx])
        fold_train_locs = loc_db[~loc_db['spatial_cluster'].isin(te_clusters)].reset_index(drop=True)
        fold_test_locs = loc_db[loc_db['spatial_cluster'].isin(te_clusters)].reset_index(drop=True)
        if len(fold_train_locs) == 0 or len(fold_test_locs) == 0:
            continue

        geo_tree, geo_data = build_geo_tree(fold_train_locs)
        train_target_vals = fold_train_locs[target].values

        loc_preds = {}
        for i in range(len(fold_test_locs)):
            q_km = np.array([fold_test_locs.iloc[i]['lat_r'] * KM_PER_LAT,
                             fold_test_locs.iloc[i]['lon_r'] * KM_PER_LON])
            pred, _ = idw_predict(geo_tree, geo_data, train_target_vals,
                                  q_km, k=GEO_K, power=GEO_POWER[target])
            loc_preds[(fold_test_locs.iloc[i]['lat_r'],
                       fold_test_locs.iloc[i]['lon_r'])] = pred

        te_preds, te_actuals = [], []
        for idx in te_idx:
            if np.isnan(y_all[idx]):
                continue
            key = (train_df.iloc[idx]['lat_r'], train_df.iloc[idx]['lon_r'])
            if key in loc_preds and not np.isnan(loc_preds[key]):
                te_preds.append(loc_preds[key])
                te_actuals.append(y_all[idx])
        if len(te_preds) > 1:
            geo_fold_scores.append(r2_score(te_actuals, te_preds))

    geo_mean_r2 = np.mean(geo_fold_scores) if geo_fold_scores else -999
    print(f"        Geo KNN R2={geo_mean_r2:+.4f}")

    # ── Step C: Blend ──
    print(f"    [C] Blending geo + feature-space KNN...")
    best_blend_r2 = -999
    best_blend_w = 0.0
    best_oof_preds = None

    for w_geo in BLEND_WEIGHTS:
        fold_scores = []
        oof_preds = np.full(len(train_df), np.nan)

        for fold, (tr_idx, te_idx) in enumerate(gkf.split(
                np.arange(len(train_df)), y_all, groups)):
            te_clusters = set(groups[te_idx])
            fold_train_locs = loc_db[~loc_db['spatial_cluster'].isin(te_clusters)].reset_index(drop=True)
            fold_test_locs = loc_db[loc_db['spatial_cluster'].isin(te_clusters)].reset_index(drop=True)
            if len(fold_train_locs) == 0 or len(fold_test_locs) == 0:
                continue

            train_target_vals = fold_train_locs[target].values
            geo_tree, geo_data = build_geo_tree(fold_train_locs)
            feat_tree, feat_data, scaler, imp = build_feature_tree(fold_train_locs)

            test_feat_raw = fold_test_locs[SIMILARITY_FEATURES].values.copy()
            test_feat_imp = imp.transform(test_feat_raw)
            test_feat_scaled = scaler.transform(test_feat_imp)

            loc_preds = {}
            for i in range(len(fold_test_locs)):
                q_km = np.array([fold_test_locs.iloc[i]['lat_r'] * KM_PER_LAT,
                                 fold_test_locs.iloc[i]['lon_r'] * KM_PER_LON])
                geo_pred, _ = idw_predict(geo_tree, geo_data, train_target_vals,
                                          q_km, k=GEO_K, power=GEO_POWER[target])
                feat_pred, _ = idw_predict(feat_tree, feat_data, train_target_vals,
                                           test_feat_scaled[i], k=best_feat_k, power=best_feat_p)

                if np.isnan(geo_pred) and np.isnan(feat_pred):
                    blended = np.nan
                elif np.isnan(geo_pred):
                    blended = feat_pred
                elif np.isnan(feat_pred):
                    blended = geo_pred
                else:
                    blended = w_geo * geo_pred + (1 - w_geo) * feat_pred

                loc_preds[(fold_test_locs.iloc[i]['lat_r'],
                           fold_test_locs.iloc[i]['lon_r'])] = blended

            te_preds_list, te_actuals_list = [], []
            for idx in te_idx:
                key = (train_df.iloc[idx]['lat_r'], train_df.iloc[idx]['lon_r'])
                if key in loc_preds and not np.isnan(loc_preds[key]):
                    oof_preds[idx] = loc_preds[key]
                    if not np.isnan(y_all[idx]):
                        te_preds_list.append(loc_preds[key])
                        te_actuals_list.append(y_all[idx])
            if len(te_preds_list) > 1:
                fold_scores.append(r2_score(te_actuals_list, te_preds_list))

        if fold_scores:
            mean_r2 = np.mean(fold_scores)
            marker = '+' if mean_r2 > 0 else '-'
            print(f"        {marker} w_geo={w_geo:.1f}: R2={mean_r2:+.4f}  "
                  f"[{', '.join(f'{s:+.3f}' for s in fold_scores)}]")
            if mean_r2 > best_blend_r2:
                best_blend_r2 = mean_r2
                best_blend_w = w_geo
                best_oof_preds = oof_preds.copy()

    print(f"        Best blend: w_geo={best_blend_w:.1f}, R2={best_blend_r2:+.4f}")

    return {
        'feat_k': best_feat_k,
        'feat_p': best_feat_p,
        'feat_r2': best_feat_r2,
        'geo_r2': geo_mean_r2,
        'blend_w': best_blend_w,
        'blend_r2': best_blend_r2,
        'oof_preds': best_oof_preds,
    }


# ============================================================
# HELPER: Generate final KNN predictions for validation
# ============================================================
def generate_knn_val_predictions(target, feat_k, feat_p, blend_w):
    """Generate blended KNN predictions for validation locations."""
    geo_tree, geo_data = build_geo_tree(loc_db)
    feat_tree, feat_data, scaler, imp = build_feature_tree(loc_db)
    train_target_vals = loc_db[target].values

    val_feat_raw = val_loc_env[SIMILARITY_FEATURES].values.copy()
    val_feat_imp = imp.transform(val_feat_raw)
    val_feat_scaled = scaler.transform(val_feat_imp)

    val_loc_preds = {}
    for i in range(len(val_loc_env)):
        lat_r = val_loc_env.iloc[i]['lat_r']
        lon_r = val_loc_env.iloc[i]['lon_r']

        q_km = np.array([lat_r * KM_PER_LAT, lon_r * KM_PER_LON])
        geo_pred, _ = idw_predict(geo_tree, geo_data, train_target_vals,
                                  q_km, k=GEO_K, power=GEO_POWER[target])
        feat_pred, _ = idw_predict(feat_tree, feat_data, train_target_vals,
                                   val_feat_scaled[i], k=feat_k, power=feat_p)

        if np.isnan(geo_pred) and np.isnan(feat_pred):
            blended = np.nan
        elif np.isnan(geo_pred):
            blended = feat_pred
        elif np.isnan(feat_pred):
            blended = geo_pred
        else:
            blended = blend_w * geo_pred + (1 - blend_w) * feat_pred

        val_loc_preds[(lat_r, lon_r)] = blended

    val_preds = np.full(len(val_df), np.nan)
    for i in range(len(val_df)):
        key = (val_df.iloc[i]['lat_r'], val_df.iloc[i]['lon_r'])
        if key in val_loc_preds:
            val_preds[i] = val_loc_preds[key]

    return val_preds


# ============================================================
# HELPER: Run Approach A (pure ML with KNN features)
# ============================================================
def run_approach_a(target, image_features, extra_features=None, use_log=False,
                   use_ridge=False, ridge_alpha=50.0):
    """
    Approach A: XGBoost/Ridge/ElasticNet on image features + KNN target features.
    Returns best R2, config name, OOF predictions, val predictions.
    """
    knn_cols = get_knn_target_cols(target)
    all_feat = list(image_features) + knn_cols
    if extra_features:
        all_feat = all_feat + [f for f in extra_features if f not in all_feat]
    all_feat = [c for c in all_feat if c in train_df.columns and c in val_df.columns]

    y = train_df[target].values
    valid_mask = ~np.isnan(y)
    y_valid = y[valid_mask]
    groups_valid = groups[valid_mask]

    if use_log:
        y_transformed = np.log1p(np.clip(y_valid, 0, None))
    else:
        y_transformed = y_valid

    X = train_df[all_feat].values[valid_mask]
    X_val = val_df[all_feat].values

    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(X)
    X_val_imp = imputer.transform(X_val)

    # Scale for Ridge/ElasticNet
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)
    X_val_scaled = scaler.transform(X_val_imp)

    model_configs = [
        ('XGB_cons', lambda: xgb.XGBRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.03,
            subsample=0.65, colsample_bytree=0.5,
            reg_alpha=5.0, reg_lambda=20.0, min_child_weight=20, gamma=0.5,
            tree_method='hist', random_state=RANDOM_STATE, verbosity=0,
        ), False),
    ]
    if use_ridge:
        model_configs.append(
            ('Ridge', lambda: Ridge(alpha=ridge_alpha), True)
        )
        model_configs.append(
            ('ElasticNet', lambda: ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000), True)
        )

    best_r2 = -999
    best_config = None
    best_oof = np.full(len(train_df), np.nan)
    best_val_pred = None

    for model_name, model_factory, needs_scaling in model_configs:
        cv_scores = []
        val_preds_fold = []
        oof = np.full(len(y_valid), np.nan)

        X_use = X_scaled if needs_scaling else X_imp
        X_val_use = X_val_scaled if needs_scaling else X_val_imp

        for fold, (tr_idx, te_idx) in enumerate(gkf.split(X_use, y_transformed, groups_valid)):
            model = model_factory()
            model.fit(X_use[tr_idx], y_transformed[tr_idx])
            fold_pred = model.predict(X_use[te_idx])

            if use_log:
                fold_pred_orig = np.expm1(fold_pred)
                fold_actual = y_valid[te_idx]
            else:
                fold_pred_orig = fold_pred
                fold_actual = y_valid[te_idx]

            oof[te_idx] = fold_pred_orig
            cv_scores.append(r2_score(fold_actual, fold_pred_orig))

            val_fold_pred = model.predict(X_val_use)
            if use_log:
                val_fold_pred = np.expm1(val_fold_pred)
            val_preds_fold.append(val_fold_pred)

        mean_r2 = np.mean(cv_scores)
        marker = '+' if mean_r2 > 0 else '-'
        print(f"        {marker} {model_name:15s} ({len(all_feat)} feats): R2={mean_r2:+.4f}  "
              f"[{', '.join(f'{s:+.3f}' for s in cv_scores)}]")

        if mean_r2 > best_r2:
            best_r2 = mean_r2
            best_config = model_name
            # Map OOF back to full train_df indices
            full_oof = np.full(len(train_df), np.nan)
            valid_indices = np.where(valid_mask)[0]
            for i, vi in enumerate(valid_indices):
                full_oof[vi] = oof[i]
            best_oof = full_oof
            best_val_pred = np.mean(val_preds_fold, axis=0)

    return best_r2, best_config, best_oof, best_val_pred


# ============================================================
# HELPER: Run Approach B (blended KNN + residual correction)
# ============================================================
def run_approach_b(target, image_features, knn_result, extra_features=None, use_log=False):
    """
    Approach B: Blended KNN + residual XGBoost on image-only features.
    Returns best R2, best alpha, OOF predictions, val predictions.
    """
    config = TARGET_CONFIG[target]
    alpha_search = config['alpha_search']
    y_all = train_df[target].values

    blend_oof = knn_result['oof_preds']
    feat_k = knn_result['feat_k']
    feat_p = knn_result['feat_p']
    blend_w = knn_result['blend_w']

    # ML features = image features only (+ EC_predicted if extra_features provided)
    ml_feats = list(image_features)
    if extra_features:
        ml_feats = ml_feats + [f for f in extra_features if f not in ml_feats]
    ml_feats = [c for c in ml_feats if c in train_df.columns]

    best_b_r2 = knn_result['blend_r2']  # alpha=0 baseline
    best_b_alpha = 0.0
    best_b_oof = blend_oof.copy() if blend_oof is not None else None
    best_b_val_pred = generate_knn_val_predictions(target, feat_k, feat_p, blend_w)

    for alpha in alpha_search:
        if alpha == 0.0:
            print(f"        + alpha=0.0 (pure KNN blend): R2={knn_result['blend_r2']:+.4f}")
            continue

        fold_scores = []
        oof_corrected = blend_oof.copy() if blend_oof is not None else np.full(len(train_df), np.nan)

        for fold, (tr_idx, te_idx) in enumerate(gkf.split(
                np.arange(len(train_df)), y_all, groups)):
            te_clusters = set(groups[te_idx])
            fold_train_locs = loc_db[~loc_db['spatial_cluster'].isin(te_clusters)].reset_index(drop=True)

            # Build LOO blend predictions for training samples in this fold
            train_target_vals = fold_train_locs[target].values
            geo_tree, geo_data = build_geo_tree(fold_train_locs)
            feat_tree, feat_data, scaler, imp = build_feature_tree(fold_train_locs)

            tr_blend_preds = np.full(len(tr_idx), np.nan)
            for j, idx in enumerate(tr_idx):
                key = (train_df.iloc[idx]['lat_r'], train_df.iloc[idx]['lon_r'])
                loc_mask = (fold_train_locs['lat_r'] == key[0]) & (fold_train_locs['lon_r'] == key[1])
                if loc_mask.sum() == 0:
                    continue
                loc_idx = loc_mask.values.argmax()

                temp_locs = fold_train_locs.drop(loc_idx).reset_index(drop=True)
                if len(temp_locs) == 0:
                    continue

                temp_geo_tree, temp_geo_data = build_geo_tree(temp_locs)
                temp_feat_tree, temp_feat_data, temp_scaler, temp_imp = build_feature_tree(temp_locs)
                temp_targets = temp_locs[target].values

                q_km = np.array([key[0] * KM_PER_LAT, key[1] * KM_PER_LON])
                g_pred, _ = idw_predict(temp_geo_tree, temp_geo_data, temp_targets,
                                        q_km, k=GEO_K, power=GEO_POWER[target])
                loc_feat = imp.transform(fold_train_locs.iloc[[loc_idx]][SIMILARITY_FEATURES].values)
                loc_feat_s = scaler.transform(loc_feat)
                f_pred, _ = idw_predict(temp_feat_tree, temp_feat_data, temp_targets,
                                        loc_feat_s[0], k=feat_k, power=feat_p)

                if not np.isnan(g_pred) and not np.isnan(f_pred):
                    tr_blend_preds[j] = blend_w * g_pred + (1 - blend_w) * f_pred
                elif not np.isnan(f_pred):
                    tr_blend_preds[j] = f_pred
                elif not np.isnan(g_pred):
                    tr_blend_preds[j] = g_pred

            tr_actuals = y_all[tr_idx]
            tr_residuals = tr_actuals - tr_blend_preds
            tr_valid = ~np.isnan(tr_residuals) & ~np.isnan(tr_actuals)
            if tr_valid.sum() < 20:
                continue

            X_tr = train_df.iloc[tr_idx][ml_feats].values[tr_valid]
            y_tr_res = tr_residuals[tr_valid]

            if use_log:
                # For DRP with log transform: transform residuals relative to blend
                # Actually residual correction in original space is fine
                pass

            ml_imp = SimpleImputer(strategy='median')
            X_tr_imp = ml_imp.fit_transform(X_tr)

            model = xgb.XGBRegressor(
                n_estimators=200, max_depth=3, learning_rate=0.03,
                subsample=0.7, colsample_bytree=0.6,
                reg_alpha=5.0, reg_lambda=20.0, min_child_weight=15,
                tree_method='hist', random_state=RANDOM_STATE, verbosity=0,
            )
            model.fit(X_tr_imp, y_tr_res)

            X_te = train_df.iloc[te_idx][ml_feats].values
            X_te_imp = ml_imp.transform(X_te)
            te_residual_pred = model.predict(X_te_imp)

            te_preds, te_actuals_list = [], []
            for j, idx in enumerate(te_idx):
                if np.isnan(y_all[idx]) or np.isnan(blend_oof[idx]):
                    continue
                final = blend_oof[idx] + alpha * te_residual_pred[j]
                oof_corrected[idx] = final
                te_preds.append(final)
                te_actuals_list.append(y_all[idx])

            if len(te_preds) > 1:
                fold_scores.append(r2_score(te_actuals_list, te_preds))

        if fold_scores:
            mean_r2 = np.mean(fold_scores)
            marker = '+' if mean_r2 > 0 else '-'
            print(f"        {marker} alpha={alpha}: R2={mean_r2:+.4f}  "
                  f"[{', '.join(f'{s:+.3f}' for s in fold_scores)}]")

            if mean_r2 > best_b_r2:
                best_b_r2 = mean_r2
                best_b_alpha = alpha
                best_b_oof = oof_corrected.copy()

    # Generate validation predictions for best alpha
    if best_b_alpha > 0:
        val_knn_preds = generate_knn_val_predictions(target, feat_k, feat_p, blend_w)

        # Train residual model on all training data (LOO blend)
        tr_blend_all = np.full(len(train_df), np.nan)
        for i in range(len(loc_db)):
            lat_r = loc_db.iloc[i]['lat_r']
            lon_r = loc_db.iloc[i]['lon_r']

            temp_locs = loc_db.drop(i).reset_index(drop=True)
            if len(temp_locs) == 0:
                continue
            temp_geo_tree, temp_geo_data = build_geo_tree(temp_locs)
            temp_feat_tree, temp_feat_data, temp_scaler, temp_imp = build_feature_tree(temp_locs)
            temp_targets = temp_locs[target].values

            q_km = np.array([lat_r * KM_PER_LAT, lon_r * KM_PER_LON])
            g_pred, _ = idw_predict(temp_geo_tree, temp_geo_data, temp_targets,
                                    q_km, k=GEO_K, power=GEO_POWER[target])

            # Use main imputer/scaler for consistency
            main_feat_tree, main_feat_data, main_scaler, main_imp = build_feature_tree(loc_db)
            loc_feat = main_imp.transform(loc_db.iloc[[i]][SIMILARITY_FEATURES].values)
            loc_feat_s = main_scaler.transform(loc_feat)
            f_pred, _ = idw_predict(temp_feat_tree, temp_feat_data, temp_targets,
                                    loc_feat_s[0], k=feat_k, power=feat_p)

            if not np.isnan(g_pred) and not np.isnan(f_pred):
                loc_blend = blend_w * g_pred + (1 - blend_w) * f_pred
            elif not np.isnan(f_pred):
                loc_blend = f_pred
            elif not np.isnan(g_pred):
                loc_blend = g_pred
            else:
                loc_blend = np.nan

            mask = (train_df['lat_r'] == lat_r) & (train_df['lon_r'] == lon_r)
            tr_blend_all[mask.values] = loc_blend

        tr_residuals = y_all - tr_blend_all
        tr_valid = ~np.isnan(tr_residuals) & ~np.isnan(y_all)

        X_tr = train_df[ml_feats].values[tr_valid]
        y_tr_res = tr_residuals[tr_valid]

        ml_imp = SimpleImputer(strategy='median')
        X_tr_imp = ml_imp.fit_transform(X_tr)

        model = xgb.XGBRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.6,
            reg_alpha=5.0, reg_lambda=20.0, min_child_weight=15,
            tree_method='hist', random_state=RANDOM_STATE, verbosity=0,
        )
        model.fit(X_tr_imp, y_tr_res)

        X_val = val_df[ml_feats].values
        X_val_imp = ml_imp.transform(X_val)
        val_residual = model.predict(X_val_imp)
        best_b_val_pred = val_knn_preds + best_b_alpha * val_residual
    else:
        best_b_val_pred = generate_knn_val_predictions(target, feat_k, feat_p, blend_w)

    return best_b_r2, best_b_alpha, best_b_oof, best_b_val_pred


# ============================================================
# [6] EC MODEL (predicted first)
# ============================================================
print("\n" + "=" * 70)
print("[6] EC MODEL (Electrical Conductance) - predicted first")
print("=" * 70)

target_ec = 'Electrical Conductance'
ec_config = TARGET_CONFIG[target_ec]

# Run blended KNN
print(f"\n  Blended KNN pipeline:")
ec_knn = run_blended_knn_cv(target_ec)

# Approach A: XGB + Ridge + ElasticNet on EC_IMAGE + KNN target features
print(f"\n  Approach A (ML on image + KNN features):")
ec_a_r2, ec_a_config, ec_a_oof, ec_a_val = run_approach_a(
    target_ec, IMAGE_FEATURES_MAP[target_ec],
    use_log=ec_config['use_log'],
    use_ridge=ec_config['use_ridge'],
    ridge_alpha=ec_config['ridge_alpha'],
)

# Approach B: Blended KNN + residual XGBoost on image features only
print(f"\n  Approach B (KNN blend + residual correction):")
ec_b_r2, ec_b_alpha, ec_b_oof, ec_b_val = run_approach_b(
    target_ec, IMAGE_FEATURES_MAP[target_ec], ec_knn,
    use_log=ec_config['use_log'],
)

print(f"\n  EC Summary:")
print(f"    Approach A ({ec_a_config}):  R2={ec_a_r2:+.4f}")
print(f"    Approach B (alpha={ec_b_alpha}): R2={ec_b_r2:+.4f}")

# Select best and generate OOF EC predictions for DRP stacking
if ec_a_r2 > ec_b_r2:
    ec_best_r2 = ec_a_r2
    ec_best_approach = f'A_{ec_a_config}'
    ec_oof_for_stacking = ec_a_oof
    ec_val_preds = ec_a_val
    print(f"    >> Using Approach A")
else:
    ec_best_r2 = ec_b_r2
    ec_best_approach = f'B_alpha{ec_b_alpha}'
    ec_oof_for_stacking = ec_b_oof
    ec_val_preds = ec_b_val
    print(f"    >> Using Approach B")

# Store EC OOF predictions for DRP stacking
train_df['EC_predicted'] = ec_oof_for_stacking
val_df['EC_predicted'] = ec_val_preds


# ============================================================
# [7] DRP MODEL (with EC_predicted stacked)
# ============================================================
print("\n" + "=" * 70)
print("[7] DRP MODEL (Dissolved Reactive Phosphorus) - with EC stacking")
print("=" * 70)

target_drp = 'Dissolved Reactive Phosphorus'
drp_config = TARGET_CONFIG[target_drp]

# DRP features + EC_predicted
drp_extra_features = ['EC_predicted']

# Run blended KNN (same as EC, no EC_predicted in KNN)
print(f"\n  Blended KNN pipeline:")
drp_knn = run_blended_knn_cv(target_drp)

# Approach A: XGB on DRP_IMAGE + EC_predicted + KNN target features
print(f"\n  Approach A (ML on image + EC_predicted + KNN features):")
drp_a_r2, drp_a_config, drp_a_oof, drp_a_val = run_approach_a(
    target_drp, IMAGE_FEATURES_MAP[target_drp],
    extra_features=drp_extra_features,
    use_log=drp_config['use_log'],
    use_ridge=drp_config['use_ridge'],
    ridge_alpha=drp_config['ridge_alpha'],
)

# Approach B: Blended KNN + residual XGBoost on DRP_IMAGE + EC_predicted
print(f"\n  Approach B (KNN blend + residual correction with EC stacking):")
drp_b_r2, drp_b_alpha, drp_b_oof, drp_b_val = run_approach_b(
    target_drp, IMAGE_FEATURES_MAP[target_drp], drp_knn,
    extra_features=drp_extra_features,
    use_log=drp_config['use_log'],
)

print(f"\n  DRP Summary:")
print(f"    Approach A ({drp_a_config}):  R2={drp_a_r2:+.4f}")
print(f"    Approach B (alpha={drp_b_alpha}): R2={drp_b_r2:+.4f}")

if drp_a_r2 > drp_b_r2:
    drp_best_r2 = drp_a_r2
    drp_best_approach = f'A_{drp_a_config}'
    drp_val_preds = drp_a_val
    print(f"    >> Using Approach A")
else:
    drp_best_r2 = drp_b_r2
    drp_best_approach = f'B_alpha{drp_b_alpha}'
    drp_val_preds = drp_b_val
    print(f"    >> Using Approach B")


# ============================================================
# [8] TA MODEL (last)
# ============================================================
print("\n" + "=" * 70)
print("[8] TA MODEL (Total Alkalinity) - last")
print("=" * 70)

target_ta = 'Total Alkalinity'
ta_config = TARGET_CONFIG[target_ta]

# Run blended KNN
print(f"\n  Blended KNN pipeline:")
ta_knn = run_blended_knn_cv(target_ta)

# Approach A: XGB + Ridge + ElasticNet on TA_IMAGE + KNN target features
print(f"\n  Approach A (ML on image + KNN features):")
ta_a_r2, ta_a_config, ta_a_oof, ta_a_val = run_approach_a(
    target_ta, IMAGE_FEATURES_MAP[target_ta],
    use_log=ta_config['use_log'],
    use_ridge=ta_config['use_ridge'],
    ridge_alpha=ta_config['ridge_alpha'],
)

# Approach B: Blended KNN + residual XGBoost on TA_IMAGE only
print(f"\n  Approach B (KNN blend + residual correction):")
ta_b_r2, ta_b_alpha, ta_b_oof, ta_b_val = run_approach_b(
    target_ta, IMAGE_FEATURES_MAP[target_ta], ta_knn,
    use_log=ta_config['use_log'],
)

print(f"\n  TA Summary:")
print(f"    Approach A ({ta_a_config}):  R2={ta_a_r2:+.4f}")
print(f"    Approach B (alpha={ta_b_alpha}): R2={ta_b_r2:+.4f}")

if ta_a_r2 > ta_b_r2:
    ta_best_r2 = ta_a_r2
    ta_best_approach = f'A_{ta_a_config}'
    ta_val_preds = ta_a_val
    print(f"    >> Using Approach A")
else:
    ta_best_r2 = ta_b_r2
    ta_best_approach = f'B_alpha{ta_b_alpha}'
    ta_val_preds = ta_b_val
    print(f"    >> Using Approach B")


# ============================================================
# [9] FINAL ENSEMBLE + SUBMISSION
# ============================================================
print("\n" + "=" * 70)
print("[9] FINAL PREDICTIONS + SUBMISSION")
print("=" * 70)

final_predictions = {
    'Electrical Conductance': ec_val_preds,
    'Dissolved Reactive Phosphorus': drp_val_preds,
    'Total Alkalinity': ta_val_preds,
}

# Physical clipping
clip_ranges = {
    'Electrical Conductance': (0, 10000),
    'Dissolved Reactive Phosphorus': (0, 5000),
    'Total Alkalinity': (0, 1000),
}

for target in ['Electrical Conductance', 'Dissolved Reactive Phosphorus', 'Total Alkalinity']:
    preds = final_predictions[target]

    # Fill NaN with global training mean
    global_mean = loc_db[target].mean()
    nan_count = np.isnan(preds).sum()
    if nan_count > 0:
        print(f"  {target}: Filling {nan_count} NaN with mean {global_mean:.1f}")
        preds[np.isnan(preds)] = global_mean

    # Clip
    lo, hi = clip_ranges[target]
    preds = np.clip(preds, lo, hi)
    final_predictions[target] = preds

    print(f"  {target}: range=[{preds.min():.1f}, {preds.max():.1f}], "
          f"mean={preds.mean():.1f}, std={preds.std():.1f}")

# Build submission
submission = submission_template.copy()
for target in ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']:
    submission[target] = final_predictions[target]

sub_path = os.path.join(SUBMISSION_DIR, 'submission_v12.csv')
sub_ts_path = os.path.join(SUBMISSION_DIR, f'submission_v12_{ts}.csv')
submission.to_csv(sub_path, index=False)
submission.to_csv(sub_ts_path, index=False)

print(f"\n  Saved: {sub_path}")
print(f"  Saved: {sub_ts_path}")

# NaN check
for target in ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']:
    n_nan = submission[target].isna().sum()
    if n_nan > 0:
        print(f"  WARNING: {target} has {n_nan} NaN values!")
    else:
        print(f"  {target}: 0 NaN values - OK")


# ============================================================
# [10] RESULTS SUMMARY + LOGGING
# ============================================================
print(f"""

{'='*70}
V12 MODEL RESULTS SUMMARY
Satellite Image-Only with KNN Backbone
{'='*70}
""")

results_summary = {
    'Electrical Conductance': {
        'knn_blend_r2': ec_knn['blend_r2'],
        'approach_a_r2': ec_a_r2,
        'approach_a_config': ec_a_config,
        'approach_b_r2': ec_b_r2,
        'approach_b_alpha': ec_b_alpha,
        'best_r2': ec_best_r2,
        'best_approach': ec_best_approach,
        'knn_feat_k': ec_knn['feat_k'],
        'knn_feat_p': ec_knn['feat_p'],
        'knn_blend_w': ec_knn['blend_w'],
    },
    'Dissolved Reactive Phosphorus': {
        'knn_blend_r2': drp_knn['blend_r2'],
        'approach_a_r2': drp_a_r2,
        'approach_a_config': drp_a_config,
        'approach_b_r2': drp_b_r2,
        'approach_b_alpha': drp_b_alpha,
        'best_r2': drp_best_r2,
        'best_approach': drp_best_approach,
        'knn_feat_k': drp_knn['feat_k'],
        'knn_feat_p': drp_knn['feat_p'],
        'knn_blend_w': drp_knn['blend_w'],
    },
    'Total Alkalinity': {
        'knn_blend_r2': ta_knn['blend_r2'],
        'approach_a_r2': ta_a_r2,
        'approach_a_config': ta_a_config,
        'approach_b_r2': ta_b_r2,
        'approach_b_alpha': ta_b_alpha,
        'best_r2': ta_best_r2,
        'best_approach': ta_best_approach,
        'knn_feat_k': ta_knn['feat_k'],
        'knn_feat_p': ta_knn['feat_p'],
        'knn_blend_w': ta_knn['blend_w'],
    },
}

print(f"  {'Target':<40s} {'KNN Blend':>10s} {'Appr A':>10s} {'Appr B':>10s} {'Best':>10s} {'Method'}")
print(f"  {'='*95}")
for target in ['Electrical Conductance', 'Dissolved Reactive Phosphorus', 'Total Alkalinity']:
    res = results_summary[target]
    print(f"  {target:<40s} {res['knn_blend_r2']:>+10.4f} {res['approach_a_r2']:>+10.4f} "
          f"{res['approach_b_r2']:>+10.4f} {res['best_r2']:>+10.4f} {res['best_approach']}")

print(f"\n  KNN Configurations:")
for target in ['Electrical Conductance', 'Dissolved Reactive Phosphorus', 'Total Alkalinity']:
    res = results_summary[target]
    print(f"    {target}:")
    print(f"      Feature KNN: K={res['knn_feat_k']}, p={res['knn_feat_p']}")
    print(f"      Blend w_geo={res['knn_blend_w']:.1f}")

print(f"""
{'='*70}
V12 KEY DESIGN CHOICES:
  - Image-only features (no climate, no geology API, no temporal)
  - Sequential prediction: EC -> DRP (with EC stacking) -> TA
  - Two approaches per target: pure ML (A) vs KNN blend + residual (B)
  - {len(SIMILARITY_FEATURES)} similarity features for KD-tree
  - EC: {len(IMAGE_FEATURES_MAP['Electrical Conductance'])} image features
  - DRP: {len(IMAGE_FEATURES_MAP['Dissolved Reactive Phosphorus'])} image features (+EC_predicted)
  - TA: {len(IMAGE_FEATURES_MAP['Total Alkalinity'])} image features

Submit: {sub_path}
{'='*70}
""")

# Save results log
log_data = {'version': 'V12', 'timestamp': ts}
for target in ['Electrical Conductance', 'Dissolved Reactive Phosphorus', 'Total Alkalinity']:
    res = results_summary[target]
    short = target.replace(' ', '_')
    log_data[f'{short}_knn_blend_r2'] = res['knn_blend_r2']
    log_data[f'{short}_approach_a_r2'] = res['approach_a_r2']
    log_data[f'{short}_approach_a_config'] = res['approach_a_config']
    log_data[f'{short}_approach_b_r2'] = res['approach_b_r2']
    log_data[f'{short}_approach_b_alpha'] = res['approach_b_alpha']
    log_data[f'{short}_best_r2'] = res['best_r2']
    log_data[f'{short}_best_approach'] = res['best_approach']
    log_data[f'{short}_knn_feat_k'] = res['knn_feat_k']
    log_data[f'{short}_knn_feat_p'] = res['knn_feat_p']
    log_data[f'{short}_knn_blend_w'] = res['knn_blend_w']

pd.DataFrame([log_data]).to_csv(
    os.path.join(LOG_DIR, f'v12_results_{ts}.csv'), index=False
)
print(f"  Results log: {os.path.join(LOG_DIR, f'v12_results_{ts}.csv')}")

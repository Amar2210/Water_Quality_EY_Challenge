"""
============================================================
EY Water Quality Challenge - V11 Model
============================================================
CORE INNOVATION: Feature-Space KNN

V10 proved that geographic KNN fails when validation is 82-282km
from training data — all predictions converge to a flat regional
mean because every validation point has roughly the same distant
geographic neighbors.

V11 fixes this by finding neighbors based on ENVIRONMENTAL
SIMILARITY instead of geographic proximity:

  "Which training stations have similar geology, land cover,
   terrain, and soil to this validation point?"

Two stations 300km apart on the same limestone with similar
cropland % should have far more similar water quality than two
stations 50km apart on different geology.

ARCHITECTURE:
  1. Geographic KNN-IDW (V10 style — spatial baseline)
  2. Feature-Space KNN-IDW (NEW — finds environmentally similar stations)
  3. Blend: test fixed ratios + distance-adaptive weighting
  4. For EC only: mild XGBoost residual correction

WHY THIS SHOULD PRODUCE MORE VARIANCE:
  Geographic KNN at 82-282km: all validation points find the
  same ~30 overlapping neighbors → flat predictions.
  
  Feature-space KNN: two validation points 10km apart can have
  COMPLETELY DIFFERENT neighbors if one sits on limestone and
  the other on granite → high-variance, informative predictions.

NO NEW DATA — reuses V9 features.
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
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
import xgboost as xgb

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

warnings.filterwarnings('ignore')
np.random.seed(42)

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
EY WATER QUALITY CHALLENGE - V11 MODEL
Feature-Space KNN: Environmental Similarity
{'='*70}
""")

# ============================================================
# [1] LOAD DATA
# ============================================================
print("[1] Loading data...")

# Load V9 features (has everything: V8 env features + KNN features)
train_df = pd.read_csv(os.path.join(EXTRACTED_DIR, 'v9_train_features.csv'))
val_df = pd.read_csv(os.path.join(EXTRACTED_DIR, 'v9_val_features.csv'))

# Load original training data for target values
wq_train = pd.read_csv(os.path.join(DATA_DIR, 'water_quality_training_dataset.csv'))

print(f"  Train samples: {len(train_df)}")
print(f"  Val samples:   {len(val_df)}")

# ============================================================
# [2] BUILD PER-LOCATION DATABASE
# ============================================================
print("\n[2] Building per-location database...")

# Round coordinates for grouping
train_df['lat_r'] = train_df['Latitude'].round(6)
train_df['lon_r'] = train_df['Longitude'].round(6)
val_df['lat_r'] = val_df['Latitude'].round(6)
val_df['lon_r'] = val_df['Longitude'].round(6)

# Per-location target statistics (from original training data)
wq_train['lat_r'] = wq_train['Latitude'].round(6)
wq_train['lon_r'] = wq_train['Longitude'].round(6)

loc_targets = wq_train.groupby(['lat_r', 'lon_r']).agg({
    'Total Alkalinity': 'mean',
    'Electrical Conductance': 'mean',
    'Dissolved Reactive Phosphorus': 'mean',
}).reset_index()

# Per-location environmental features (take first row per location — these are static)
# ─────────────────────────────────────────────────────────────────
# SIMILARITY FEATURES: What defines "environmental similarity"?
# These must be LOCATION-INVARIANT (don't change with sample date)
# ─────────────────────────────────────────────────────────────────

SIMILARITY_FEATURES = [
    # Terrain
    'elevation_m', 'slope_deg', 'terrain_roughness',
    # Soil
    'soil_clay', 'soil_sand', 'soil_soc', 'soil_phh2o', 'soil_cec', 'soil_nitrogen',
    # Land cover 5km (landscape scale)
    'lc_5km_pct_cropland', 'lc_5km_pct_forest', 'lc_5km_pct_grassland',
    'lc_5km_pct_urban', 'lc_5km_pct_bare', 'lc_5km_pct_water',
    'lc_5km_human_impact', 'lc_5km_natural_veg',
    # Geology
    'geo_ec_impact', 'geo_alk_impact',
    'geo_is_carbonate', 'geo_is_siliciclastic', 'geo_is_fine_sedimentary',
    'geo_is_felsic_igneous', 'geo_is_mafic_igneous',
]

# Filter to features that actually exist
SIMILARITY_FEATURES = [f for f in SIMILARITY_FEATURES if f in train_df.columns]
print(f"  Similarity features available: {len(SIMILARITY_FEATURES)}")
for f in SIMILARITY_FEATURES:
    print(f"    {f}")

# Build location database from training data
loc_env = train_df.groupby(['lat_r', 'lon_r'])[SIMILARITY_FEATURES].median().reset_index()

# Merge targets with environmental features
loc_db = loc_env.merge(loc_targets, on=['lat_r', 'lon_r'], how='inner')
print(f"  Training locations: {len(loc_db)}")

# Build validation location features (take first row per location)
val_loc_env = val_df.groupby(['lat_r', 'lon_r'])[SIMILARITY_FEATURES].median().reset_index()
print(f"  Validation locations: {len(val_loc_env)}")

# Check NaN rates in similarity features
nan_rates = loc_db[SIMILARITY_FEATURES].isna().mean()
high_nan = nan_rates[nan_rates > 0.5]
if len(high_nan) > 0:
    print(f"\n  WARNING: High NaN features (>50%):")
    for f, r in high_nan.items():
        print(f"    {f}: {r:.1%}")

# ============================================================
# [3] COORDINATE CONVERSION FOR GEOGRAPHIC KNN
# ============================================================
print("\n[3] Building coordinate systems...")

MEAN_LAT = loc_db['lat_r'].mean()
KM_PER_LAT = 111.32
KM_PER_LON = 111.32 * np.cos(np.radians(MEAN_LAT))

print(f"  Mean latitude: {MEAN_LAT:.2f}°")
print(f"  1° lat ≈ {KM_PER_LAT:.1f} km, 1° lon ≈ {KM_PER_LON:.1f} km")


# ============================================================
# [4] KNN PREDICTION FUNCTIONS
# ============================================================

def build_geo_tree(locs):
    """Build KD-tree from geographic coordinates (in km)."""
    coords_km = np.column_stack([
        locs['lat_r'].values * KM_PER_LAT,
        locs['lon_r'].values * KM_PER_LON
    ])
    return cKDTree(coords_km), coords_km


def build_feature_tree(locs, scaler=None):
    """Build KD-tree from environmental feature space."""
    feat_vals = locs[SIMILARITY_FEATURES].values.copy()
    
    # Impute NaN with column medians
    imp = SimpleImputer(strategy='median')
    feat_vals = imp.fit_transform(feat_vals)
    
    # Scale features
    if scaler is None:
        scaler = StandardScaler()
        feat_scaled = scaler.fit_transform(feat_vals)
    else:
        feat_scaled = scaler.transform(feat_vals)
    
    return cKDTree(feat_scaled), feat_scaled, scaler, imp


def idw_predict(tree, tree_data, target_vals, query_point, k=10, power=1.5):
    """
    IDW prediction: weighted average of K nearest neighbors.
    Returns (prediction, distance_to_nearest).
    """
    max_k = min(k, len(tree_data))
    if max_k == 0:
        return np.nan, np.nan
    
    dists, idxs = tree.query(query_point, k=max_k)
    
    # Handle single-neighbor case
    if max_k == 1:
        dists = np.array([dists])
        idxs = np.array([idxs])
    
    # Filter out NaN targets
    valid = ~np.isnan(target_vals[idxs])
    if valid.sum() == 0:
        return np.nan, dists[0] if len(dists) > 0 else np.nan
    
    v_dists = dists[valid]
    v_vals = target_vals[idxs][valid]
    
    # Floor distance to prevent division by zero
    v_dists = np.maximum(v_dists, 1e-6)
    
    weights = 1.0 / (v_dists ** power)
    prediction = np.sum(v_vals * weights) / np.sum(weights)
    
    return prediction, dists[0]


# ============================================================
# [5] SPATIAL CV SETUP
# ============================================================
print("\n[5] Setting up spatial CV...")

coords = train_df[['Latitude', 'Longitude']].values
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
train_df['spatial_cluster'] = kmeans.fit_predict(coords)

# Map each location to its cluster
loc_cluster = train_df.groupby(['lat_r', 'lon_r'])['spatial_cluster'].first().reset_index()
loc_db = loc_db.merge(loc_cluster, on=['lat_r', 'lon_r'], how='left')

# Show cluster sizes
for c in range(N_CLUSTERS):
    n_locs = (loc_db['spatial_cluster'] == c).sum()
    n_samples = (train_df['spatial_cluster'] == c).sum()
    print(f"  Cluster {c}: {n_locs} locations, {n_samples} samples")


# ============================================================
# [6] ENVIRONMENTAL FEATURES FOR ML CORRECTION
# ============================================================
# These are the features the residual XGBoost model can use.
# Focused set — proven to work in V8/V9.

ML_FEATURES = [f for f in [
    # Terrain
    'elevation_m', 'slope_deg', 'terrain_roughness', 'log_elevation',
    # Soil
    'soil_clay', 'soil_sand', 'soil_soc', 'soil_phh2o', 'soil_cec', 'soil_nitrogen',
    'soil_buffer_capacity', 'soil_sand_clay_ratio',
    # Land cover 5km
    'lc_5km_pct_cropland', 'lc_5km_pct_forest', 'lc_5km_pct_grassland',
    'lc_5km_pct_urban', 'lc_5km_pct_bare', 'lc_5km_pct_water',
    'lc_5km_human_impact', 'lc_5km_natural_veg',
    # Geology
    'geo_ec_impact', 'geo_alk_impact',
    'geo_is_carbonate', 'geo_is_siliciclastic', 'geo_is_fine_sedimentary',
    # Climate (current month)
    'ppt', 'tmax', 'tmin', 'vpd', 'aet', 'aridity_index',
    'ppt_3m_sum', 'tmax_3m_avg',
    # Landsat (unified)
    'unified_nir', 'unified_green', 'unified_swir16', 'unified_swir22',
    'unified_NDMI', 'unified_MNDWI', 'unified_NDVI',
    # Temporal
    'month_sin', 'month_cos', 'is_wet_season', 'is_cool_dry',
    # Key interactions
    'crop_x_ppt', 'urban_x_ppt', 'carb_x_ppt', 'slope_x_clay',
    'anthro_pressure', 'natural_buffer',
] if f in train_df.columns]

print(f"\n  ML correction features: {len(ML_FEATURES)}")


# ============================================================
# [7] MAIN EXPERIMENT: FEATURE-SPACE KNN + BLENDING
# ============================================================
print("\n" + "=" * 70)
print("[7] FEATURE-SPACE KNN EVALUATION")
print("=" * 70)

# Parameters to test
FEAT_K_VALUES = [3, 5, 7, 10, 15, 20]
FEAT_POWER_VALUES = [1.0, 1.5, 2.0]

# Geographic KNN params (V10 best)
GEO_K = 30
GEO_POWER = {
    'Total Alkalinity': 1.5,
    'Electrical Conductance': 1.5,
    'Dissolved Reactive Phosphorus': 1.0,
}

# Blend weights: w_geo in [0, 1], where 0 = pure feature KNN, 1 = pure geo KNN
BLEND_WEIGHTS = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]

# ML correction alphas (only for EC)
ML_ALPHAS = [0.0, 0.1, 0.2, 0.3]

gkf = GroupKFold(n_splits=N_CLUSTERS)
groups = train_df['spatial_cluster'].values

all_results = {}  # target -> list of result dicts

for target in TARGETS:
    print(f"\n{'─'*70}")
    print(f"TARGET: {target}")
    print(f"{'─'*70}")
    
    short = target.replace(' ', '_')
    y_all = train_df[target].values
    valid_mask = ~np.isnan(y_all)
    
    target_results = []
    
    # ── STEP A: Find best feature-space K and power ──────────────
    print(f"\n  [A] Tuning feature-space KNN (K and power)...")
    print(f"      (using w_geo=0.0, pure feature-space KNN)")
    
    best_feat_r2 = -999
    best_feat_k = 10
    best_feat_p = 1.5
    
    for feat_k in FEAT_K_VALUES:
        for feat_p in FEAT_POWER_VALUES:
            fold_scores = []
            
            for fold, (tr_idx, te_idx) in enumerate(gkf.split(
                    np.arange(len(train_df)), y_all, groups)):
                
                # Get fold clusters
                te_clusters = set(groups[te_idx])
                
                # Split locations by cluster
                fold_train_locs = loc_db[~loc_db['spatial_cluster'].isin(te_clusters)].reset_index(drop=True)
                fold_test_locs = loc_db[loc_db['spatial_cluster'].isin(te_clusters)].reset_index(drop=True)
                
                if len(fold_train_locs) == 0 or len(fold_test_locs) == 0:
                    continue
                
                # Build feature-space tree from train locations
                feat_tree, feat_data, scaler, imp = build_feature_tree(fold_train_locs)
                train_target_vals = fold_train_locs[target].values
                
                # Prepare test location features
                test_feat_raw = fold_test_locs[SIMILARITY_FEATURES].values.copy()
                test_feat_imp = imp.transform(test_feat_raw)
                test_feat_scaled = scaler.transform(test_feat_imp)
                
                # Predict per test location
                loc_preds = {}
                for i in range(len(fold_test_locs)):
                    pred, _ = idw_predict(
                        feat_tree, feat_data, train_target_vals,
                        test_feat_scaled[i], k=feat_k, power=feat_p
                    )
                    lat_r = fold_test_locs.iloc[i]['lat_r']
                    lon_r = fold_test_locs.iloc[i]['lon_r']
                    loc_preds[(lat_r, lon_r)] = pred
                
                # Map to samples
                te_preds = []
                te_actuals = []
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
                marker = '✓' if mean_r2 > 0 else '✗'
                
                if mean_r2 > best_feat_r2:
                    best_feat_r2 = mean_r2
                    best_feat_k = feat_k
                    best_feat_p = feat_p
                
                # Only print a subset to keep output manageable
                if feat_p == 1.5 or (feat_k == best_feat_k and feat_p == best_feat_p):
                    print(f"      {marker} K={feat_k:2d}, p={feat_p}: R² = {mean_r2:+.4f}  "
                          f"[{', '.join(f'{s:+.3f}' for s in fold_scores)}]")
    
    print(f"\n      ★ Best feature-space: K={best_feat_k}, p={best_feat_p}, "
          f"R² = {best_feat_r2:+.4f}")
    
    # ── STEP B: Geographic KNN baseline (V10 style) ──────────────
    print(f"\n  [B] Geographic KNN baseline (K={GEO_K}, p={GEO_POWER[target]})...")
    
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
            q_km = np.array([
                fold_test_locs.iloc[i]['lat_r'] * KM_PER_LAT,
                fold_test_locs.iloc[i]['lon_r'] * KM_PER_LON
            ])
            pred, _ = idw_predict(
                geo_tree, geo_data, train_target_vals,
                q_km, k=GEO_K, power=GEO_POWER[target]
            )
            lat_r = fold_test_locs.iloc[i]['lat_r']
            lon_r = fold_test_locs.iloc[i]['lon_r']
            loc_preds[(lat_r, lon_r)] = pred
        
        te_preds = []
        te_actuals = []
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
    marker = '✓' if geo_mean_r2 > 0 else '✗'
    print(f"      {marker} Geographic KNN: R² = {geo_mean_r2:+.4f}  "
          f"[{', '.join(f'{s:+.3f}' for s in geo_fold_scores)}]")
    
    # ── STEP C: Blend geographic + feature-space KNN ──────────────
    print(f"\n  [C] Blending geographic + feature-space KNN...")
    print(f"      Feature: K={best_feat_k}, p={best_feat_p}")
    print(f"      Geographic: K={GEO_K}, p={GEO_POWER[target]}")
    
    best_blend_r2 = -999
    best_blend_w = 0.0
    best_blend_fold_preds = None  # Store per-sample OOF predictions for ML correction
    
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
            
            # Build both trees
            geo_tree, geo_data = build_geo_tree(fold_train_locs)
            feat_tree, feat_data, scaler, imp = build_feature_tree(fold_train_locs)
            
            # Prepare test features
            test_feat_raw = fold_test_locs[SIMILARITY_FEATURES].values.copy()
            test_feat_imp = imp.transform(test_feat_raw)
            test_feat_scaled = scaler.transform(test_feat_imp)
            
            # Predict per test location
            loc_preds = {}
            for i in range(len(fold_test_locs)):
                # Geographic prediction
                q_km = np.array([
                    fold_test_locs.iloc[i]['lat_r'] * KM_PER_LAT,
                    fold_test_locs.iloc[i]['lon_r'] * KM_PER_LON
                ])
                geo_pred, geo_dist = idw_predict(
                    geo_tree, geo_data, train_target_vals,
                    q_km, k=GEO_K, power=GEO_POWER[target]
                )
                
                # Feature-space prediction
                feat_pred, feat_dist = idw_predict(
                    feat_tree, feat_data, train_target_vals,
                    test_feat_scaled[i], k=best_feat_k, power=best_feat_p
                )
                
                # Blend
                if np.isnan(geo_pred) and np.isnan(feat_pred):
                    blended = np.nan
                elif np.isnan(geo_pred):
                    blended = feat_pred
                elif np.isnan(feat_pred):
                    blended = geo_pred
                else:
                    blended = w_geo * geo_pred + (1 - w_geo) * feat_pred
                
                lat_r = fold_test_locs.iloc[i]['lat_r']
                lon_r = fold_test_locs.iloc[i]['lon_r']
                loc_preds[(lat_r, lon_r)] = blended
            
            # Map to samples
            te_preds_list = []
            te_actuals_list = []
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
            marker = '✓' if mean_r2 > 0 else '✗'
            print(f"      {marker} w_geo={w_geo:.1f}: R² = {mean_r2:+.4f}  "
                  f"[{', '.join(f'{s:+.3f}' for s in fold_scores)}]")
            
            target_results.append({
                'type': 'blend', 'w_geo': w_geo, 'r2': mean_r2,
                'fold_scores': fold_scores, 'alpha': 0.0
            })
            
            if mean_r2 > best_blend_r2:
                best_blend_r2 = mean_r2
                best_blend_w = w_geo
                best_blend_fold_preds = oof_preds.copy()
    
    print(f"\n      ★ Best blend: w_geo={best_blend_w:.1f}, R² = {best_blend_r2:+.4f}")
    
    # ── STEP D: ML Residual Correction (EC primary, test others) ──
    print(f"\n  [D] ML residual correction on best blend...")
    
    # Decide which targets get ML correction
    # TA: V9 showed alpha=0.0 is best. Skip unless blend is strong.
    # EC: V9 showed alpha=0.2 helps. Test.
    # DRP: Test cautiously.
    test_ml = True  # Test for all, but expect EC to benefit most
    
    best_final_r2 = best_blend_r2
    best_final_alpha = 0.0
    best_final_config = f"blend_w{best_blend_w:.1f}_a0.0"
    
    if test_ml and best_blend_fold_preds is not None:
        for alpha in ML_ALPHAS:
            if alpha == 0.0:
                continue  # Already have this from blend
            
            fold_scores = []
            
            for fold, (tr_idx, te_idx) in enumerate(gkf.split(
                    np.arange(len(train_df)), y_all, groups)):
                
                # Get blend predictions for train fold (need to recompute for train samples)
                te_clusters = set(groups[te_idx])
                fold_train_locs = loc_db[~loc_db['spatial_cluster'].isin(te_clusters)].reset_index(drop=True)
                
                # Build trees for this fold
                train_target_vals = fold_train_locs[target].values
                geo_tree, geo_data = build_geo_tree(fold_train_locs)
                feat_tree, feat_data, scaler, imp = build_feature_tree(fold_train_locs)
                
                # Compute blend predictions for TRAINING samples in this fold
                tr_blend_preds = np.full(len(tr_idx), np.nan)
                for j, idx in enumerate(tr_idx):
                    key = (train_df.iloc[idx]['lat_r'], train_df.iloc[idx]['lon_r'])
                    
                    # Find this location in fold_train_locs
                    loc_mask = (fold_train_locs['lat_r'] == key[0]) & (fold_train_locs['lon_r'] == key[1])
                    if loc_mask.sum() == 0:
                        continue
                    
                    loc_idx = loc_mask.values.argmax()
                    
                    # Geo prediction (leave-one-out: exclude this location)
                    # Build a temporary tree without this location
                    temp_locs = fold_train_locs.drop(loc_idx).reset_index(drop=True)
                    if len(temp_locs) == 0:
                        continue
                    
                    temp_geo_tree, temp_geo_data = build_geo_tree(temp_locs)
                    temp_feat_tree, temp_feat_data, temp_scaler, temp_imp = build_feature_tree(temp_locs)
                    temp_targets = temp_locs[target].values
                    
                    q_km = np.array([key[0] * KM_PER_LAT, key[1] * KM_PER_LON])
                    geo_pred, _ = idw_predict(temp_geo_tree, temp_geo_data, temp_targets,
                                              q_km, k=GEO_K, power=GEO_POWER[target])
                    
                    loc_feat = imp.transform(fold_train_locs.iloc[[loc_idx]][SIMILARITY_FEATURES].values)
                    loc_feat_s = scaler.transform(loc_feat)
                    feat_pred, _ = idw_predict(temp_feat_tree, temp_feat_data, temp_targets,
                                               loc_feat_s[0], k=best_feat_k, power=best_feat_p)
                    
                    if not np.isnan(geo_pred) and not np.isnan(feat_pred):
                        tr_blend_preds[j] = best_blend_w * geo_pred + (1 - best_blend_w) * feat_pred
                    elif not np.isnan(feat_pred):
                        tr_blend_preds[j] = feat_pred
                    elif not np.isnan(geo_pred):
                        tr_blend_preds[j] = geo_pred
                
                # Compute residuals for training
                tr_actuals = y_all[tr_idx]
                tr_residuals = tr_actuals - tr_blend_preds
                
                # Filter valid
                tr_valid = ~np.isnan(tr_residuals) & ~np.isnan(tr_actuals)
                if tr_valid.sum() < 20:
                    continue
                
                # Train residual model
                X_tr = train_df.iloc[tr_idx][ML_FEATURES].values[tr_valid]
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
                
                # Predict residual for test fold
                X_te = train_df.iloc[te_idx][ML_FEATURES].values
                X_te_imp = ml_imp.transform(X_te)
                te_residual_pred = model.predict(X_te_imp)
                
                # Use OOF blend predictions from step C
                te_preds = []
                te_actuals_list = []
                for j, idx in enumerate(te_idx):
                    if np.isnan(y_all[idx]) or np.isnan(best_blend_fold_preds[idx]):
                        continue
                    final = best_blend_fold_preds[idx] + alpha * te_residual_pred[j]
                    te_preds.append(final)
                    te_actuals_list.append(y_all[idx])
                
                if len(te_preds) > 1:
                    fold_scores.append(r2_score(te_actuals_list, te_preds))
            
            if fold_scores:
                mean_r2 = np.mean(fold_scores)
                marker = '✓' if mean_r2 > 0 else '✗'
                print(f"      {marker} alpha={alpha:.1f}: R² = {mean_r2:+.4f}  "
                      f"[{', '.join(f'{s:+.3f}' for s in fold_scores)}]")
                
                target_results.append({
                    'type': 'blend+ml', 'w_geo': best_blend_w,
                    'r2': mean_r2, 'fold_scores': fold_scores, 'alpha': alpha
                })
                
                if mean_r2 > best_final_r2:
                    best_final_r2 = mean_r2
                    best_final_alpha = alpha
                    best_final_config = f"blend_w{best_blend_w:.1f}_a{alpha:.1f}"
    
    all_results[target] = {
        'best_feat_k': best_feat_k,
        'best_feat_p': best_feat_p,
        'feat_only_r2': best_feat_r2,
        'geo_only_r2': geo_mean_r2,
        'best_blend_w': best_blend_w,
        'best_blend_r2': best_blend_r2,
        'best_final_r2': best_final_r2,
        'best_final_alpha': best_final_alpha,
        'best_config': best_final_config,
        'details': target_results,
    }
    
    print(f"\n  ★ FINAL BEST for {target}:")
    print(f"    Feature-space KNN only: R² = {best_feat_r2:+.4f}")
    print(f"    Geographic KNN only:    R² = {geo_mean_r2:+.4f}")
    print(f"    Best blend (w_geo={best_blend_w:.1f}): R² = {best_blend_r2:+.4f}")
    print(f"    Best with ML (α={best_final_alpha:.1f}): R² = {best_final_r2:+.4f}")


# ============================================================
# [8] GENERATE FINAL PREDICTIONS FOR VALIDATION
# ============================================================
print("\n" + "=" * 70)
print("[8] GENERATING VALIDATION PREDICTIONS")
print("=" * 70)

final_predictions = {}

for target in TARGETS:
    print(f"\n  {target}:")
    res = all_results[target]
    w_geo = res['best_blend_w']
    feat_k = res['best_feat_k']
    feat_p = res['best_feat_p']
    alpha = res['best_final_alpha']
    
    print(f"    Config: w_geo={w_geo:.1f}, feat_K={feat_k}, feat_p={feat_p}, alpha={alpha:.1f}")
    
    # Build trees from ALL training locations
    geo_tree, geo_data = build_geo_tree(loc_db)
    feat_tree, feat_data, scaler, imp = build_feature_tree(loc_db)
    train_target_vals = loc_db[target].values
    
    # Prepare validation features
    val_feat_raw = val_loc_env[SIMILARITY_FEATURES].values.copy()
    val_feat_imp = imp.transform(val_feat_raw)
    val_feat_scaled = scaler.transform(val_feat_imp)
    
    # Predict per validation location
    val_loc_preds = {}
    val_loc_geo_dist = {}
    for i in range(len(val_loc_env)):
        lat_r = val_loc_env.iloc[i]['lat_r']
        lon_r = val_loc_env.iloc[i]['lon_r']
        
        # Geographic
        q_km = np.array([lat_r * KM_PER_LAT, lon_r * KM_PER_LON])
        geo_pred, geo_dist = idw_predict(
            geo_tree, geo_data, train_target_vals,
            q_km, k=GEO_K, power=GEO_POWER[target]
        )
        
        # Feature-space
        feat_pred, feat_dist = idw_predict(
            feat_tree, feat_data, train_target_vals,
            val_feat_scaled[i], k=feat_k, power=feat_p
        )
        
        # Blend
        if np.isnan(geo_pred) and np.isnan(feat_pred):
            blended = np.nan
        elif np.isnan(geo_pred):
            blended = feat_pred
        elif np.isnan(feat_pred):
            blended = geo_pred
        else:
            blended = w_geo * geo_pred + (1 - w_geo) * feat_pred
        
        val_loc_preds[(lat_r, lon_r)] = blended
        val_loc_geo_dist[(lat_r, lon_r)] = geo_dist
    
    # Map to validation samples
    val_preds = np.full(len(val_df), np.nan)
    for i in range(len(val_df)):
        key = (val_df.iloc[i]['lat_r'], val_df.iloc[i]['lon_r'])
        if key in val_loc_preds:
            val_preds[i] = val_loc_preds[key]
    
    # ML residual correction
    if alpha > 0:
        print(f"    Applying ML correction (alpha={alpha:.1f})...")
        
        # Compute blend predictions for ALL training data (leave-one-out)
        tr_blend = np.full(len(train_df), np.nan)
        # For speed, compute per unique training location
        for i in range(len(loc_db)):
            lat_r = loc_db.iloc[i]['lat_r']
            lon_r = loc_db.iloc[i]['lon_r']
            
            # Build trees without this location
            temp_locs = loc_db.drop(i).reset_index(drop=True)
            if len(temp_locs) == 0:
                continue
            temp_geo_tree, temp_geo_data = build_geo_tree(temp_locs)
            temp_feat_tree, temp_feat_data, temp_scaler, temp_imp = build_feature_tree(temp_locs)
            temp_targets = temp_locs[target].values
            
            q_km = np.array([lat_r * KM_PER_LAT, lon_r * KM_PER_LON])
            g_pred, _ = idw_predict(temp_geo_tree, temp_geo_data, temp_targets,
                                     q_km, k=GEO_K, power=GEO_POWER[target])
            
            loc_feat = imp.transform(loc_db.iloc[[i]][SIMILARITY_FEATURES].values)
            loc_feat_s = scaler.transform(loc_feat)
            f_pred, _ = idw_predict(temp_feat_tree, temp_feat_data, temp_targets,
                                     loc_feat_s[0], k=feat_k, power=feat_p)
            
            if not np.isnan(g_pred) and not np.isnan(f_pred):
                loc_blend = w_geo * g_pred + (1 - w_geo) * f_pred
            elif not np.isnan(f_pred):
                loc_blend = f_pred
            elif not np.isnan(g_pred):
                loc_blend = g_pred
            else:
                loc_blend = np.nan
            
            # Assign to all samples from this location
            mask = (train_df['lat_r'] == lat_r) & (train_df['lon_r'] == lon_r)
            tr_blend[mask] = loc_blend
        
        # Train residual model on all training data
        tr_residuals = y_all - tr_blend
        tr_valid = ~np.isnan(tr_residuals) & ~np.isnan(y_all)
        
        X_tr = train_df[ML_FEATURES].values[tr_valid]
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
        
        # Predict residual for validation
        X_val = val_df[ML_FEATURES].values
        X_val_imp = ml_imp.transform(X_val)
        val_residual = model.predict(X_val_imp)
        
        val_preds = val_preds + alpha * val_residual
        print(f"    Residual range: [{val_residual.min():.1f}, {val_residual.max():.1f}]")
    
    # Fill any remaining NaN with global training mean
    global_mean = loc_db[target].mean()
    nan_count = np.isnan(val_preds).sum()
    if nan_count > 0:
        print(f"    Filling {nan_count} NaN with global mean {global_mean:.1f}")
        val_preds[np.isnan(val_preds)] = global_mean
    
    # Clip to physical ranges
    if target == 'Total Alkalinity':
        val_preds = np.clip(val_preds, 0, 1000)
    elif target == 'Electrical Conductance':
        val_preds = np.clip(val_preds, 0, 10000)
    elif target == 'Dissolved Reactive Phosphorus':
        val_preds = np.clip(val_preds, 0, 5000)
    
    final_predictions[target] = val_preds
    print(f"    Prediction range: [{val_preds.min():.1f}, {val_preds.max():.1f}], "
          f"mean={val_preds.mean():.1f}, std={val_preds.std():.1f}")
    
    # Compare variance with V10
    print(f"    V10 comparison: V10 range was much narrower (flat predictions)")


# ============================================================
# [9] GENERATE SUBMISSION
# ============================================================
print("\n" + "=" * 70)
print("[9] GENERATING SUBMISSION")
print("=" * 70)

submission = pd.read_csv(os.path.join(DATA_DIR, 'submission_template.csv'))
for target in TARGETS:
    submission[target] = final_predictions[target]

sub_path = os.path.join(SUBMISSION_DIR, 'submission_v11.csv')
sub_ts_path = os.path.join(SUBMISSION_DIR, f'submission_v11_{ts}.csv')
submission.to_csv(sub_path, index=False)
submission.to_csv(sub_ts_path, index=False)

print(f"  Saved: {sub_path}")
print(f"  Saved: {sub_ts_path}")

for target in TARGETS:
    n_nan = submission[target].isna().sum()
    if n_nan > 0:
        print(f"  WARNING: {target} has {n_nan} NaN values!")


# ============================================================
# [10] RESULTS SUMMARY
# ============================================================
print(f"""

{'='*70}
V11 MODEL RESULTS SUMMARY
{'='*70}
""")

print(f"  {'Target':<40s} {'Feat KNN':>10s} {'Geo KNN':>10s} {'Blend':>10s} {'+ ML':>10s}")
print(f"  {'─'*80}")

for target in TARGETS:
    res = all_results[target]
    print(f"  {target:<40s} "
          f"{res['feat_only_r2']:>+10.4f} "
          f"{res['geo_only_r2']:>+10.4f} "
          f"{res['best_blend_r2']:>+10.4f} "
          f"{res['best_final_r2']:>+10.4f}")

print(f"\n  Best configurations:")
for target in TARGETS:
    res = all_results[target]
    print(f"    {target}:")
    print(f"      Feature K={res['best_feat_k']}, p={res['best_feat_p']}")
    print(f"      Blend w_geo={res['best_blend_w']:.1f}, alpha={res['best_final_alpha']:.1f}")

print(f"""
{'─'*70}
KEY V11 INNOVATION:
  ★ Feature-Space KNN: Find training stations with similar
    geology, land cover, terrain, and soil — regardless of
    geographic distance.
  
  ★ This should produce HIGHER VARIANCE predictions than V10's
    geographic KNN, because environmentally different validation
    points get different neighbors even when they're all 
    100+ km from training data.

Submit: {sub_path}
{'='*70}
""")

# Save results log
log_data = {'version': 'V11', 'timestamp': ts}
for target in TARGETS:
    res = all_results[target]
    short = target.replace(' ', '_')
    log_data[f'{short}_feat_knn_r2'] = res['feat_only_r2']
    log_data[f'{short}_geo_knn_r2'] = res['geo_only_r2']
    log_data[f'{short}_blend_r2'] = res['best_blend_r2']
    log_data[f'{short}_blend_w_geo'] = res['best_blend_w']
    log_data[f'{short}_final_r2'] = res['best_final_r2']
    log_data[f'{short}_final_alpha'] = res['best_final_alpha']
    log_data[f'{short}_feat_k'] = res['best_feat_k']
    log_data[f'{short}_feat_p'] = res['best_feat_p']

pd.DataFrame([log_data]).to_csv(
    os.path.join(LOG_DIR, f'v11_results_{ts}.csv'), index=False
)
print(f"  Results log: {os.path.join(LOG_DIR, f'v11_results_{ts}.csv')}")

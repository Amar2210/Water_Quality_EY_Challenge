"""
============================================================
EY Water Quality Challenge - V9 Feature Engineering
============================================================
THE BIG CHANGE: Spatial KNN Target Features

Instead of just using environmental features, V9 adds:
  "What are the actual water quality values at the K nearest
   training locations?"

This is the critical missing piece. When predicting EC at a test
location, the single most informative feature is: "What was EC
at the nearest training station?" — adjusted slightly for
seasonal/environmental differences.

NEW IN V9:
  1. KNN target features:
     - Mean target value at K=3,5,10 nearest training locations
     - Distance-weighted mean (closer = higher weight)
     - Min/Max target at K nearest
     - Distance to nearest training point (km)
  
  2. Spatial interpolation features:
     - IDW (Inverse Distance Weighted) prediction for each target
     - Spatial residual: actual - IDW predicted (training only)
  
  3. Regional target encoding:
     - Mean target by spatial cluster
     - Leave-one-out encoding to avoid leakage
  
  4. Log-transform DRP (highly right-skewed)
  
  5. All V8 features (land cover, geology, climate, Landsat, etc.)

WHY THIS SHOULD WORK:
  The top teams score 0.82. That's impossible from just 
  environmental features — water quality has too much unexplained
  variance. They MUST be using spatial proximity to training data.
  
  If test location A is 3km from training location B where 
  EC=500 and alkalinity=200, then EC≈500 and alkalinity≈200
  at location A is already a strong baseline. The environmental
  features then explain the residual variation.

OUTPUT:
  data/extracted/v9_train_features.csv
  data/extracted/v9_val_features.csv
  data/extracted/v9_feature_columns.txt
============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import os
from scipy.spatial import cKDTree

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'Jupyter Notebook Package')
EXTRACTED_DIR = os.path.join(BASE_DIR, '..', 'data', 'extracted')
TARGETS = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
eps = 1e-8

print("=" * 70)
print("EY WATER QUALITY - V9 FEATURE ENGINEERING")
print("Spatial KNN target features + all V8 features")
print("=" * 70)


# ============================================================
# [1] LOAD V8 FEATURES (already has land cover, geology, etc.)
# ============================================================
print("\n[1] Loading V8 features as base...")

v8_train_path = os.path.join(EXTRACTED_DIR, 'v8_train_features.csv')
v8_val_path = os.path.join(EXTRACTED_DIR, 'v8_val_features.csv')
v8_feat_path = os.path.join(EXTRACTED_DIR, 'v8_feature_columns.txt')

if not os.path.exists(v8_train_path):
    print("  ERROR: V8 features not found! Run 18_feature_engineering_v8.py first")
    import sys; sys.exit(1)

train_df = pd.read_csv(v8_train_path)
val_df = pd.read_csv(v8_val_path)
with open(v8_feat_path, 'r') as f:
    v8_feature_cols = [line.strip() for line in f if line.strip()]

print(f"  Train: {train_df.shape}")
print(f"  Val:   {val_df.shape}")
print(f"  V8 features: {len(v8_feature_cols)}")


# ============================================================
# [2] BUILD SPATIAL KNN TARGET FEATURES
# ============================================================
print("\n" + "=" * 70)
print("[2] BUILDING SPATIAL KNN TARGET FEATURES")
print("=" * 70)

# We need the original target values for KNN computation
wq_train = pd.read_csv(os.path.join(DATA_DIR, 'water_quality_training_dataset.csv'))

# Build a lookup: for each unique location, get all target measurements
# Group by location, compute per-location stats
wq_train['Latitude'] = wq_train['Latitude'].round(6)
wq_train['Longitude'] = wq_train['Longitude'].round(6)

loc_stats = wq_train.groupby(['Latitude', 'Longitude']).agg({
    'Total Alkalinity': ['mean', 'std', 'median', 'count'],
    'Electrical Conductance': ['mean', 'std', 'median', 'count'],
    'Dissolved Reactive Phosphorus': ['mean', 'std', 'median', 'count'],
}).reset_index()

# Flatten column names
loc_stats.columns = ['_'.join(c).strip('_') for c in loc_stats.columns]
loc_stats.rename(columns={
    'Latitude_': 'Latitude', 'Longitude_': 'Longitude'
}, inplace=True)

# Fix column names
rename_map = {}
for target in TARGETS:
    short = target.replace(' ', '_')
    for stat in ['mean', 'std', 'median', 'count']:
        old = f'{target}_{stat}'
        new = f'loc_{short}_{stat}'
        if old in loc_stats.columns:
            rename_map[old] = new
loc_stats.rename(columns=rename_map, inplace=True)

print(f"  Unique training locations: {len(loc_stats)}")
print(f"  Location stats columns: {loc_stats.shape[1]}")

# Build KD-tree from training locations
train_coords = loc_stats[['Latitude', 'Longitude']].values
# Convert to approximate km (rough conversion at ~30°S latitude)
train_coords_km = train_coords.copy()
train_coords_km[:, 0] *= 111.32  # lat degrees to km
train_coords_km[:, 1] *= 111.32 * np.cos(np.radians(np.mean(train_coords[:, 0])))  # lon degrees to km

tree = cKDTree(train_coords_km)

K_VALUES = [3, 5, 10, 20]


def compute_knn_features(query_lat, query_lon, target_values_by_loc, k_values, 
                          is_training=False, exclude_self=True):
    """
    Compute KNN target features for a single query location.
    
    For training data: use leave-one-out (exclude the query location itself)
    For validation data: use all training locations
    
    Args:
        query_lat, query_lon: query location
        target_values_by_loc: dict mapping (lat, lon) -> {target: mean_value}
        k_values: list of K values [3, 5, 10, 20]
        is_training: if True, exclude the exact location from neighbors
        exclude_self: whether to exclude exact same location
    
    Returns:
        dict of features
    """
    result = {}
    
    # Convert query to km
    q_km = np.array([
        query_lat * 111.32,
        query_lon * 111.32 * np.cos(np.radians(np.mean(train_coords[:, 0])))
    ])
    
    # Query for K+1 neighbors (extra in case we need to skip self)
    max_k = max(k_values) + 5
    distances_km, indices = tree.query(q_km, k=min(max_k, len(train_coords)))
    
    # For training points, skip self (distance ≈ 0)
    if is_training and exclude_self:
        mask = distances_km > 0.01  # > 10m away
        distances_km = distances_km[mask]
        indices = indices[mask]
    
    # Distance to nearest training location
    if len(distances_km) > 0:
        result['dist_nearest_km'] = distances_km[0]
        result['dist_5th_nearest_km'] = distances_km[min(4, len(distances_km)-1)]
    else:
        result['dist_nearest_km'] = np.nan
        result['dist_5th_nearest_km'] = np.nan
    
    # For each target and each K
    for target in TARGETS:
        short = target.replace(' ', '_')
        col_name = f'loc_{short}_mean'
        
        # Get target values for neighboring locations
        neighbor_values = []
        neighbor_dists = []
        for i, idx in enumerate(indices):
            if idx < len(loc_stats):
                val = loc_stats.iloc[idx].get(col_name, np.nan)
                if not np.isnan(val):
                    neighbor_values.append(val)
                    neighbor_dists.append(max(distances_km[i], 0.01))  # floor at 10m
        
        neighbor_values = np.array(neighbor_values)
        neighbor_dists = np.array(neighbor_dists)
        
        for k in k_values:
            prefix = f'knn{k}_{short}'
            
            if len(neighbor_values) >= 1:
                kv = neighbor_values[:k]
                kd = neighbor_dists[:k]
                
                # Simple mean of K nearest
                result[f'{prefix}_mean'] = np.mean(kv)
                
                # Distance-weighted mean (IDW)
                weights = 1.0 / (kd ** 2)  # Inverse distance squared
                result[f'{prefix}_idw'] = np.sum(kv * weights) / np.sum(weights)
                
                # Min / max / std
                result[f'{prefix}_min'] = np.min(kv)
                result[f'{prefix}_max'] = np.max(kv)
                if len(kv) > 1:
                    result[f'{prefix}_std'] = np.std(kv)
                else:
                    result[f'{prefix}_std'] = 0.0
                
                # Range (spatial heterogeneity)
                result[f'{prefix}_range'] = np.max(kv) - np.min(kv)
                
            else:
                for suffix in ['mean', 'idw', 'min', 'max', 'std', 'range']:
                    result[f'{prefix}_{suffix}'] = np.nan
    
    return result


# Compute KNN features for training data (leave-one-out)
print("\n  Computing KNN features for TRAINING data (leave-one-out)...")
train_knn_results = []
for i in range(len(train_df)):
    lat = train_df.loc[i, 'Latitude']
    lon = train_df.loc[i, 'Longitude']
    knn_feats = compute_knn_features(lat, lon, None, K_VALUES, 
                                      is_training=True, exclude_self=True)
    train_knn_results.append(knn_feats)
    
    if (i + 1) % 500 == 0 or (i + 1) == len(train_df):
        print(f"    {i+1}/{len(train_df)} ({100*(i+1)/len(train_df):.1f}%)")

train_knn_df = pd.DataFrame(train_knn_results)
print(f"  Training KNN features: {train_knn_df.shape}")

# Compute KNN features for validation data (use all training)
print("\n  Computing KNN features for VALIDATION data...")
val_knn_results = []
for i in range(len(val_df)):
    lat = val_df.loc[i, 'Latitude']
    lon = val_df.loc[i, 'Longitude']
    knn_feats = compute_knn_features(lat, lon, None, K_VALUES,
                                      is_training=False, exclude_self=False)
    val_knn_results.append(knn_feats)
    
    if (i + 1) % 200 == 0 or (i + 1) == len(val_df):
        print(f"    {i+1}/{len(val_df)} ({100*(i+1)/len(val_df):.1f}%)")

val_knn_df = pd.DataFrame(val_knn_results)
print(f"  Validation KNN features: {val_knn_df.shape}")


# ============================================================
# [3] ADD KNN FEATURES TO DATAFRAMES
# ============================================================
print("\n[3] Merging KNN features...")

# Concatenate KNN features
train_df = pd.concat([train_df.reset_index(drop=True), train_knn_df.reset_index(drop=True)], axis=1)
val_df = pd.concat([val_df.reset_index(drop=True), val_knn_df.reset_index(drop=True)], axis=1)

print(f"  Train after KNN: {train_df.shape}")
print(f"  Val after KNN:   {val_df.shape}")


# ============================================================
# [4] ADDITIONAL V9 ENGINEERED FEATURES
# ============================================================
print("\n[4] Engineering V9-specific features...")

def add_v9_features(df, dataset_name):
    """Add V9-specific derived features."""
    n_before = df.shape[1]
    
    # --- IDW residuals (training only: actual - IDW predicted) ---
    # These capture "how different is this location from its neighbors?"
    # A large positive residual means "unusually high for this area"
    for target in TARGETS:
        short = target.replace(' ', '_')
        idw_col = f'knn5_{short}_idw'
        
        if target in df.columns and idw_col in df.columns:
            df[f'residual_{short}'] = df[target] - df[idw_col]
    
    # --- Spatial anomaly features ---
    # How unusual is each KNN prediction compared to global training mean?
    for target in TARGETS:
        short = target.replace(' ', '_')
        idw_col = f'knn5_{short}_idw'
        mean_col = f'knn5_{short}_mean'
        
        if idw_col in df.columns:
            global_mean = train_df[target].mean() if target in train_df.columns else np.nan
            if not np.isnan(global_mean) and global_mean != 0:
                df[f'spatial_anomaly_{short}'] = (df[idw_col] - global_mean) / max(abs(global_mean), 1)
    
    # --- KNN × environment interactions ---
    # "KNN says EC=500, AND it's hot and dry" → EC probably even higher
    for target in TARGETS:
        short = target.replace(' ', '_')
        idw_col = f'knn5_{short}_idw'
        
        if idw_col in df.columns:
            # KNN prediction × season
            if 'is_wet_season' in df.columns:
                df[f'knn_{short}_x_wet'] = df[idw_col] * df['is_wet_season']
            
            # KNN prediction × precipitation (dilution effect)
            if 'ppt' in df.columns:
                df[f'knn_{short}_x_ppt'] = df[idw_col] * df['ppt'] / 100  # Scale down
    
    # --- Distance-based confidence ---
    # How much should we trust the KNN predictions?
    if 'dist_nearest_km' in df.columns:
        df['knn_confidence'] = 1.0 / (1.0 + df['dist_nearest_km'])
        df['log_dist_nearest'] = np.log1p(df['dist_nearest_km'])
    
    # --- Neighborhood homogeneity ---
    # If all K neighbors have similar values, prediction is more reliable
    for target in TARGETS:
        short = target.replace(' ', '_')
        std_col = f'knn5_{short}_std'
        mean_col = f'knn5_{short}_mean'
        
        if std_col in df.columns and mean_col in df.columns:
            df[f'knn_cv_{short}'] = np.where(
                df[mean_col].abs() > eps,
                df[std_col] / df[mean_col].abs(),
                np.nan
            )
    
    n_after = df.shape[1]
    print(f"    [{dataset_name}] Features: {n_before} -> {n_after} (+{n_after - n_before})")
    return df


train_df = add_v9_features(train_df, 'TRAIN')
val_df = add_v9_features(val_df, 'VAL')


# ============================================================
# [5] SELECT FEATURE COLUMNS
# ============================================================
print("\n[5] Selecting feature columns...")

exclude_cols = set(TARGETS + ['Latitude', 'Longitude', 'Sample Date', 'spatial_cluster'])
# Also exclude residual features (they use target values, only exist in training)
residual_cols = [c for c in train_df.columns if c.startswith('residual_')]

feature_cols = [c for c in train_df.columns 
                if c not in exclude_cols 
                and c not in residual_cols
                and train_df[c].dtype != 'object'
                and c in val_df.columns]  # Must exist in BOTH train and val

# Drop >90% NaN features
nan_rates = train_df[feature_cols].isna().mean()
high_nan = nan_rates[nan_rates > 0.9].index.tolist()
if high_nan:
    print(f"  Dropping {len(high_nan)} features with >90% NaN")
    feature_cols = [c for c in feature_cols if c not in high_nan]

# Remove duplicate columns
feature_cols = list(dict.fromkeys(feature_cols))

print(f"  Total features: {len(feature_cols)}")

# Feature group breakdown
groups = {
    'KNN Target Features (NEW)': [c for c in feature_cols if c.startswith('knn') or 
                                   c.startswith('dist_') or c.startswith('spatial_anomaly')
                                   or c.startswith('log_dist')],
    'Land Cover': [c for c in feature_cols if c.startswith('lc_')],
    'Geology': [c for c in feature_cols if c.startswith('geo_') or c.startswith('lith_') 
                or c.startswith('rock_') or c.startswith('ph_class_')],
    'Full Landsat': [c for c in feature_cols if c.startswith('ls_')],
    'EY Landsat': [c for c in feature_cols if c.startswith('ey_')],
    'Unified Landsat': [c for c in feature_cols if c.startswith('unified_')],
    'TerraClimate': [c for c in feature_cols if any(c.startswith(p) for p in 
                      ['ppt', 'tmax', 'tmin', 'aet', 'def_', 'q', 'vpd', 'pdsi', 'pet', 'srad', 'ro'])
                      and not c.startswith('soil_') and '_x_' not in c],
    'Soil': [c for c in feature_cols if c.startswith('soil_') and 'geo_' not in c],
    'Interactions': [c for c in feature_cols if '_x_' in c],
}

print(f"\n  Feature breakdown:")
accounted = set()
for group_name, cols in groups.items():
    if cols:
        marker = " ★" if "NEW" in group_name else ""
        print(f"    {group_name:35s}: {len(cols):3d} features{marker}")
        accounted.update(cols)
other = [c for c in feature_cols if c not in accounted]
if other:
    print(f"    {'Other/Derived':35s}: {len(other):3d} features")


# ============================================================
# [6] SAVE
# ============================================================
print("\n[6] Saving V9 features...")

train_out = train_df[['Latitude', 'Longitude', 'Sample Date'] + TARGETS + feature_cols]
val_out = val_df[['Latitude', 'Longitude', 'Sample Date'] + feature_cols]

train_path = os.path.join(EXTRACTED_DIR, 'v9_train_features.csv')
val_path = os.path.join(EXTRACTED_DIR, 'v9_val_features.csv')
feat_path = os.path.join(EXTRACTED_DIR, 'v9_feature_columns.txt')

train_out.to_csv(train_path, index=False)
val_out.to_csv(val_path, index=False)
with open(feat_path, 'w') as f:
    for col in feature_cols:
        f.write(col + '\n')

print(f"  Train: {train_out.shape} -> {train_path}")
print(f"  Val:   {val_out.shape} -> {val_path}")
print(f"  Features: {len(feature_cols)} -> {feat_path}")

# Quick sanity check: what do the KNN features look like?
print(f"\n  KNN feature sanity check (training data):")
for target in TARGETS:
    short = target.replace(' ', '_')
    idw_col = f'knn5_{short}_idw'
    if idw_col in train_df.columns:
        actual = train_df[target]
        predicted = train_df[idw_col]
        valid = actual.notna() & predicted.notna()
        if valid.sum() > 0:
            corr = actual[valid].corr(predicted[valid])
            print(f"    {target:40s}: IDW-actual correlation = {corr:.4f}")

print(f"\n  Distance to nearest training location (validation):")
if 'dist_nearest_km' in val_df.columns:
    d = val_df['dist_nearest_km']
    print(f"    Min:    {d.min():.2f} km")
    print(f"    Median: {d.median():.2f} km")
    print(f"    Mean:   {d.mean():.2f} km")
    print(f"    Max:    {d.max():.2f} km")
    print(f"    <10km:  {(d < 10).sum()}/{len(d)} ({100*(d < 10).mean():.1f}%)")
    print(f"    <50km:  {(d < 50).sum()}/{len(d)} ({100*(d < 50).mean():.1f}%)")

print(f"""
{'='*70}
V9 FEATURE ENGINEERING COMPLETE
{'='*70}

★ THE KEY INNOVATION: KNN Target Features
  For each sample, we now include:
  - Mean/IDW water quality at 3, 5, 10, 20 nearest training stations
  - Distance to nearest training station
  - Spatial heterogeneity (std/range of neighbors)
  - KNN × climate interactions

  This gives the model an "anchor" — instead of predicting from
  scratch, it starts from "nearby rivers have EC≈X" and adjusts.

Next: Run python scripts/21_model_v9.py
{'='*70}
""")

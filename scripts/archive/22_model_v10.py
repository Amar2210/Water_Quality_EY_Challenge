"""
============================================================
EY Water Quality Challenge - V10 Model
============================================================
PURE KNN-IDW BASELINE — ZERO ML

V9 showed that pure KNN-IDW baseline R²=0.31 (leave-one-location-out)
was better than any ML model. The ML residual models and ensembles
only degraded performance.

V10 Strategy:
  1. Pure KNN-IDW interpolation — NO ML adjustment
  2. Tune K and IDW power per target
  3. Use leave-one-location-out CV (matching V9's evaluation)
     plus spatial GroupKFold as secondary check
  4. Try mean vs median aggregation

Uses V9 feature data (has KNN columns already computed) for
validation predictions. Recomputes KNN with tuned params.
============================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.spatial import cKDTree
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, '..', 'ey_challenge_notebooks')
EXTRACTED_DIR = os.path.join(BASE_DIR, '..', 'data', 'extracted')
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
EY WATER QUALITY CHALLENGE - V10 MODEL
Pure KNN-IDW Baseline — Zero ML
{'='*70}
""")

# ============================================================
# [1] LOAD RAW DATA
# ============================================================
print("[1] Loading data...")

wq_train = pd.read_csv(os.path.join(DATA_DIR, 'water_quality_training_dataset.csv'))
submission_template = pd.read_csv(os.path.join(DATA_DIR, 'submission_template.csv'))

print(f"  Training samples: {len(wq_train)}")
print(f"  Validation samples: {len(submission_template)}")

wq_train['Latitude'] = wq_train['Latitude'].round(6)
wq_train['Longitude'] = wq_train['Longitude'].round(6)

# ============================================================
# [2] BUILD LOCATION-LEVEL STATISTICS
# ============================================================
print("\n[2] Building per-location target statistics...")

loc_stats = wq_train.groupby(['Latitude', 'Longitude']).agg({
    'Total Alkalinity': ['mean', 'median'],
    'Electrical Conductance': ['mean', 'median'],
    'Dissolved Reactive Phosphorus': ['mean', 'median'],
}).reset_index()

loc_stats.columns = ['_'.join(c).strip('_') for c in loc_stats.columns]
loc_stats.rename(columns={'Latitude_': 'Latitude', 'Longitude_': 'Longitude'}, inplace=True)

n_locs = len(loc_stats)
print(f"  Unique training locations: {n_locs}")

for target in TARGETS:
    col = f'{target}_mean'
    vals = loc_stats[col].dropna()
    print(f"  {target}: mean={vals.mean():.1f}, std={vals.std():.1f}, "
          f"min={vals.min():.1f}, max={vals.max():.1f}")

# ============================================================
# [3] KD-TREE SETUP
# ============================================================
print("\n[3] Building spatial KD-tree...")

train_lats = loc_stats['Latitude'].values
train_lons = loc_stats['Longitude'].values

mean_lat = np.mean(train_lats)
KM_PER_DEG_LAT = 111.32
KM_PER_DEG_LON = 111.32 * np.cos(np.radians(mean_lat))

train_coords_km = np.column_stack([
    train_lats * KM_PER_DEG_LAT,
    train_lons * KM_PER_DEG_LON
])

tree = cKDTree(train_coords_km)
print(f"  KD-tree built with {n_locs} locations")
print(f"  Coordinate conversion: 1° lat ≈ {KM_PER_DEG_LAT:.1f} km, "
      f"1° lon ≈ {KM_PER_DEG_LON:.1f} km")


# ============================================================
# [4] IDW PREDICTION FUNCTION
# ============================================================

def idw_predict_from_tree(query_lats, query_lons, kd_tree, loc_df,
                           target_col, k, power, exclude_locs=None):
    """
    IDW predictions using a KD-tree and location dataframe.
    
    Args:
        query_lats, query_lons: arrays of query coordinates
        kd_tree: scipy cKDTree built from loc_df coordinates
        loc_df: DataFrame with Latitude, Longitude, and target_col
        target_col: column name for target values
        k: number of neighbors
        power: IDW power
        exclude_locs: set of (lat, lon) tuples to exclude (for LOO)
    
    Returns:
        predictions array
    """
    n_tree = kd_tree.n
    preds = np.full(len(query_lats), np.nan)
    lats_arr = loc_df['Latitude'].values
    lons_arr = loc_df['Longitude'].values
    vals_arr = loc_df[target_col].values
    
    # Query enough neighbors to handle exclusions
    max_query = min(k + 10, n_tree)
    
    for i in range(len(query_lats)):
        q_km = np.array([
            query_lats[i] * KM_PER_DEG_LAT,
            query_lons[i] * KM_PER_DEG_LON
        ])
        
        dists, idxs = kd_tree.query(q_km, k=max_query)
        dists = np.atleast_1d(dists)
        idxs = np.atleast_1d(idxs)
        
        vals = []
        ds = []
        for d, ix in zip(dists, idxs):
            if ix >= n_tree:
                continue
            
            # Skip excluded locations (for leave-one-out)
            if exclude_locs is not None:
                loc_key = (round(lats_arr[ix], 4), round(lons_arr[ix], 4))
                if loc_key in exclude_locs:
                    continue
            
            v = vals_arr[ix]
            if not np.isnan(v):
                vals.append(v)
                ds.append(max(d, 0.01))
                if len(vals) >= k:
                    break
        
        if vals:
            v_arr = np.array(vals)
            d_arr = np.array(ds)
            w = 1.0 / (d_arr ** power)
            preds[i] = np.sum(v_arr * w) / np.sum(w)
    
    return preds


# ============================================================
# [5] TUNE K AND POWER — LEAVE-ONE-LOCATION-OUT CV
# ============================================================
# This matches how V9 computed its "Pure KNN-IDW baseline R²: +0.31"
# For each training SAMPLE, predict using KNN from all OTHER locations
# (exclude the sample's own location from the neighbor set)

print("\n" + "=" * 70)
print("[5] TUNING K AND POWER — Leave-One-Location-Out CV")
print("    (Matches V9's KNN baseline evaluation method)")
print("=" * 70)

K_VALUES = [3, 5, 7, 10, 15, 20, 30]
POWER_VALUES = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
AGG_METHODS = ['mean', 'median']

# Assign each training sample to its location
wq_train['loc_key'] = list(zip(
    wq_train['Latitude'].round(4),
    wq_train['Longitude'].round(4)
))

# Build full KD-tree from all locations (for LOO, we query all and skip self)
# We need a mapping from loc_key to location index
loc_stats['loc_key'] = list(zip(
    loc_stats['Latitude'].round(4),
    loc_stats['Longitude'].round(4)
))

best_params = {}

for target in TARGETS:
    print(f"\n  {'─'*50}")
    print(f"  TARGET: {target}")
    print(f"  {'─'*50}")
    
    y = wq_train[target].values
    valid_mask = ~np.isnan(y)
    y_valid = y[valid_mask]
    lats_valid = wq_train['Latitude'].values[valid_mask]
    lons_valid = wq_train['Longitude'].values[valid_mask]
    locs_valid = wq_train['loc_key'].values[valid_mask]
    
    best_r2 = -999
    best_cfg = {}
    all_results = []
    
    print(f"\n    {'Config':30s} {'LOO R²':>10s}")
    print(f"    {'─'*45}")
    
    for agg in AGG_METHODS:
        target_col = f'{target}_{agg}'
        
        for k in K_VALUES:
            for power in POWER_VALUES:
                # Leave-one-location-out: for each sample, exclude its location
                preds = np.full(len(y_valid), np.nan)
                
                for i in range(len(y_valid)):
                    q_km = np.array([
                        lats_valid[i] * KM_PER_DEG_LAT,
                        lons_valid[i] * KM_PER_DEG_LON
                    ])
                    
                    # Query enough neighbors
                    max_q = min(k + 10, n_locs)
                    dists, idxs = tree.query(q_km, k=max_q)
                    dists = np.atleast_1d(dists)
                    idxs = np.atleast_1d(idxs)
                    
                    sample_loc = locs_valid[i]
                    
                    vals = []
                    ds = []
                    for d, ix in zip(dists, idxs):
                        if ix >= n_locs:
                            continue
                        # Skip own location
                        if loc_stats['loc_key'].iloc[ix] == sample_loc:
                            continue
                        
                        v = loc_stats.iloc[ix][target_col]
                        if not np.isnan(v):
                            vals.append(v)
                            ds.append(max(d, 0.01))
                            if len(vals) >= k:
                                break
                    
                    if vals:
                        v_arr = np.array(vals)
                        d_arr = np.array(ds)
                        w = 1.0 / (d_arr ** power)
                        preds[i] = np.sum(v_arr * w) / np.sum(w)
                
                valid_pred = ~np.isnan(preds)
                if valid_pred.sum() > 100:
                    loo_r2 = r2_score(y_valid[valid_pred], preds[valid_pred])
                else:
                    loo_r2 = np.nan
                
                config = f"{agg} K={k:2d}, p={power:.1f}"
                marker = '✓' if loo_r2 > 0 else '✗'
                print(f"    {marker} {config:30s} {loo_r2:+.4f}")
                
                all_results.append({
                    'agg': agg, 'k': k, 'power': power, 'r2': loo_r2
                })
                
                if loo_r2 > best_r2:
                    best_r2 = loo_r2
                    best_cfg = {'agg': agg, 'k': k, 'power': power, 'loo_r2': best_r2}
    
    best_params[target] = best_cfg
    
    # Show top 5
    all_results.sort(key=lambda x: x['r2'] if not np.isnan(x['r2']) else -999, reverse=True)
    print(f"\n    Top 5 configs:")
    for r in all_results[:5]:
        print(f"      {r['agg']:6s} K={r['k']:2d} p={r['power']:.1f} → LOO R²={r['r2']:+.4f}")
    
    print(f"\n    ★ BEST: {best_cfg['agg']} K={best_cfg['k']}, "
          f"p={best_cfg['power']:.1f}, LOO R² = {best_cfg['loo_r2']:+.4f}")


# ============================================================
# [6] SPATIAL GROUPKFOLD CV (secondary validation)
# ============================================================
print("\n" + "=" * 70)
print("[6] SPATIAL GROUPKFOLD CV (secondary check)")
print("    Using best params from LOO tuning")
print("=" * 70)

coords = wq_train[['Latitude', 'Longitude']].values
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
wq_train['spatial_cluster'] = kmeans.fit_predict(coords)
groups = wq_train['spatial_cluster'].values

for target in TARGETS:
    print(f"\n  {target}:")
    bp = best_params[target]
    
    y = wq_train[target].values
    valid_mask = ~np.isnan(y)
    y_v = y[valid_mask]
    lats_v = wq_train['Latitude'].values[valid_mask]
    lons_v = wq_train['Longitude'].values[valid_mask]
    groups_v = groups[valid_mask]
    
    gkf = GroupKFold(n_splits=N_CLUSTERS)
    fold_scores = []
    
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(lats_v, y_v, groups_v)):
        # Build fold-specific location stats
        fold_df = wq_train.iloc[np.where(valid_mask)[0][tr_idx]]
        if bp['agg'] == 'mean':
            fold_loc = fold_df.groupby(['Latitude','Longitude'])[target].mean().reset_index()
        else:
            fold_loc = fold_df.groupby(['Latitude','Longitude'])[target].median().reset_index()
        fold_loc.rename(columns={target: 'target_val'}, inplace=True)
        
        # Build fold KD-tree
        f_coords = np.column_stack([
            fold_loc['Latitude'].values * KM_PER_DEG_LAT,
            fold_loc['Longitude'].values * KM_PER_DEG_LON
        ])
        f_tree = cKDTree(f_coords)
        f_vals = fold_loc['target_val'].values
        n_fold_locs = len(fold_loc)
        
        # Predict test fold
        test_preds = np.full(len(te_idx), np.nan)
        for j in range(len(te_idx)):
            q_km = np.array([
                lats_v[te_idx[j]] * KM_PER_DEG_LAT,
                lons_v[te_idx[j]] * KM_PER_DEG_LON
            ])
            max_q = min(bp['k'] + 5, n_fold_locs)
            if max_q < 1:
                continue
            dists, idxs = f_tree.query(q_km, k=max_q)
            dists = np.atleast_1d(dists)
            idxs = np.atleast_1d(idxs)
            
            vals, ds = [], []
            for d, ix in zip(dists, idxs):
                if ix < n_fold_locs:
                    v = f_vals[ix]
                    if not np.isnan(v):
                        vals.append(v)
                        ds.append(max(d, 0.01))
                        if len(vals) >= bp['k']:
                            break
            
            if vals:
                v_arr, d_arr = np.array(vals), np.array(ds)
                w = 1.0 / (d_arr ** bp['power'])
                test_preds[j] = np.sum(v_arr * w) / np.sum(w)
        
        vp = ~np.isnan(test_preds)
        if vp.sum() > 10:
            fold_scores.append(r2_score(y_v[te_idx][vp], test_preds[vp]))
        else:
            fold_scores.append(np.nan)
    
    spatial_r2 = np.nanmean(fold_scores)
    fold_str = ', '.join(f'{s:+.3f}' for s in fold_scores)
    marker = '✓' if spatial_r2 > 0 else '✗'
    print(f"    {marker} Spatial CV R² = {spatial_r2:+.4f}  [{fold_str}]")
    print(f"    (LOO R² was {bp['loo_r2']:+.4f})")
    best_params[target]['spatial_cv_r2'] = spatial_r2


# ============================================================
# [7] GENERATE FINAL PREDICTIONS
# ============================================================
print("\n" + "=" * 70)
print("[7] GENERATING FINAL PREDICTIONS")
print("=" * 70)

val_lats = submission_template['Latitude'].values
val_lons = submission_template['Longitude'].values

final_predictions = {}

for target in TARGETS:
    print(f"\n  {target}:")
    bp = best_params[target]
    target_col = f'{target}_{bp["agg"]}'
    
    print(f"    Using {bp['agg']} K={bp['k']}, power={bp['power']:.1f} "
          f"(LOO R² = {bp['loo_r2']:+.4f})")
    
    # Predict using full location set (no exclusions for validation)
    preds = np.full(len(val_lats), np.nan)
    for i in range(len(val_lats)):
        q_km = np.array([
            val_lats[i] * KM_PER_DEG_LAT,
            val_lons[i] * KM_PER_DEG_LON
        ])
        max_q = min(bp['k'] + 5, n_locs)
        dists, idxs = tree.query(q_km, k=max_q)
        dists = np.atleast_1d(dists)
        idxs = np.atleast_1d(idxs)
        
        vals, ds = [], []
        for d, ix in zip(dists, idxs):
            if ix < n_locs:
                v = loc_stats.iloc[ix][target_col]
                if not np.isnan(v):
                    vals.append(v)
                    ds.append(max(d, 0.01))
                    if len(vals) >= bp['k']:
                        break
        
        if vals:
            v_arr, d_arr = np.array(vals), np.array(ds)
            w = 1.0 / (d_arr ** bp['power'])
            preds[i] = np.sum(v_arr * w) / np.sum(w)
        
        if (i + 1) % 50 == 0 or (i + 1) == len(val_lats):
            print(f"      {i+1}/{len(val_lats)}")
    
    # Fill NaN
    n_nan = np.isnan(preds).sum()
    if n_nan > 0:
        fill = loc_stats[target_col].mean()
        print(f"    WARNING: {n_nan} NaN predictions, filling with {fill:.1f}")
        preds[np.isnan(preds)] = fill
    
    # Clip to physical ranges
    if target == 'Total Alkalinity':
        preds = np.clip(preds, 0, 1000)
    elif target == 'Electrical Conductance':
        preds = np.clip(preds, 0, 10000)
    elif target == 'Dissolved Reactive Phosphorus':
        preds = np.clip(preds, 0, 5000)
    
    final_predictions[target] = preds
    print(f"    Range: [{preds.min():.1f}, {preds.max():.1f}], mean={preds.mean():.1f}")

# Distance stats
print("\n  Validation distance to nearest training station:")
all_dists = []
for i in range(len(val_lats)):
    q_km = np.array([val_lats[i] * KM_PER_DEG_LAT, val_lons[i] * KM_PER_DEG_LON])
    d, _ = tree.query(q_km, k=1)
    all_dists.append(d)
all_dists = np.array(all_dists)
print(f"    Min:    {all_dists.min():.1f} km")
print(f"    Median: {np.median(all_dists):.1f} km")
print(f"    Max:    {all_dists.max():.1f} km")


# ============================================================
# [8] GENERATE SUBMISSION
# ============================================================
print("\n" + "=" * 70)
print("[8] GENERATING SUBMISSION")
print("=" * 70)

submission = submission_template.copy()
for target in TARGETS:
    submission[target] = final_predictions[target]

for target in TARGETS:
    n_nan = submission[target].isna().sum()
    if n_nan > 0:
        print(f"  ERROR: {target} has {n_nan} NaN values!")

sub_path = os.path.join(SUBMISSION_DIR, 'submission_v10.csv')
sub_ts_path = os.path.join(SUBMISSION_DIR, f'submission_v10_{ts}.csv')
submission.to_csv(sub_path, index=False)
submission.to_csv(sub_ts_path, index=False)
print(f"  Saved: {sub_path}")
print(f"  Saved: {sub_ts_path}")


# ============================================================
# [9] SUMMARY
# ============================================================
print(f"""

{'='*70}
V10 MODEL RESULTS SUMMARY
{'='*70}

  Pure KNN-IDW — NO ML, NO RESIDUALS, NO ENSEMBLE
""")

for target in TARGETS:
    bp = best_params[target]
    preds = final_predictions[target]
    marker = '✓' if bp['loo_r2'] > 0 else '✗'
    print(f"  {marker} {target}")
    print(f"    Config:     {bp['agg']} K={bp['k']}, power={bp['power']:.1f}")
    print(f"    LOO R²:     {bp['loo_r2']:+.4f}")
    print(f"    Spatial CV: {bp.get('spatial_cv_r2', 'N/A')}")
    print(f"    Predictions: [{preds.min():.1f}, {preds.max():.1f}], mean={preds.mean():.1f}")

print(f"""
{'─'*70}
V9 → V10 CHANGES:
  V9: KNN-IDW + XGBoost residuals + ensemble → score 0.2
  V10: PURE KNN-IDW with tuned K and power

  ★ Removed ALL ML models
  ★ Removed residual learning
  ★ Removed ensemble blending
  ★ Added per-target K, power, and aggregation tuning
  ★ LOO CV matches V9's baseline evaluation

  LOO R² measures in-sample generalization (nearby locations).
  Spatial CV R² measures cross-region generalization.
  The actual leaderboard score will be somewhere between these.

Submit: {sub_path}
Next: python scripts/23_model_v11.py
{'='*70}
""")

# Save log
log_data = {'version': 'V10', 'timestamp': ts}
for target in TARGETS:
    bp = best_params[target]
    log_data[f'{target}_loo_r2'] = bp['loo_r2']
    log_data[f'{target}_spatial_cv_r2'] = bp.get('spatial_cv_r2', np.nan)
    log_data[f'{target}_k'] = bp['k']
    log_data[f'{target}_power'] = bp['power']
    log_data[f'{target}_agg'] = bp['agg']
pd.DataFrame([log_data]).to_csv(os.path.join(LOG_DIR, f'v10_results_{ts}.csv'), index=False)
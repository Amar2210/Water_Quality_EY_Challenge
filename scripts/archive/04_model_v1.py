"""
============================================================
EY Water Quality Challenge - V1 Model
============================================================
Merges ALL extracted features:
  - Landsat spectral bands + indices (original)
  - TerraClimate EXPANDED (50 climate features with lags)
  - Elevation
  - Soil properties (pH, clay, sand, organic carbon, CEC, nitrogen)
  + Engineered interactions and ratios

Trains separate XGBoost models per target with:
  - Spatial cross-validation (GroupKFold by location clusters)
  - Multiple hyperparameter configs
  - Log transform option
  - Feature importance analysis

Expected: Major improvement over V0.1 (which had negative spatial CV)
============================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
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

print("=" * 70)
print("EY WATER QUALITY CHALLENGE - V1 MODEL")
print("All extracted features: Landsat + TerraClimate + Elevation + Soil")
print("=" * 70)

# ============================================================
# [1] LOAD ALL DATA SOURCES
# ============================================================
print("\n[1] Loading all data sources...")

# --- Targets ---
wq_train = pd.read_csv(os.path.join(DATA_DIR, 'water_quality_training_dataset.csv'))
submission_template = pd.read_csv(os.path.join(DATA_DIR, 'submission_template.csv'))

# --- Landsat (original) ---
landsat_train = pd.read_csv(os.path.join(DATA_DIR, 'landsat_features_training.csv'))
landsat_val = pd.read_csv(os.path.join(DATA_DIR, 'landsat_features_validation.csv'))

# --- TerraClimate EXPANDED (our new extraction) ---
terra_exp_train = pd.read_csv(os.path.join(EXTRACTED_DIR, 'terraclimate_expanded_training.csv'))
terra_exp_val = pd.read_csv(os.path.join(EXTRACTED_DIR, 'terraclimate_expanded_validation.csv'))

# --- Elevation ---
elevation = pd.read_csv(os.path.join(EXTRACTED_DIR, 'elevation_data.csv'))

# --- Soil ---
soil = pd.read_csv(os.path.join(EXTRACTED_DIR, 'soil_data.csv'))

print(f"  Water quality training: {wq_train.shape}")
print(f"  Landsat train/val: {landsat_train.shape} / {landsat_val.shape}")
print(f"  TerraClimate expanded train/val: {terra_exp_train.shape} / {terra_exp_val.shape}")
print(f"  Elevation: {elevation.shape}")
print(f"  Soil: {soil.shape}")

# ============================================================
# [2] MERGE EVERYTHING
# ============================================================
print("\n[2] Merging all data sources...")

join_keys = ['Latitude', 'Longitude', 'Sample Date']
loc_keys = ['Latitude', 'Longitude']

# --- Training set ---
train_df = wq_train.copy()
train_df = train_df.merge(landsat_train, on=join_keys, how='left')
train_df = train_df.merge(terra_exp_train, on=join_keys, how='left')
train_df = train_df.merge(elevation, on=loc_keys, how='left')
train_df = train_df.merge(soil, on=loc_keys, how='left')

# --- Validation set ---
val_df = submission_template[['Latitude', 'Longitude', 'Sample Date']].copy()
val_df = val_df.merge(landsat_val, on=join_keys, how='left')
val_df = val_df.merge(terra_exp_val, on=join_keys, how='left')
val_df = val_df.merge(elevation, on=loc_keys, how='left')
val_df = val_df.merge(soil, on=loc_keys, how='left')

print(f"  Train after merge: {train_df.shape}")
print(f"  Val after merge: {val_df.shape}")

for col in ['ppt', 'elevation_m', 'soil_clay']:
    if col in train_df.columns:
        valid = train_df[col].notna().sum()
        print(f"  {col}: {valid}/{len(train_df)} valid in train")

# ============================================================
# [3] FEATURE ENGINEERING
# ============================================================
print("\n[3] Feature engineering...")

def engineer_features(df):
    df = df.copy()
    
    # ── Temporal features ──
    df['parsed_date'] = pd.to_datetime(df['Sample Date'], dayfirst=True, errors='coerce')
    df['month'] = df['parsed_date'].dt.month
    df['year'] = df['parsed_date'].dt.year
    df['day_of_year'] = df['parsed_date'].dt.dayofyear
    
    season_map = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
                  6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    df['season'] = df['month'].map(season_map)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # ── Landsat-derived indices ──
    eps = 1e-10
    if 'nir' in df.columns and 'green' in df.columns:
        df['nir_green_ratio'] = df['nir'] / (df['green'] + eps)
        df['pseudo_ndvi'] = (df['nir'] - df['green']) / (df['nir'] + df['green'] + eps)
        df['brightness'] = (df['nir'] + df['green'] + df.get('swir16', 0) + df.get('swir22', 0)) / 4
    
    if 'swir16' in df.columns and 'swir22' in df.columns:
        df['swir_ratio'] = df['swir16'] / (df['swir22'] + eps)
        df['green_swir22_ratio'] = df['green'] / (df['swir22'] + eps)
        df['nir_swir16_ratio'] = df['nir'] / (df['swir16'] + eps)
        df['bsi'] = ((df['swir16'] + df['green']) - (df['nir'] + df['swir22'])) / \
                    ((df['swir16'] + df['green']) + (df['nir'] + df['swir22']) + eps)
    
    # ── Climate-derived features ──
    if 'tmax' in df.columns and 'tmin' in df.columns:
        df['temp_range'] = df['tmax'] - df['tmin']
        df['temp_mean'] = (df['tmax'] + df['tmin']) / 2
    
    if 'pet' in df.columns and 'ppt' in df.columns:
        df['aridity_index'] = df['pet'] / df['ppt'].replace(0, np.nan)
        df['moisture_surplus'] = df['ppt'] - df['pet']
    
    if 'aet' in df.columns and 'pet' in df.columns:
        df['aet_pet_ratio'] = df['aet'] / (df['pet'] + eps)
    
    if 'ppt' in df.columns and 'ppt_lag1' in df.columns:
        df['ppt_change'] = df['ppt'] - df['ppt_lag1']
    
    if 'q' in df.columns and 'ppt' in df.columns:
        df['runoff_ratio'] = df['q'] / (df['ppt'] + eps)
    
    # ── Soil-derived features ──
    if 'soil_clay' in df.columns and 'soil_sand' in df.columns:
        df['clay_sand_ratio'] = df['soil_clay'] / (df['soil_sand'] + eps)
    
    if 'soil_soc' in df.columns and 'soil_clay' in df.columns:
        df['soc_clay_ratio'] = df['soil_soc'] / (df['soil_clay'] + eps)
    
    # ── Geographic interactions ──
    if 'elevation_m' in df.columns:
        df['elev_x_ppt'] = df['elevation_m'] * df.get('ppt', 0)
        df['elev_x_temp'] = df['elevation_m'] * df.get('tmax', 0)
    
    if 'Latitude' in df.columns:
        df['lat_x_ppt'] = df['Latitude'] * df.get('ppt', 0)
        df['lat_x_pet'] = df['Latitude'] * df.get('pet', 0)
        df['lon_x_ppt'] = df['Longitude'] * df.get('ppt', 0)
        df['lat_x_month'] = df['Latitude'] * df['month']
        df['lat_abs'] = df['Latitude'].abs()
    
    # ── Spectral × Climate interactions ──
    if 'NDMI' in df.columns and 'ppt' in df.columns:
        df['ndmi_x_ppt'] = df['NDMI'] * df['ppt']
    if 'MNDWI' in df.columns and 'soil' in df.columns:
        df['mndwi_x_soilmoist'] = df['MNDWI'] * df['soil']
    
    df = df.drop(columns=['parsed_date', 'Sample Date'], errors='ignore')
    return df

train_featured = engineer_features(train_df)
val_featured = engineer_features(val_df)

feature_cols = [c for c in train_featured.columns if c not in TARGETS]
print(f"  Total features: {len(feature_cols)}")

groups = {
    'Landsat': [c for c in feature_cols if c in ['nir','green','swir16','swir22','NDMI','MNDWI'] or 
                c.endswith('ratio') or c in ['brightness','pseudo_ndvi','bsi']],
    'TerraClimate current': [c for c in feature_cols if c in ['ppt','tmax','tmin','soil','aet','def','q','vpd','pdsi','pet','srad']],
    'TerraClimate lagged': [c for c in feature_cols if '_lag' in c],
    'TerraClimate rolling': [c for c in feature_cols if '_3m_' in c],
    'Elevation': [c for c in feature_cols if 'elev' in c.lower()],
    'Soil': [c for c in feature_cols if 'soil_' in c],
    'Temporal': [c for c in feature_cols if c in ['month','year','day_of_year','season','month_sin','month_cos']],
    'Interactions': [c for c in feature_cols if '_x_' in c],
}
for name, cols in groups.items():
    if cols:
        print(f"    {name}: {len(cols)} features")

# ============================================================
# [4] HANDLE MISSING VALUES
# ============================================================
print("\n[4] Handling missing values...")

train_featured = train_featured.replace([np.inf, -np.inf], np.nan)
val_featured = val_featured.replace([np.inf, -np.inf], np.nan)

missing_train = train_featured[feature_cols].isnull().sum()
cols_with_missing = missing_train[missing_train > 0]
if len(cols_with_missing) > 0:
    print(f"  Columns with missing values ({len(cols_with_missing)}):")
    for col, cnt in cols_with_missing.head(10).items():
        print(f"    {col}: {cnt}/{len(train_featured)} missing ({cnt/len(train_featured)*100:.1f}%)")
    if len(cols_with_missing) > 10:
        print(f"    ... and {len(cols_with_missing) - 10} more")

print("  XGBoost handles NaN natively — no imputation needed")

# ============================================================
# [5] SPATIAL CROSS-VALIDATION
# ============================================================
print("\n[5] Setting up spatial cross-validation...")

train_coords = train_featured[['Latitude', 'Longitude']].values
kmeans = KMeans(n_clusters=10, random_state=RANDOM_STATE, n_init=10)
spatial_groups = kmeans.fit_predict(train_coords)

n_unique_groups = len(np.unique(spatial_groups))
print(f"  Spatial clusters: {n_unique_groups}")
for g in range(n_unique_groups):
    cnt = (spatial_groups == g).sum()
    print(f"    Cluster {g}: {cnt} samples")

# ============================================================
# [6] TRAIN MODELS
# ============================================================
print("\n[6] Training models...")

configs = {
    'conservative': {
        'n_estimators': 500,
        'max_depth': 4,
        'learning_rate': 0.03,
        'subsample': 0.7,
        'colsample_bytree': 0.6,
        'min_child_weight': 20,
        'reg_alpha': 2.0,
        'reg_lambda': 5.0,
        'gamma': 0.1,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbosity': 0,
    },
    'moderate': {
        'n_estimators': 800,
        'max_depth': 5,
        'learning_rate': 0.03,
        'subsample': 0.75,
        'colsample_bytree': 0.7,
        'min_child_weight': 10,
        'reg_alpha': 1.0,
        'reg_lambda': 3.0,
        'gamma': 0.05,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbosity': 0,
    },
    'aggressive': {
        'n_estimators': 1200,
        'max_depth': 6,
        'learning_rate': 0.02,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'reg_alpha': 0.5,
        'reg_lambda': 2.0,
        'gamma': 0.0,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbosity': 0,
    },
}

X_train_all = train_featured[feature_cols]
X_val = val_featured[feature_cols]

best_models = {}
best_scores = {}
all_results = []

for target in TARGETS:
    print(f"\n  {'═' * 60}")
    print(f"  TARGET: {target}")
    print(f"  {'═' * 60}")
    
    y_raw = train_featured[target].copy()
    
    valid_mask = y_raw.notna()
    if valid_mask.sum() < len(y_raw):
        print(f"    WARNING: {(~valid_mask).sum()} NaN targets — dropping them")
    
    best_r2 = -999
    best_config_name = ""
    best_model = None
    best_log = False
    
    log_options = [False, True]
    
    for config_name, params in configs.items():
        for use_log in log_options:
            label = f"{config_name} {'+ log' if use_log else '+ raw'}"
            
            if use_log:
                if (y_raw[valid_mask] < 0).any():
                    continue
                y = np.log1p(y_raw)
            else:
                y = y_raw.copy()
            
            gkf = GroupKFold(n_splits=min(5, n_unique_groups))
            fold_scores = []
            
            X_cv = X_train_all[valid_mask]
            y_cv = y[valid_mask]
            groups_cv = spatial_groups[valid_mask]
            
            for fold, (train_idx, val_idx) in enumerate(gkf.split(X_cv, y_cv, groups=groups_cv)):
                X_tr = X_cv.iloc[train_idx]
                y_tr = y_cv.iloc[train_idx]
                X_va = X_cv.iloc[val_idx]
                y_va = y_cv.iloc[val_idx]
                
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
            print(f"    {label:35s} → R² = {mean_r2:.4f} (±{std_r2:.4f})")
            
            all_results.append({
                'target': target, 'config': label,
                'mean_r2': mean_r2, 'std_r2': std_r2,
                'fold_scores': fold_scores,
            })
            
            if mean_r2 > best_r2:
                best_r2 = mean_r2
                best_config_name = label
                best_log = use_log
                
                y_full = np.log1p(y_raw[valid_mask]) if use_log else y_raw[valid_mask]
                best_model = xgb.XGBRegressor(**params)
                best_model.fit(X_train_all[valid_mask], y_full)
    
    print(f"\n    ★ BEST: {best_config_name} → R² = {best_r2:.4f}")
    best_models[target] = {'model': best_model, 'use_log': best_log}
    best_scores[target] = best_r2
    
    imps = best_model.feature_importances_
    feat_imp = sorted(zip(feature_cols, imps), key=lambda x: x[1], reverse=True)
    print(f"\n    Top 15 features:")
    for rank, (fn, fi) in enumerate(feat_imp[:15], 1):
        bar = '█' * int(fi * 100)
        print(f"      {rank:2d}. {fn:30s} {fi:.4f} {bar}")

# ============================================================
# [7] SUMMARY
# ============================================================
print(f"\n{'=' * 70}")
print(f"V1 MODEL — SPATIAL CV RESULTS")
print(f"{'=' * 70}")
for t, s in best_scores.items():
    status = "✓" if s > 0 else "✗"
    print(f"  {status} {t:40s}: R² = {s:.4f}")
mean_r2 = np.mean(list(best_scores.values()))
print(f"  {'─' * 55}")
print(f"    {'MEAN R²':40s}: {mean_r2:.4f}")

if mean_r2 > 0:
    print(f"\n  ► IMPROVEMENT over V0.1 baseline!")
else:
    print(f"\n  ► Still struggling with spatial generalization.")

# ============================================================
# [8] GENERATE SUBMISSION
# ============================================================
print(f"\n[8] Generating submission...")

submission = submission_template.copy()
for target in TARGETS:
    m = best_models[target]
    preds = m['model'].predict(X_val)
    if m['use_log']:
        preds = np.expm1(preds)
    preds = np.maximum(preds, 0)
    submission[target] = preds
    print(f"  {target}:")
    print(f"    Range: [{preds.min():.2f}, {preds.max():.2f}], Mean: {preds.mean():.2f}")

ts = time.strftime("%Y%m%d_%H%M%S")
sub_path = os.path.join(SUBMISSION_DIR, f'submission_v1_{ts}.csv')
sub_path_latest = os.path.join(SUBMISSION_DIR, 'submission_v1.csv')
submission.to_csv(sub_path, index=False)
submission.to_csv(sub_path_latest, index=False)

print(f"\n  Saved: {sub_path}")
print(f"  Saved: {sub_path_latest}")

print(f"\n  Preview:")
print(submission.head(5).to_string())

print(f"""
{'=' * 70}
WHAT'S NEXT (if R² still needs improvement)
{'=' * 70}

V1 added ~60 features across 4 data sources. If spatial CV R² is:

  > 0.3  : Good foundation — tune hyperparameters, try LightGBM
  > 0.5  : Strong — try stacking, target-specific feature selection
  > 0.7  : Excellent — focus on DRP (usually hardest), fine-tune
  < 0.0  : The model overfits to training geography — need:
           - Stronger regularization (reduce max_depth to 3)
           - Remove Lat/Lon as features (they leak geography)
           - Use only climate + soil (geography-invariant features)
           - Try separate models for each geographic cluster

Priority improvements for V2:
  1. Target-specific feature selection (remove noisy features per target)
  2. LightGBM / CatBoost ensemble
  3. Stacking with Ridge as meta-learner
  4. Hyperparameter optimization with Optuna
{'=' * 70}
""")

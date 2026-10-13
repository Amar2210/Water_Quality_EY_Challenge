"""
============================================================
EY Water Quality Challenge - V2 Model
============================================================
KEY CHANGES FROM V1:
  V1 Problem: Negative spatial CV because Lat/Lon dominated 
  feature importance — model memorized coordinates instead of
  learning transferable climate/soil/spectral relationships.

  V2 Strategy:
  1. DROP all coordinate features (Lat, Lon, lat_abs, _x_ coords)
  2. KEEP only geography-INVARIANT features:
     - Climate conditions (transferable physics)
     - Soil properties (characterize location without coordinates)
     - Elevation (physical, not coordinate-based)
     - Spectral indices (reflect actual water conditions)
     - Temporal patterns (seasonality)
  3. ADD location "fingerprint" from climate normals
  4. STRONGER regularization (max_depth=3, high penalties)
  5. TRY Ridge regression alongside XGBoost (often better for
     spatial generalization because it can't memorize)
  6. ENSEMBLE best models per target

Expected: Spatial CV should go from negative to positive
============================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.pipeline import Pipeline
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
print("EY WATER QUALITY CHALLENGE - V2 MODEL")
print("Strategy: Geography-invariant features + strong regularization")
print("=" * 70)

# ============================================================
# [1] LOAD & MERGE (same as V1)
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
# [2] FEATURE ENGINEERING (NO COORDINATES)
# ============================================================
print("\n[2] Feature engineering (geography-invariant only)...")

def engineer_features_v2(df):
    """
    Create features that are TRANSFERABLE across geographies.
    NO Latitude, Longitude, or coordinate-based interactions.
    """
    df = df.copy()
    
    # ── Temporal ──
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
    
    # ── Landsat spectral indices ──
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
    
    # ── Climate-derived (transferable physics) ──
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
    
    # ── Soil-derived (transferable — characterize location) ──
    if 'soil_clay' in df.columns and 'soil_sand' in df.columns:
        df['clay_sand_ratio'] = df['soil_clay'] / (df['soil_sand'] + eps)
        df['clay_fraction'] = df['soil_clay'] / (df['soil_clay'] + df['soil_sand'] + eps)
    
    if 'soil_soc' in df.columns:
        df['soc_clay_ratio'] = df['soil_soc'] / (df['soil_clay'] + eps)
    
    if 'soil_phh2o' in df.columns:
        # pH is a key driver of alkalinity
        df['soil_ph_centered'] = df['soil_phh2o'] - 70  # Center around neutral
    
    # ── Elevation × Climate interactions (transferable!) ──
    # "High elevation + heavy rain" means different things than 
    # "low elevation + heavy rain" for water quality
    if 'elevation_m' in df.columns:
        df['elev_x_ppt'] = df['elevation_m'] * df.get('ppt', 0)
        df['elev_x_temp'] = df['elevation_m'] * df.get('temp_mean', df.get('tmax', 0))
        df['elev_x_runoff'] = df['elevation_m'] * df.get('q', 0)
    
    # ── Soil × Climate interactions (transferable!) ──
    # "Clay soil + heavy rain" = more phosphorus runoff
    if 'soil_clay' in df.columns and 'ppt' in df.columns:
        df['clay_x_ppt'] = df['soil_clay'] * df['ppt']
    if 'soil_soc' in df.columns and 'ppt' in df.columns:
        df['soc_x_ppt'] = df['soil_soc'] * df['ppt']
    
    # ── Spectral × Climate interactions ──
    if 'NDMI' in df.columns and 'ppt' in df.columns:
        df['ndmi_x_ppt'] = df['NDMI'] * df['ppt']
    if 'MNDWI' in df.columns and 'soil' in df.columns:
        df['mndwi_x_soilmoist'] = df['MNDWI'] * df['soil']
    
    # ── Location "fingerprint" from climate normals ──
    # Instead of coordinates, describe the location by its climate PROPERTIES
    # The 3-month averages already serve as a proxy for "what climate is like here"
    # Additional: ratio of current to lag (is it wetter/drier than usual?)
    if 'ppt' in df.columns and 'ppt_3m_avg' in df.columns:
        df['ppt_vs_avg'] = df['ppt'] / (df['ppt_3m_avg'] + eps)
    if 'tmax' in df.columns and 'tmax_3m_avg' in df.columns:
        df['tmax_vs_avg'] = df['tmax'] / (df['tmax_3m_avg'] + eps)
    
    # ── DROP coordinate columns ──
    df = df.drop(columns=['parsed_date', 'Sample Date', 'Latitude', 'Longitude'], errors='ignore')
    
    return df

train_featured = engineer_features_v2(train_df)
val_featured = engineer_features_v2(val_df)

# Remove targets from feature list
feature_cols = [c for c in train_featured.columns if c not in TARGETS]
print(f"  Total features (no coordinates!): {len(feature_cols)}")

# Verify no coordinate leakage
coord_leaks = [c for c in feature_cols if 'lat' in c.lower() or 'lon' in c.lower()]
if coord_leaks:
    print(f"  WARNING: Coordinate features still present: {coord_leaks}")
    feature_cols = [c for c in feature_cols if c not in coord_leaks]
    print(f"  Removed. Features now: {len(feature_cols)}")
else:
    print(f"  ✓ No coordinate leakage detected")

# ============================================================
# [3] HANDLE MISSING VALUES
# ============================================================
print("\n[3] Handling missing values...")

train_featured = train_featured.replace([np.inf, -np.inf], np.nan)
val_featured = val_featured.replace([np.inf, -np.inf], np.nan)

missing = train_featured[feature_cols].isnull().sum()
cols_missing = missing[missing > 0].sort_values(ascending=False)
if len(cols_missing) > 0:
    print(f"  {len(cols_missing)} columns with missing values")
    for col, cnt in cols_missing.head(8).items():
        print(f"    {col}: {cnt}/{len(train_featured)} ({cnt/len(train_featured)*100:.1f}%)")

# For Ridge regression, we need to impute. For XGBoost, NaN is fine.
# We'll impute with median for Ridge, leave NaN for XGBoost.

# ============================================================
# [4] SPATIAL CV SETUP
# ============================================================
print("\n[4] Spatial cross-validation setup...")

# Use ORIGINAL coordinates for spatial grouping (not as features!)
train_coords = train_df[['Latitude', 'Longitude']].values
kmeans = KMeans(n_clusters=10, random_state=RANDOM_STATE, n_init=10)
spatial_groups = kmeans.fit_predict(train_coords)
n_groups = len(np.unique(spatial_groups))
print(f"  Spatial clusters: {n_groups}")

# ============================================================
# [5] TRAIN MODELS — Multiple approaches per target
# ============================================================
print("\n[5] Training models...")

X_train = train_featured[feature_cols]
X_val = val_featured[feature_cols]

# XGBoost configs — much more conservative than V1
xgb_configs = {
    'very_conservative': {
        'n_estimators': 300,
        'max_depth': 3,          # Shallow!
        'learning_rate': 0.03,
        'subsample': 0.6,
        'colsample_bytree': 0.5,
        'min_child_weight': 30,  # High!
        'reg_alpha': 5.0,        # Strong L1
        'reg_lambda': 10.0,      # Strong L2
        'gamma': 0.5,            # Pruning
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbosity': 0,
    },
    'conservative': {
        'n_estimators': 500,
        'max_depth': 3,
        'learning_rate': 0.02,
        'subsample': 0.65,
        'colsample_bytree': 0.55,
        'min_child_weight': 25,
        'reg_alpha': 3.0,
        'reg_lambda': 8.0,
        'gamma': 0.3,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbosity': 0,
    },
    'moderate': {
        'n_estimators': 800,
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
}

best_models = {}
best_scores = {}

for target in TARGETS:
    print(f"\n  {'═' * 60}")
    print(f"  TARGET: {target}")
    print(f"  {'═' * 60}")
    
    y_raw = train_featured[target].copy()
    valid_mask = y_raw.notna()
    y_valid = y_raw[valid_mask]
    X_valid = X_train[valid_mask]
    groups_valid = spatial_groups[valid_mask]
    
    best_r2 = -999
    best_config_name = ""
    best_model_obj = None
    best_model_type = ""
    best_log = False
    
    # ── A. XGBoost models ──
    for config_name, params in xgb_configs.items():
        for use_log in [False, True]:
            label = f"XGB {config_name} {'+ log' if use_log else '+ raw'}"
            
            if use_log and (y_valid < 0).any():
                continue
            
            y = np.log1p(y_valid) if use_log else y_valid
            
            gkf = GroupKFold(n_splits=min(5, n_groups))
            fold_scores = []
            
            for train_idx, val_idx in gkf.split(X_valid, y, groups=groups_valid):
                X_tr, X_va = X_valid.iloc[train_idx], X_valid.iloc[val_idx]
                y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
                
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
            print(f"    {label:40s} → R² = {mean_r2:.4f} (±{std_r2:.4f})")
            
            if mean_r2 > best_r2:
                best_r2 = mean_r2
                best_config_name = label
                best_model_type = 'xgb'
                best_log = use_log
    
    # ── B. Ridge Regression (often better for spatial generalization) ──
    for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        for use_log in [False, True]:
            label = f"Ridge α={alpha} {'+ log' if use_log else '+ raw'}"
            
            if use_log and (y_valid < 0).any():
                continue
            
            y = np.log1p(y_valid) if use_log else y_valid
            
            gkf = GroupKFold(n_splits=min(5, n_groups))
            fold_scores = []
            
            for train_idx, val_idx in gkf.split(X_valid, y, groups=groups_valid):
                X_tr, X_va = X_valid.iloc[train_idx], X_valid.iloc[val_idx]
                y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
                
                # Ridge needs imputed + scaled data
                pipe = Pipeline([
                    ('imputer', StandardScaler()),  # NaN -> 0 after scaling
                ])
                
                # Manual impute with median
                X_tr_imp = X_tr.fillna(X_tr.median())
                X_va_imp = X_va.fillna(X_tr.median())  # Use train median!
                
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
            print(f"    {label:40s} → R² = {mean_r2:.4f} (±{std_r2:.4f})")
            
            if mean_r2 > best_r2:
                best_r2 = mean_r2
                best_config_name = label
                best_model_type = 'ridge'
                best_log = use_log
    
    # ── C. ElasticNet (sparse feature selection) ──
    for alpha in [0.1, 1.0, 10.0]:
        for l1_ratio in [0.1, 0.5, 0.9]:
            for use_log in [False, True]:
                label = f"ElasticNet α={alpha} l1={l1_ratio} {'+ log' if use_log else '+ raw'}"
                
                if use_log and (y_valid < 0).any():
                    continue
                
                y = np.log1p(y_valid) if use_log else y_valid
                
                gkf = GroupKFold(n_splits=min(5, n_groups))
                fold_scores = []
                
                for train_idx, val_idx in gkf.split(X_valid, y, groups=groups_valid):
                    X_tr, X_va = X_valid.iloc[train_idx], X_valid.iloc[val_idx]
                    y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
                    
                    X_tr_imp = X_tr.fillna(X_tr.median())
                    X_va_imp = X_va.fillna(X_tr.median())
                    
                    scaler = StandardScaler()
                    X_tr_sc = scaler.fit_transform(X_tr_imp)
                    X_va_sc = scaler.transform(X_va_imp)
                    
                    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, 
                                       random_state=RANDOM_STATE, max_iter=5000)
                    model.fit(X_tr_sc, y_tr)
                    preds = model.predict(X_va_sc)
                    
                    if use_log:
                        r2 = r2_score(np.expm1(y_va), np.expm1(preds))
                    else:
                        r2 = r2_score(y_va, preds)
                    fold_scores.append(r2)
                
                mean_r2 = np.mean(fold_scores)
                std_r2 = np.std(fold_scores)
                # Only print if competitive
                if mean_r2 > best_r2 - 0.1:
                    print(f"    {label:40s} → R² = {mean_r2:.4f} (±{std_r2:.4f})")
                
                if mean_r2 > best_r2:
                    best_r2 = mean_r2
                    best_config_name = label
                    best_model_type = 'elasticnet'
                    best_log = use_log
    
    # ── Retrain best on all data ──
    print(f"\n    ★ BEST: {best_config_name} → R² = {best_r2:.4f}")
    
    y_full = np.log1p(y_valid) if best_log else y_valid
    
    if best_model_type == 'xgb':
        # Find the matching params
        config_key = best_config_name.split(' ')[1]  # e.g., 'conservative'
        params = xgb_configs[config_key]
        best_model_obj = xgb.XGBRegressor(**params)
        best_model_obj.fit(X_valid, y_full)
        
        # Feature importance
        imps = best_model_obj.feature_importances_
        feat_imp = sorted(zip(feature_cols, imps), key=lambda x: x[1], reverse=True)
        print(f"\n    Top 10 features:")
        for rank, (fn, fi) in enumerate(feat_imp[:10], 1):
            bar = '█' * int(fi * 100)
            print(f"      {rank:2d}. {fn:30s} {fi:.4f} {bar}")
    
    elif best_model_type in ('ridge', 'elasticnet'):
        # Parse alpha from label
        alpha = float(best_config_name.split('α=')[1].split(' ')[0])
        
        X_imp = X_valid.fillna(X_valid.median())
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X_imp)
        
        if best_model_type == 'ridge':
            best_model_obj = Ridge(alpha=alpha, random_state=RANDOM_STATE)
        else:
            l1 = float(best_config_name.split('l1=')[1].split(' ')[0])
            best_model_obj = ElasticNet(alpha=alpha, l1_ratio=l1, 
                                         random_state=RANDOM_STATE, max_iter=5000)
        
        best_model_obj.fit(X_sc, y_full)
        
        # Store scaler and impute medians for prediction
        best_model_obj._v2_scaler = scaler
        best_model_obj._v2_medians = X_valid.median()
        
        # Feature importance for linear models
        coefs = np.abs(best_model_obj.coef_)
        feat_imp = sorted(zip(feature_cols, coefs), key=lambda x: x[1], reverse=True)
        print(f"\n    Top 10 features (|coefficient|):")
        for rank, (fn, fi) in enumerate(feat_imp[:10], 1):
            print(f"      {rank:2d}. {fn:30s} {fi:.4f}")
    
    best_models[target] = {
        'model': best_model_obj, 
        'use_log': best_log, 
        'type': best_model_type,
    }
    best_scores[target] = best_r2

# ============================================================
# [6] SUMMARY
# ============================================================
print(f"\n{'=' * 70}")
print(f"V2 MODEL — SPATIAL CV RESULTS (NO COORDINATE FEATURES)")
print(f"{'=' * 70}")
for t, s in best_scores.items():
    status = "✓" if s > 0 else "✗"
    print(f"  {status} {t:40s}: R² = {s:.4f}")
mean_r2 = np.mean(list(best_scores.values()))
print(f"  {'─' * 55}")
print(f"    {'MEAN R²':40s}: {mean_r2:.4f}")

# Compare with V1
v1_mean = -0.4009
print(f"\n    V1 mean R² was: {v1_mean:.4f}")
print(f"    V2 improvement: {mean_r2 - v1_mean:+.4f}")

# ============================================================
# [7] GENERATE SUBMISSION
# ============================================================
print(f"\n[7] Generating submission...")

submission = submission_template.copy()
for target in TARGETS:
    m = best_models[target]
    
    if m['type'] == 'xgb':
        preds = m['model'].predict(X_val)
    else:
        # Linear model needs imputed + scaled data
        X_val_imp = X_val.fillna(m['model']._v2_medians)
        X_val_sc = m['model']._v2_scaler.transform(X_val_imp)
        preds = m['model'].predict(X_val_sc)
    
    if m['use_log']:
        preds = np.expm1(preds)
    
    preds = np.maximum(preds, 0)
    submission[target] = preds
    print(f"  {target}:")
    print(f"    Range: [{preds.min():.2f}, {preds.max():.2f}], Mean: {preds.mean():.2f}")

ts = time.strftime("%Y%m%d_%H%M%S")
submission.to_csv(os.path.join(SUBMISSION_DIR, f'submission_v2_{ts}.csv'), index=False)
submission.to_csv(os.path.join(SUBMISSION_DIR, 'submission_v2.csv'), index=False)

print(f"\n  Saved: submissions/submission_v2.csv")
print(f"\n  Preview:")
print(submission.head(5).to_string())

print(f"""
{'=' * 70}
V2 ANALYSIS
{'=' * 70}
Key changes from V1:
  - Removed ALL coordinate features (Lat/Lon/lat_abs)
  - Added Ridge + ElasticNet (better spatial generalization)
  - Stronger XGBoost regularization (max_depth=3, high penalties)
  - Added climate-soil interactions (transferable physics)

If V2 spatial CV is still negative:
  → Submit anyway (CV may be pessimistic for this geography)
  → Try V3 with: Optuna tuning, LightGBM, stacking ensemble
  → Extract more Landsat bands (blue, red, thermal)
  → Add distance-to-river, land use features

If V2 spatial CV is positive:
  → Great! Submit and iterate with Optuna for hyperparameter tuning
{'=' * 70}
""")

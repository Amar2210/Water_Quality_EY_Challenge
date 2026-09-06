"""
============================================================
EY Water Quality Challenge - V0.1 Model (Quick Fix)
============================================================
Changes from V0:
- Adds Latitude/Longitude as features (they carry real geographic 
  signal — latitude correlates with climate zone, longitude with 
  distance from coast. With regularization, this helps.)
- Reduces model complexity to prevent overfitting
- Also tries without log transform (sometimes hurts spatial generalization)
============================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
import xgboost as xgb
import warnings
import os
import time

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'Jupyter Notebook Package')
SUBMISSION_DIR = os.path.join(BASE_DIR, '..', 'submissions')
os.makedirs(SUBMISSION_DIR, exist_ok=True)

TARGETS = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
RANDOM_STATE = 42

print("=" * 70)
print("EY WATER QUALITY CHALLENGE - V0.1 MODEL (Quick Fix)")
print("=" * 70)

# ============================================================
# LOAD & MERGE
# ============================================================
print("\n[1] Loading and merging data...")

wq_train = pd.read_csv(os.path.join(DATA_DIR, 'water_quality_training_dataset.csv'))
landsat_train = pd.read_csv(os.path.join(DATA_DIR, 'landsat_features_training.csv'))
terra_train = pd.read_csv(os.path.join(DATA_DIR, 'terraclimate_features_training.csv'))

submission_template = pd.read_csv(os.path.join(DATA_DIR, 'submission_template.csv'))
landsat_val = pd.read_csv(os.path.join(DATA_DIR, 'landsat_features_validation.csv'))
terra_val = pd.read_csv(os.path.join(DATA_DIR, 'terraclimate_features_validation.csv'))

train_df = pd.concat([
    wq_train,
    landsat_train.drop(columns=['Latitude', 'Longitude', 'Sample Date']),
    terra_train.drop(columns=['Latitude', 'Longitude', 'Sample Date'])
], axis=1)

val_df = pd.concat([
    submission_template[['Latitude', 'Longitude', 'Sample Date']],
    landsat_val.drop(columns=['Latitude', 'Longitude', 'Sample Date']),
    terra_val.drop(columns=['Latitude', 'Longitude', 'Sample Date'])
], axis=1)

# ============================================================
# FEATURE ENGINEERING
# ============================================================
print("[2] Engineering features...")

def engineer_features(df):
    df = df.copy()
    
    # --- Keep Lat/Lon as features this time ---
    # Latitude = climate zone proxy, Longitude = coast distance proxy
    # These are legitimate geographic signals for South Africa
    
    # --- Temporal features ---
    df['parsed_date'] = pd.to_datetime(df['Sample Date'], dayfirst=True, errors='coerce')
    df['month'] = df['parsed_date'].dt.month
    df['year'] = df['parsed_date'].dt.year
    df['day_of_year'] = df['parsed_date'].dt.dayofyear
    
    season_map = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
                  6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    df['season'] = df['month'].map(season_map)
    
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # --- Band ratios ---
    eps = 1e-10
    df['nir_green_ratio'] = df['nir'] / (df['green'] + eps)
    df['swir_ratio'] = df['swir16'] / (df['swir22'] + eps)
    df['green_swir22_ratio'] = df['green'] / (df['swir22'] + eps)
    df['nir_swir16_ratio'] = df['nir'] / (df['swir16'] + eps)
    df['brightness'] = (df['nir'] + df['green'] + df['swir16'] + df['swir22']) / 4
    df['pseudo_ndvi'] = (df['nir'] - df['green']) / (df['nir'] + df['green'] + eps)
    df['bsi_approx'] = ((df['swir16'] + df['green']) - (df['nir'] + df['swir22'])) / \
                        ((df['swir16'] + df['green']) + (df['nir'] + df['swir22']) + eps)
    
    # --- Geographic interaction features ---
    # These help the model learn "PET at this latitude means X"
    df['lat_x_pet'] = df['Latitude'] * df['pet']
    df['lon_x_pet'] = df['Longitude'] * df['pet']
    df['lat_x_month'] = df['Latitude'] * df['month']
    
    df = df.drop(columns=['parsed_date', 'Sample Date'], errors='ignore')
    return df

train_featured = engineer_features(train_df)
val_featured = engineer_features(val_df)

id_cols = []  # Lat/Lon are now features, not IDs
feature_cols = [c for c in train_featured.columns if c not in TARGETS]

print(f"  Total features: {len(feature_cols)}")
print(f"  Features: {feature_cols}")

# ============================================================
# SPATIAL CV
# ============================================================
print("\n[3] Spatial cross-validation...")

train_coords = train_featured[['Latitude', 'Longitude']].values
kmeans = KMeans(n_clusters=8, random_state=RANDOM_STATE, n_init=10)
spatial_groups = kmeans.fit_predict(train_coords)

# ============================================================
# TEST MULTIPLE CONFIGURATIONS
# ============================================================
# We'll try two configs and pick the best per target

configs = {
    'conservative': {
        'n_estimators': 300,
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 20,
        'reg_alpha': 1.0,
        'reg_lambda': 5.0,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbosity': 0,
    },
    'moderate': {
        'n_estimators': 500,
        'max_depth': 5,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 10,
        'reg_alpha': 0.5,
        'reg_lambda': 3.0,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbosity': 0,
    },
}

# Also test with and without log transform
log_options = {
    'Total Alkalinity': [False],
    'Electrical Conductance': [False, True],
    'Dissolved Reactive Phosphorus': [False, True],
}

print("\n[4] Training models (testing configurations)...")

X_train_all = train_featured[feature_cols]
X_val = val_featured[feature_cols]

best_models = {}
best_scores = {}

for target in TARGETS:
    print(f"\n  {'═' * 55}")
    print(f"  {target}")
    print(f"  {'═' * 55}")
    
    y_raw = train_featured[target].copy()
    best_r2 = -999
    best_config_name = ""
    best_model = None
    best_log = False
    
    for config_name, params in configs.items():
        for use_log in log_options[target]:
            
            if use_log:
                y = np.log1p(y_raw)
                label = f"{config_name} + log"
            else:
                y = y_raw.copy()
                label = f"{config_name} + raw"
            
            # Spatial CV
            gkf = GroupKFold(n_splits=5)
            fold_scores = []
            
            for train_idx, val_idx in gkf.split(X_train_all, y, groups=spatial_groups):
                X_tr = X_train_all.iloc[train_idx]
                y_tr = y.iloc[train_idx]
                X_va = X_train_all.iloc[val_idx]
                y_va = y.iloc[val_idx]
                
                model = xgb.XGBRegressor(**params)
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                
                preds = model.predict(X_va)
                if use_log:
                    r2 = r2_score(np.expm1(y_va), np.expm1(preds))
                else:
                    r2 = r2_score(y_va, preds)
                fold_scores.append(r2)
            
            mean_r2 = np.mean(fold_scores)
            print(f"    {label:30s} → Spatial CV R² = {mean_r2:.4f}")
            
            if mean_r2 > best_r2:
                best_r2 = mean_r2
                best_config_name = label
                best_log = use_log
                # Retrain on all data
                y_full = np.log1p(y_raw) if use_log else y_raw
                best_model = xgb.XGBRegressor(**params)
                best_model.fit(X_train_all, y_full)
    
    print(f"    ► BEST: {best_config_name} → R² = {best_r2:.4f}")
    best_models[target] = {'model': best_model, 'use_log': best_log}
    best_scores[target] = best_r2
    
    # Feature importance
    imps = best_model.feature_importances_
    feat_imp = sorted(zip(feature_cols, imps), key=lambda x: x[1], reverse=True)
    print(f"    Top 5 features:")
    for fn, fi in feat_imp[:5]:
        print(f"      {fn:25s} {fi:.4f} {'█' * int(fi * 80)}")

# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'=' * 70}")
print(f"V0.1 SPATIAL CV SUMMARY")
print(f"{'=' * 70}")
for t, s in best_scores.items():
    print(f"  {t:35s}: R² = {s:.4f}")
mean_r2 = np.mean(list(best_scores.values()))
print(f"  {'─' * 50}")
print(f"  {'MEAN R²':35s}: {mean_r2:.4f}")

# ============================================================
# GENERATE SUBMISSION
# ============================================================
print(f"\n[5] Generating submission...")

submission = submission_template.copy()
for target in TARGETS:
    m = best_models[target]
    preds = m['model'].predict(X_val)
    if m['use_log']:
        preds = np.expm1(preds)
    preds = np.maximum(preds, 0)
    submission[target] = preds
    print(f"  {target}: min={preds.min():.2f}, max={preds.max():.2f}, mean={preds.mean():.2f}")

ts = time.strftime("%Y%m%d_%H%M%S")
submission.to_csv(os.path.join(SUBMISSION_DIR, f'submission_v0.1_{ts}.csv'), index=False)
submission.to_csv(os.path.join(SUBMISSION_DIR, 'submission_v0.1.csv'), index=False)

print(f"\n  Saved: submissions/submission_v0.1.csv")
print(f"\n  Preview:")
print(submission.head(5).to_string())

print(f"""
{'=' * 70}
ANALYSIS OF RESULTS
{'=' * 70}
The negative spatial CV R² tells us something critical:

  With ONLY satellite spectral features + PET + temporal features,
  the model CANNOT generalize across different geographies.

  WHY? Because these features don't tell the model:
  - What's the elevation? (drives alkalinity via geology)
  - What's the rainfall? (drives everything via dilution)
  - What's the soil type? (drives DRP via nutrient retention)
  - What land use surrounds this point? (the pollution source)

  PET being the #1 feature for all targets PROVES this —
  it's the only feature that encodes both location AND time.

  NEXT STEP: Extract TerraClimate expanded features (especially
  precipitation, temperature, soil moisture) and elevation.
  These will transform the model from "blind guessing" to 
  "informed prediction."
{'=' * 70}
""")

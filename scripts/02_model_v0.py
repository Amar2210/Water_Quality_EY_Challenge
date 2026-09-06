"""
============================================================
EY Water Quality Challenge - V0 Model (Quick Win)
============================================================
Uses ONLY existing provided data, but SMARTER:
- All 6 Landsat features (benchmark uses only 4)
- PET from TerraClimate
- Temporal features from date
- Band ratios computed from existing bands
- XGBoost instead of RandomForest
- Log transform for skewed targets (DRP, EC)
- Spatial cross-validation for honest evaluation

Expected: Significant improvement over 0.20 baseline
============================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import warnings
import os
import time

warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'Jupyter Notebook Package')
SUBMISSION_DIR = os.path.join(BASE_DIR, '..', 'submissions')
os.makedirs(SUBMISSION_DIR, exist_ok=True)

TARGETS = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
# Which targets to log-transform (skewed distributions)
LOG_TRANSFORM_TARGETS = ['Electrical Conductance', 'Dissolved Reactive Phosphorus']

RANDOM_STATE = 42
N_SPATIAL_CLUSTERS = 8   # For spatial cross-validation
N_CV_FOLDS = 5

print("=" * 70)
print("EY WATER QUALITY CHALLENGE - V0 MODEL")
print("=" * 70)

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("\n[STEP 1] Loading data...")

# Training data
wq_train = pd.read_csv(os.path.join(DATA_DIR, 'water_quality_training_dataset.csv'))
landsat_train = pd.read_csv(os.path.join(DATA_DIR, 'landsat_features_training.csv'))
terra_train = pd.read_csv(os.path.join(DATA_DIR, 'terraclimate_features_training.csv'))

# Validation data
submission_template = pd.read_csv(os.path.join(DATA_DIR, 'submission_template.csv'))
landsat_val = pd.read_csv(os.path.join(DATA_DIR, 'landsat_features_validation.csv'))
terra_val = pd.read_csv(os.path.join(DATA_DIR, 'terraclimate_features_validation.csv'))

print(f"  Training samples:   {len(wq_train)}")
print(f"  Validation samples: {len(submission_template)}")

# ============================================================
# STEP 2: MERGE DATASETS
# ============================================================
print("\n[STEP 2] Merging datasets...")

# Training: concat along columns (rows are aligned across all files)
train_df = pd.concat([
    wq_train,
    landsat_train.drop(columns=['Latitude', 'Longitude', 'Sample Date']),
    terra_train.drop(columns=['Latitude', 'Longitude', 'Sample Date'])
], axis=1)

# Validation: concat along columns
val_df = pd.concat([
    submission_template[['Latitude', 'Longitude', 'Sample Date']],
    landsat_val.drop(columns=['Latitude', 'Longitude', 'Sample Date']),
    terra_val.drop(columns=['Latitude', 'Longitude', 'Sample Date'])
], axis=1)

print(f"  Training shape after merge:   {train_df.shape}")
print(f"  Validation shape after merge: {val_df.shape}")
print(f"  Training NaN rows (Landsat):  {train_df['nir'].isna().sum()}")
print(f"  Validation NaN rows (Landsat): {val_df['nir'].isna().sum()}")

# ============================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================
print("\n[STEP 3] Engineering features...")

def engineer_features(df):
    """
    Create all features from existing data. No external extraction needed.
    """
    df = df.copy()
    
    # --- Temporal features from date ---
    df['parsed_date'] = pd.to_datetime(df['Sample Date'], dayfirst=True, errors='coerce')
    df['month'] = df['parsed_date'].dt.month
    df['year'] = df['parsed_date'].dt.year
    df['day_of_year'] = df['parsed_date'].dt.dayofyear
    
    # South African seasons (Southern Hemisphere!)
    # Summer: Dec-Feb, Autumn: Mar-May, Winter: Jun-Aug, Spring: Sep-Nov
    season_map = {12: 0, 1: 0, 2: 0,    # Summer
                  3: 1, 4: 1, 5: 1,      # Autumn
                  6: 2, 7: 2, 8: 2,      # Winter
                  9: 3, 10: 3, 11: 3}    # Spring
    df['season'] = df['month'].map(season_map)
    
    # Cyclical encoding for month (so December and January are close)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Cyclical encoding for day of year
    df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # --- Band ratios from existing Landsat data ---
    # These capture surface characteristics better than raw bands alone
    eps = 1e-10  # avoid division by zero
    
    # NIR/Green ratio - indicates vegetation vigor vs water
    df['nir_green_ratio'] = df['nir'] / (df['green'] + eps)
    
    # SWIR ratio - surface moisture indicator
    df['swir_ratio'] = df['swir16'] / (df['swir22'] + eps)
    
    # Green/SWIR22 ratio - water clarity indicator
    df['green_swir22_ratio'] = df['green'] / (df['swir22'] + eps)
    
    # NIR/SWIR16 ratio - similar to NDMI but as ratio (sometimes more useful for trees)
    df['nir_swir16_ratio'] = df['nir'] / (df['swir16'] + eps)
    
    # Overall brightness - general surface reflectance level
    df['brightness'] = (df['nir'] + df['green'] + df['swir16'] + df['swir22']) / 4
    
    # NDVI approximation - we don't have Red band, but NIR/Green spread is related
    # Not true NDVI but captures some vegetation signal
    df['pseudo_ndvi'] = (df['nir'] - df['green']) / (df['nir'] + df['green'] + eps)
    
    # Bare soil index approximation
    df['bsi_approx'] = ((df['swir16'] + df['green']) - (df['nir'] + df['swir22'])) / \
                        ((df['swir16'] + df['green']) + (df['nir'] + df['swir22']) + eps)
    
    # --- PET interaction with season ---
    # PET means different things in summer vs winter
    df['pet_x_season'] = df['pet'] * df['season']
    
    # Drop helper columns
    df = df.drop(columns=['parsed_date', 'Sample Date'], errors='ignore')
    
    return df


# Apply feature engineering
train_featured = engineer_features(train_df)
val_featured = engineer_features(val_df)

# Define feature columns (everything that's not a target or ID)
id_cols = ['Latitude', 'Longitude']
feature_cols = [c for c in train_featured.columns 
                if c not in TARGETS + id_cols]

print(f"  Total features created: {len(feature_cols)}")
print(f"  Features: {feature_cols}")

# ============================================================
# STEP 4: SPATIAL CROSS-VALIDATION SETUP
# ============================================================
print("\n[STEP 4] Setting up spatial cross-validation...")

# Cluster training locations into spatial groups
train_coords = train_featured[['Latitude', 'Longitude']].values
kmeans = KMeans(n_clusters=N_SPATIAL_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
spatial_groups = kmeans.fit_predict(train_coords)

# Show cluster distribution
unique_groups, group_counts = np.unique(spatial_groups, return_counts=True)
print(f"  Created {N_SPATIAL_CLUSTERS} spatial clusters:")
for g, c in zip(unique_groups, group_counts):
    print(f"    Cluster {g}: {c} samples")

# ============================================================
# STEP 5: TRAIN MODELS
# ============================================================
print("\n[STEP 5] Training XGBoost models...")

# XGBoost parameters for v0 (reasonable defaults)
xgb_params = {
    'n_estimators': 800,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbosity': 0,
}

# Store results
models = {}
cv_scores = {}

X_train_all = train_featured[feature_cols]
X_val = val_featured[feature_cols]

for target in TARGETS:
    print(f"\n  {'─' * 50}")
    print(f"  Training: {target}")
    print(f"  {'─' * 50}")
    
    y_raw = train_featured[target].copy()
    
    # Apply log transform for skewed targets
    use_log = target in LOG_TRANSFORM_TARGETS
    if use_log:
        # log1p handles zero values safely: log1p(x) = log(1 + x)
        y = np.log1p(y_raw)
        print(f"  Applied log1p transform (skewness was {y_raw.skew():.2f}, now {y.skew():.2f})")
    else:
        y = y_raw.copy()
        print(f"  No transform needed (skewness: {y_raw.skew():.2f})")
    
    # --- Spatial Cross-Validation ---
    gkf = GroupKFold(n_splits=N_CV_FOLDS)
    fold_scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train_all, y, groups=spatial_groups)):
        X_fold_train = X_train_all.iloc[train_idx]
        y_fold_train = y.iloc[train_idx]
        X_fold_val = X_train_all.iloc[val_idx]
        y_fold_val = y.iloc[val_idx]
        
        fold_model = xgb.XGBRegressor(**xgb_params)
        fold_model.fit(
            X_fold_train, y_fold_train,
            eval_set=[(X_fold_val, y_fold_val)],
            verbose=False
        )
        
        y_fold_pred = fold_model.predict(X_fold_val)
        
        # If log-transformed, convert back for R² calculation
        if use_log:
            r2 = r2_score(np.expm1(y_fold_val), np.expm1(y_fold_pred))
        else:
            r2 = r2_score(y_fold_val, y_fold_pred)
        
        fold_scores.append(r2)
        print(f"    Fold {fold_idx + 1}: R² = {r2:.4f}")
    
    mean_cv_r2 = np.mean(fold_scores)
    std_cv_r2 = np.std(fold_scores)
    print(f"    ► Mean Spatial CV R²: {mean_cv_r2:.4f} (±{std_cv_r2:.4f})")
    cv_scores[target] = mean_cv_r2
    
    # --- Train final model on ALL training data ---
    print(f"  Training final model on all {len(X_train_all)} samples...")
    final_model = xgb.XGBRegressor(**xgb_params)
    final_model.fit(X_train_all, y)
    
    # Store
    models[target] = {
        'model': final_model,
        'use_log': use_log
    }
    
    # --- Feature importance (top 10) ---
    importances = final_model.feature_importances_
    feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    print(f"  Top 10 features:")
    for fname, fimp in feat_imp[:10]:
        bar = '█' * int(fimp * 100)
        print(f"    {fname:25s} {fimp:.4f} {bar}")

# ============================================================
# STEP 6: OVERALL CV SCORE
# ============================================================
print(f"\n{'=' * 70}")
print(f"SPATIAL CROSS-VALIDATION SUMMARY")
print(f"{'=' * 70}")
for target, score in cv_scores.items():
    print(f"  {target:35s}: R² = {score:.4f}")
mean_overall = np.mean(list(cv_scores.values()))
print(f"  {'─' * 50}")
print(f"  {'MEAN R² (Leaderboard estimate)':35s}: {mean_overall:.4f}")
print(f"{'=' * 70}")

# ============================================================
# STEP 7: GENERATE SUBMISSION
# ============================================================
print(f"\n[STEP 7] Generating submission...")

submission = submission_template.copy()

for target in TARGETS:
    model_info = models[target]
    final_model = model_info['model']
    use_log = model_info['use_log']
    
    # Predict
    preds = final_model.predict(X_val)
    
    # Inverse transform if log was applied
    if use_log:
        preds = np.expm1(preds)
    
    # Ensure no negative predictions (water quality params can't be negative)
    preds = np.maximum(preds, 0)
    
    submission[target] = preds
    print(f"  {target}: min={preds.min():.2f}, max={preds.max():.2f}, mean={preds.mean():.2f}")

# Save submission
timestamp = time.strftime("%Y%m%d_%H%M%S")
submission_path = os.path.join(SUBMISSION_DIR, f'submission_v0_{timestamp}.csv')
submission.to_csv(submission_path, index=False)

# Also save a copy with a simple name for easy upload
simple_path = os.path.join(SUBMISSION_DIR, 'submission_v0.csv')
submission.to_csv(simple_path, index=False)

print(f"\n  Saved: {submission_path}")
print(f"  Saved: {simple_path}")

# Preview
print(f"\n  Submission preview:")
print(submission.head(10).to_string())

# ============================================================
# STEP 8: COMPARISON WITH BENCHMARK
# ============================================================
print(f"\n{'=' * 70}")
print(f"V0 MODEL vs BENCHMARK COMPARISON")
print(f"{'=' * 70}")
print(f"""
  Benchmark (RandomForest, 4 features):
    Internal test R²: ~0.55 avg
    Leaderboard R²:   ~0.20
    
  V0 Model (XGBoost, {len(feature_cols)} features):
    Spatial CV R²:    {mean_overall:.4f} (honest estimate)
    
  Improvements in V0:
    ✓ Used ALL 6 Landsat features (not just 4)
    ✓ Added {len([c for c in feature_cols if c.startswith(('month','year','day','season','doy'))])} temporal features  
    ✓ Added {len([c for c in feature_cols if 'ratio' in c or c in ['brightness','pseudo_ndvi','bsi_approx','pet_x_season']])} engineered features (band ratios, indices)
    ✓ XGBoost instead of RandomForest
    ✓ Log transform for skewed targets (DRP, EC)
    ✓ Native NaN handling (no median imputation)
    ✓ Spatial CV (honest evaluation)
    
  Next steps (V1+):
    → Extract expanded TerraClimate (precipitation, temperature, soil moisture, runoff)
    → Add elevation from SRTM
    → Add soil properties from SoilGrids
    → Hyperparameter tuning with Optuna
    → Ensemble with LightGBM and CatBoost
""")

print("Done! Upload submissions/submission_v0.csv to the challenge platform.")

"""
EY Water Quality Challenge - Data Analysis Script
Run this first to understand what we have and what we need.
"""
import pandas as pd
import numpy as np
import os

# ============================================
# PATHS
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, '..', 'Jupyter Notebook Package')

# ============================================
# LOAD ALL DATA
# ============================================
print("=" * 70)
print("EY WATER QUALITY CHALLENGE - DATA ANALYSIS")
print("=" * 70)

# Training targets
wq_train = pd.read_csv(os.path.join(RAW_DIR, 'water_quality_training_dataset.csv'))
print(f"\n--- TRAINING TARGETS ---")
print(f"Shape: {wq_train.shape}")
print(f"Columns: {list(wq_train.columns)}")
print(f"\nFirst 3 rows:")
print(wq_train.head(3))
print(f"\nDate range: {wq_train['Sample Date'].iloc[0]} to {wq_train['Sample Date'].iloc[-1]}")

# Unique locations
unique_locs = wq_train.groupby(['Latitude', 'Longitude']).size().reset_index(name='count')
print(f"\nUnique sampling locations: {len(unique_locs)}")
print(f"Total samples: {len(wq_train)}")
print(f"Avg samples per location: {len(wq_train) / len(unique_locs):.1f}")

# Target distributions
print(f"\n--- TARGET DISTRIBUTIONS ---")
for col in ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']:
    vals = wq_train[col]
    print(f"\n{col}:")
    print(f"  Mean: {vals.mean():.2f}, Median: {vals.median():.2f}, Std: {vals.std():.2f}")
    print(f"  Min: {vals.min():.2f}, Max: {vals.max():.2f}")
    print(f"  Skewness: {vals.skew():.2f}")
    print(f"  Nulls: {vals.isna().sum()}")

# Lat/Lon ranges for training
print(f"\n--- TRAINING GEOGRAPHY ---")
print(f"Latitude:  min={wq_train['Latitude'].min():.4f}, max={wq_train['Latitude'].max():.4f}")
print(f"Longitude: min={wq_train['Longitude'].min():.4f}, max={wq_train['Longitude'].max():.4f}")

# ============================================
# LANDSAT FEATURES
# ============================================
landsat_train = pd.read_csv(os.path.join(RAW_DIR, 'landsat_features_training.csv'))
landsat_val = pd.read_csv(os.path.join(RAW_DIR, 'landsat_features_validation.csv'))

print(f"\n--- LANDSAT FEATURES (TRAINING) ---")
print(f"Shape: {landsat_train.shape}")
print(f"Columns: {list(landsat_train.columns)}")
print(f"Null counts:")
print(landsat_train.isnull().sum())
print(f"\nFeature stats:")
for col in ['nir', 'green', 'swir16', 'swir22', 'NDMI', 'MNDWI']:
    vals = landsat_train[col].dropna()
    print(f"  {col:8s}: mean={vals.mean():10.2f}, std={vals.std():10.2f}, nulls={landsat_train[col].isna().sum()}")

print(f"\n--- LANDSAT FEATURES (VALIDATION) ---")
print(f"Shape: {landsat_val.shape}")
print(f"Null counts:")
print(landsat_val.isnull().sum())
null_pct = landsat_val.isnull().sum() / len(landsat_val) * 100
print(f"\nNull percentage per column:")
for col in landsat_val.columns:
    print(f"  {col}: {null_pct[col]:.1f}%")

# ============================================
# TERRACLIMATE FEATURES
# ============================================
terra_train = pd.read_csv(os.path.join(RAW_DIR, 'terraclimate_features_training.csv'))
terra_val = pd.read_csv(os.path.join(RAW_DIR, 'terraclimate_features_validation.csv'))

print(f"\n--- TERRACLIMATE FEATURES (TRAINING) ---")
print(f"Shape: {terra_train.shape}")
print(f"Columns: {list(terra_train.columns)}")
print(f"PET stats: mean={terra_train['pet'].mean():.2f}, std={terra_train['pet'].std():.2f}")
print(f"PET nulls: {terra_train['pet'].isna().sum()}")

print(f"\n--- TERRACLIMATE FEATURES (VALIDATION) ---")
print(f"Shape: {terra_val.shape}")
print(f"PET nulls: {terra_val['pet'].isna().sum()}")

# ============================================
# SUBMISSION TEMPLATE
# ============================================
submission = pd.read_csv(os.path.join(RAW_DIR, 'submission_template.csv'))
print(f"\n--- SUBMISSION TEMPLATE ---")
print(f"Shape: {submission.shape}")
print(f"Columns: {list(submission.columns)}")

# Validation geography
print(f"\n--- VALIDATION GEOGRAPHY ---")
print(f"Latitude:  min={submission['Latitude'].min():.4f}, max={submission['Latitude'].max():.4f}")
print(f"Longitude: min={submission['Longitude'].min():.4f}, max={submission['Longitude'].max():.4f}")

# Unique validation locations
val_locs = submission.groupby(['Latitude', 'Longitude']).size().reset_index(name='count')
print(f"Unique validation locations: {len(val_locs)}")
print(f"Total validation samples: {len(submission)}")

# ============================================
# GEOGRAPHIC OVERLAP ANALYSIS
# ============================================
print(f"\n{'=' * 70}")
print(f"GEOGRAPHIC OVERLAP ANALYSIS")
print(f"{'=' * 70}")
print(f"\nTraining region:   Lat [{wq_train['Latitude'].min():.2f}, {wq_train['Latitude'].max():.2f}] x Lon [{wq_train['Longitude'].min():.2f}, {wq_train['Longitude'].max():.2f}]")
print(f"Validation region: Lat [{submission['Latitude'].min():.2f}, {submission['Latitude'].max():.2f}] x Lon [{submission['Longitude'].min():.2f}, {submission['Longitude'].max():.2f}]")

# Check for overlapping locations
train_locs = set(zip(wq_train['Latitude'].round(4), wq_train['Longitude'].round(4)))
val_locs_set = set(zip(submission['Latitude'].round(4), submission['Longitude'].round(4)))
overlap = train_locs.intersection(val_locs_set)
print(f"\nOverlapping locations: {len(overlap)}")
if len(overlap) == 0:
    print(">>> ZERO OVERLAP - Model must generalize to completely new geography!")

# ============================================
# TEMPORAL ANALYSIS
# ============================================
print(f"\n{'=' * 70}")
print(f"TEMPORAL ANALYSIS")
print(f"{'=' * 70}")
wq_train['parsed_date'] = pd.to_datetime(wq_train['Sample Date'], dayfirst=True)
submission['parsed_date'] = pd.to_datetime(submission['Sample Date'], dayfirst=True)

print(f"\nTraining date range:   {wq_train['parsed_date'].min()} to {wq_train['parsed_date'].max()}")
print(f"Validation date range: {submission['parsed_date'].min()} to {submission['parsed_date'].max()}")

# Monthly distribution
print(f"\nTraining samples per month:")
monthly = wq_train['parsed_date'].dt.month.value_counts().sort_index()
for m, c in monthly.items():
    print(f"  Month {m:2d}: {c:4d} samples {'█' * (c // 20)}")

print(f"\nValidation samples per month:")
monthly_val = submission['parsed_date'].dt.month.value_counts().sort_index()
for m, c in monthly_val.items():
    print(f"  Month {m:2d}: {c:4d} samples {'█' * (c // 2)}")

# ============================================
# KEY FINDINGS SUMMARY
# ============================================
print(f"\n{'=' * 70}")
print(f"KEY FINDINGS & IMPLICATIONS")
print(f"{'=' * 70}")
print(f"""
1. DATASET SIZE: {len(wq_train)} training samples, {len(submission)} validation samples
2. FEATURES PROVIDED: Only {len(landsat_train.columns) - 3} Landsat + {len(terra_train.columns) - 3} TerraClimate = {len(landsat_train.columns) - 3 + len(terra_train.columns) - 3} total features
3. GEOGRAPHIC GAP: Training and validation are in DIFFERENT regions (zero overlap)
4. MISSING DATA: Landsat validation has significant NaN rows (cloud cover issues)
5. DRP SKEWNESS: {wq_train['Dissolved Reactive Phosphorus'].skew():.2f} - needs log transform
6. BENCHMARK USES: Only 4 features (swir22, NDMI, MNDWI, pet) out of {len(landsat_train.columns) - 3 + len(terra_train.columns) - 3} available

IMMEDIATE OPPORTUNITIES:
- Use ALL 6 Landsat features (not just 4) → FREE improvement
- Add temporal features from date (month, season) → FREE improvement  
- Extract more TerraClimate variables (ppt, tmin, tmax, soil, ro, aet, def, PDSI, vpd)
- Add elevation data (SRTM DEM)
- Switch from RandomForest to XGBoost/LightGBM
""")

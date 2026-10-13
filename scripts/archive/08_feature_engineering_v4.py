"""
============================================================
EY Water Quality Challenge - V4 Feature Engineering
============================================================
RESEARCH-BACKED FEATURE STRATEGY:

Paper 1 (DRP/Phosphate):
  → Conductivity, Slope, Month, Flow rate drive phosphate
  → We have: slope (new!), month, precipitation (flow proxy)
  → Stacking idea: predict EC first, use as DRP feature

Paper 2 (Electrical Conductance):
  → Total Hardness (TH), Magnesium (Mg), Chloride (Cl)
  → All geology-driven → proxied by elevation + soil properties
  → Temperature/evaporation concentrates salts

Paper 3 (Total Alkalinity):
  → Total Hardness, Bicarbonate → same geology story
  → Soil pH and CEC are direct proxies for mineral content

Paper 4 (Satellite Review - 113 studies):
  → Band ratios and indices from spectral bands are key
  → Ensemble methods help for optically inactive parameters
  → DNN/LSTM for EC if XGBoost plateaus

THIS SCRIPT:
  1. Loads ALL extracted data sources
  2. Merges them on (Lat, Lon, Date) or (Lat, Lon)
  3. Engineers new features from existing data:
     - Landsat band ratios (no new extraction needed)
     - Seasonal encoding (sin/cos)
     - Soil-climate interactions
     - Precipitation patterns and drought indicators
     - Slope + elevation derived features
  4. Outputs model-ready train/val CSVs

OUTPUT:
  data/extracted/v4_train_features.csv
  data/extracted/v4_val_features.csv

RUNTIME: ~10 seconds (no API calls, just math)
============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import os
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, '..', 'ey_challenge_notebooks')
EXTRACTED_DIR = os.path.join(BASE_DIR, '..', 'data', 'extracted')

print("=" * 70)
print("EY WATER QUALITY - V4 FEATURE ENGINEERING")
print("Research-backed features for Alkalinity, EC, and DRP")
print("=" * 70)


# ============================================================
# [1] LOAD ALL DATA SOURCES
# ============================================================
print("\n[1] Loading all data sources...")

# --- Targets (training only) ---
wq_train = pd.read_csv(os.path.join(DATA_DIR, 'water_quality_training_dataset.csv'))
submission_template = pd.read_csv(os.path.join(DATA_DIR, 'submission_template.csv'))
print(f"  Training samples: {len(wq_train)}")
print(f"  Validation samples: {len(submission_template)}")

# --- Landsat ---
landsat_train = pd.read_csv(os.path.join(DATA_DIR, 'landsat_features_training.csv'))
landsat_val = pd.read_csv(os.path.join(DATA_DIR, 'landsat_features_validation.csv'))
print(f"  Landsat train: {landsat_train.shape}, NaN rows: {landsat_train['nir'].isna().sum()}")
print(f"  Landsat val:   {landsat_val.shape}, NaN rows: {landsat_val['nir'].isna().sum()}")

# --- TerraClimate expanded ---
terra_train = pd.read_csv(os.path.join(EXTRACTED_DIR, 'terraclimate_expanded_training.csv'))
terra_val = pd.read_csv(os.path.join(EXTRACTED_DIR, 'terraclimate_expanded_validation.csv'))
print(f"  TerraClimate train: {terra_train.shape}")
print(f"  TerraClimate val:   {terra_val.shape}")

# --- Elevation ---
elevation = pd.read_csv(os.path.join(EXTRACTED_DIR, 'elevation_data.csv'))
print(f"  Elevation: {len(elevation)} locations, valid: {elevation['elevation_m'].notna().sum()}")

# --- Soil ---
soil = pd.read_csv(os.path.join(EXTRACTED_DIR, 'soil_data.csv'))
soil_cols = [c for c in soil.columns if c.startswith('soil_')]
valid_soil = soil[soil_cols[0]].notna().sum() if soil_cols else 0
print(f"  Soil: {len(soil)} locations, valid: {valid_soil}, properties: {soil_cols}")

# --- Slope (may not exist yet) ---
slope_path = os.path.join(EXTRACTED_DIR, 'slope_data.csv')
has_slope = os.path.exists(slope_path)
if has_slope:
    slope = pd.read_csv(slope_path)
    valid_slope = slope['slope_deg'].notna().sum()
    print(f"  Slope: {len(slope)} locations, valid: {valid_slope}")
else:
    slope = None
    print(f"  Slope: NOT FOUND — run 07_extract_slope.py first!")
    print(f"         (Continuing without slope — model will work but miss a key feature)")


# ============================================================
# [2] MERGE ALL SOURCES
# ============================================================
print(f"\n[2] Merging all data sources...")

join_keys = ['Latitude', 'Longitude', 'Sample Date']
loc_keys = ['Latitude', 'Longitude']

# --- Build training set ---
train_df = wq_train.copy()
train_df = train_df.merge(landsat_train, on=join_keys, how='left')
train_df = train_df.merge(terra_train, on=join_keys, how='left')
train_df = train_df.merge(elevation, on=loc_keys, how='left')
train_df = train_df.merge(soil, on=loc_keys, how='left')
if has_slope:
    train_df = train_df.merge(slope, on=loc_keys, how='left')

# --- Build validation set ---
val_df = submission_template.copy()
val_df = val_df.merge(landsat_val, on=join_keys, how='left')
val_df = val_df.merge(terra_val, on=join_keys, how='left')
val_df = val_df.merge(elevation, on=loc_keys, how='left')
val_df = val_df.merge(soil, on=loc_keys, how='left')
if has_slope:
    val_df = val_df.merge(slope, on=loc_keys, how='left')

print(f"  Train after merge: {train_df.shape}")
print(f"  Val after merge:   {val_df.shape}")

# Check for duplicate columns from merge
for df, name in [(train_df, 'train'), (val_df, 'val')]:
    dupe_cols = [c for c in df.columns if c.endswith('_x') or c.endswith('_y')]
    if dupe_cols:
        print(f"  WARNING: Duplicate columns in {name}: {dupe_cols}")
        # Keep _x version, drop _y
        for col in dupe_cols:
            if col.endswith('_y'):
                df.drop(columns=[col], inplace=True)
            elif col.endswith('_x'):
                df.rename(columns={col: col.replace('_x', '')}, inplace=True)


# ============================================================
# [3] FEATURE ENGINEERING
# ============================================================
print(f"\n[3] Engineering features...")


def engineer_features(df, dataset_name=''):
    """
    Apply all feature engineering to a dataframe.
    All transformations are done in-place for memory efficiency.
    """
    n_before = len(df.columns)
    
    # ─────────────────────────────────────────────────────
    # 3A. TEMPORAL FEATURES (from Sample Date)
    # Research: "month" was #3 feature for DRP prediction
    # ─────────────────────────────────────────────────────
    df['parsed_date'] = pd.to_datetime(df['Sample Date'], dayfirst=True, errors='coerce')
    df['month'] = df['parsed_date'].dt.month
    df['year'] = df['parsed_date'].dt.year
    df['day_of_year'] = df['parsed_date'].dt.dayofyear
    
    # Cyclical encoding — captures that December and January are adjacent
    # This is better than raw month number for ML models
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # South Africa seasons (Southern Hemisphere):
    #   Summer: Dec-Feb (wet season for eastern SA)
    #   Autumn: Mar-May
    #   Winter: Jun-Aug (dry season)
    #   Spring: Sep-Nov
    df['season'] = df['month'].map({
        12: 0, 1: 0, 2: 0,   # Summer (0)
        3: 1, 4: 1, 5: 1,    # Autumn (1)
        6: 2, 7: 2, 8: 2,    # Winter (2)
        9: 3, 10: 3, 11: 3   # Spring (3)
    })
    
    # Is wet season? (Oct-Mar for most of SA)
    df['is_wet_season'] = df['month'].isin([10, 11, 12, 1, 2, 3]).astype(int)
    
    # Drop helper
    df.drop(columns=['parsed_date'], inplace=True, errors='ignore')
    
    # ─────────────────────────────────────────────────────
    # 3B. LANDSAT BAND RATIOS & INDICES
    # Research paper 4: "band ratios improve prediction
    # of optically inactive parameters"
    # Available bands: nir, green, swir16, swir22
    # NOTE: We don't have red or blue, so no NDVI
    # ─────────────────────────────────────────────────────
    has_landsat = 'nir' in df.columns and 'green' in df.columns
    
    if has_landsat:
        # Safe division helper
        def safe_ratio(a, b, fill=0.0):
            return np.where(b != 0, a / b, fill)
        
        def safe_norm_diff(a, b, fill=0.0):
            denom = a + b
            return np.where(denom != 0, (a - b) / denom, fill)
        
        nir = df['nir'].values.astype(float)
        green = df['green'].values.astype(float)
        swir16 = df['swir16'].values.astype(float)
        swir22 = df['swir22'].values.astype(float)
        
        # --- Band ratios ---
        # These capture relationships between spectral regions
        # that correlate with water properties
        
        # NIR/Green — related to vegetation/turbidity near water
        df['ratio_nir_green'] = safe_ratio(nir, green)
        
        # SWIR16/NIR — moisture content indicator
        df['ratio_swir16_nir'] = safe_ratio(swir16, nir)
        
        # SWIR22/SWIR16 — mineral/clay content indicator
        # Research: correlates with geological substrate
        df['ratio_swir22_swir16'] = safe_ratio(swir22, swir16)
        
        # Green/SWIR16 — water turbidity indicator
        df['ratio_green_swir16'] = safe_ratio(green, swir16)
        
        # NIR/SWIR22 — moisture vs mineral balance
        df['ratio_nir_swir22'] = safe_ratio(nir, swir22)
        
        # --- Normalized difference indices ---
        # NDMI and MNDWI already exist, add more
        
        # SWIR ratio index: (swir16 - swir22) / (swir16 + swir22)
        # Sensitive to mineral composition and moisture
        df['ndsi_swir'] = safe_norm_diff(swir16, swir22)
        
        # Green-NIR normalized: (nir - green) / (nir + green)
        # Pseudo-vegetation index (since we don't have red for NDVI)
        df['ndvi_proxy'] = safe_norm_diff(nir, green)
        
        # Bare Soil Index proxy: (swir16 + green) - (nir + swir22)) / ((swir16 + green) + (nir + swir22))
        # Indicates exposed soil/mining areas
        bsi_num = (swir16 + green) - (nir + swir22)
        bsi_den = (swir16 + green) + (nir + swir22)
        df['bsi_proxy'] = np.where(bsi_den != 0, bsi_num / bsi_den, 0.0)
        
        # Built-up Index proxy: (swir16 - nir) / (swir16 + nir)
        # Higher in urban areas (relevant for EC from sewage/runoff)
        df['ndbi_proxy'] = safe_norm_diff(swir16, nir)
        # Note: ndbi_proxy is the negative of NDMI, but keeping both
        # as separate features improves interpretability
        
        # Water brightness — sum of all bands (brighter = more sediment)
        df['water_brightness'] = nir + green + swir16 + swir22
        
        # Spectral variability — std across bands
        # High variability might indicate mixed land/water pixels
        band_stack = np.column_stack([nir, green, swir16, swir22])
        df['spectral_std'] = np.nanstd(band_stack, axis=1)
        df['spectral_mean'] = np.nanmean(band_stack, axis=1)
        df['spectral_cv'] = np.where(
            df['spectral_mean'] != 0,
            df['spectral_std'] / df['spectral_mean'],
            0.0
        )
    
    # ─────────────────────────────────────────────────────
    # 3C. CLIMATE-DERIVED FEATURES
    # Research: precipitation + temperature patterns drive
    # dilution (EC, Alkalinity) and transport (DRP)
    # ─────────────────────────────────────────────────────
    
    # Precipitation patterns
    if 'ppt' in df.columns:
        # Is dry month? (< 20mm)
        df['is_dry_month'] = (df['ppt'] < 20).astype(int)
        
        # Precipitation change from previous month
        if 'ppt_lag1' in df.columns:
            df['ppt_change_1m'] = df['ppt'] - df['ppt_lag1']
            # Rainfall surge: big increase from last month → runoff event → DRP spike
            df['ppt_surge'] = np.maximum(df['ppt_change_1m'], 0)
        
        if 'ppt_lag2' in df.columns:
            df['ppt_change_2m'] = df['ppt'] - df['ppt_lag2']
        
        # Cumulative drought indicator: if last 3 months all < 30mm
        if all(c in df.columns for c in ['ppt', 'ppt_lag1', 'ppt_lag2']):
            df['drought_3m'] = (
                (df['ppt'] < 30) & (df['ppt_lag1'] < 30) & (df['ppt_lag2'] < 30)
            ).astype(int)
    
    # Temperature-evaporation dynamics
    # Research: hot dry periods concentrate salts → higher EC
    if 'tmax' in df.columns and 'ppt' in df.columns:
        # Evaporation potential relative to rainfall
        # High ratio = evaporation dominates → salt concentration
        df['evap_rain_ratio'] = np.where(
            df['ppt'] > 0,
            df.get('pet', df['tmax'] * 5) / df['ppt'],  # crude if no PET
            np.nan
        )
    
    if 'pet' in df.columns and 'aet' in df.columns:
        # Unmet evaporative demand — how much drier than potential?
        df['evap_deficit_ratio'] = np.where(
            df['pet'] > 0,
            df['aet'] / df['pet'],
            np.nan
        )
    
    # VPD patterns — atmospheric dryness
    if 'vpd' in df.columns and 'vpd_lag1' in df.columns:
        df['vpd_change_1m'] = df['vpd'] - df['vpd_lag1']
    
    # Soil moisture trends
    if 'soil' in df.columns and 'soil_lag1' in df.columns:
        df['soil_moisture_change'] = df['soil'] - df['soil_lag1']
        # Drying soil = less water holding → more concentrated runoff
    
    # Runoff patterns
    if 'q' in df.columns:
        if 'q_lag1' in df.columns:
            df['runoff_change'] = df['q'] - df['q_lag1']
        # Runoff intensity: runoff relative to precipitation
        if 'ppt' in df.columns:
            df['runoff_coeff'] = np.where(df['ppt'] > 0, df['q'] / df['ppt'], 0)
    
    # PDSI trends (drought severity)
    if 'pdsi' in df.columns and 'pdsi_lag1' in df.columns:
        df['pdsi_trend'] = df['pdsi'] - df['pdsi_lag1']
    
    # ─────────────────────────────────────────────────────
    # 3D. TERRAIN-DERIVED FEATURES
    # Research: elevation is #1 geology proxy,
    # slope is #2 feature for DRP
    # ─────────────────────────────────────────────────────
    
    if 'elevation_m' in df.columns:
        # Elevation bins — different geological zones
        # South Africa: coast (0-200m), lowveld (200-600m),
        #               highveld (600-1400m), escarpment (1400m+)
        df['elev_bin'] = pd.cut(
            df['elevation_m'],
            bins=[-np.inf, 200, 600, 1400, np.inf],
            labels=[0, 1, 2, 3]
        ).astype(float)
        
        # Log elevation (diminishing returns at higher elevations)
        df['log_elevation'] = np.log1p(df['elevation_m'].clip(lower=0))
    
    # Slope interactions with climate
    if has_slope and 'slope_deg' in df.columns:
        # Slope × precipitation → erosion/runoff potential
        # Steep slope + heavy rain = extreme phosphorus transport
        if 'ppt' in df.columns:
            df['slope_x_ppt'] = df['slope_deg'] * df['ppt']
        
        # Slope × soil clay → sediment transport potential
        if 'soil_clay' in df.columns:
            df['slope_x_clay'] = df['slope_deg'] * df['soil_clay']
    
    # ─────────────────────────────────────────────────────
    # 3E. SOIL-GEOLOGY INTERACTIONS
    # Research: soil mineralogy determines alkalinity and EC
    # ─────────────────────────────────────────────────────
    
    # Soil pH × CEC interaction — buffering capacity proxy
    # High pH + high CEC = alkaline, well-buffered water
    if 'soil_phh2o' in df.columns and 'soil_cec' in df.columns:
        df['soil_buffer_capacity'] = df['soil_phh2o'] * df['soil_cec']
    
    # Sand/Clay ratio — drainage indicator
    # Sandy soil = fast drainage, less mineral contact time
    # Clay soil = slow drainage, more mineral dissolution
    if 'soil_sand' in df.columns and 'soil_clay' in df.columns:
        df['soil_sand_clay_ratio'] = np.where(
            df['soil_clay'] > 0,
            df['soil_sand'] / df['soil_clay'],
            np.nan
        )
    
    # SOC/Nitrogen ratio — nutrient status
    if 'soil_soc' in df.columns and 'soil_nitrogen' in df.columns:
        df['soil_cn_ratio'] = np.where(
            df['soil_nitrogen'] > 0,
            df['soil_soc'] / df['soil_nitrogen'],
            np.nan
        )
    
    # Soil moisture × soil clay → phosphorus retention capacity
    # Clay soils hold more phosphorus; wet clay releases more
    if 'soil' in df.columns and 'soil_clay' in df.columns:
        df['moisture_x_clay'] = df['soil'] * df['soil_clay']
    
    # Soil pH × precipitation → leaching potential
    # Acid rain on alkaline soil = more mineral dissolution
    if 'soil_phh2o' in df.columns and 'ppt' in df.columns:
        df['soilph_x_ppt'] = df['soil_phh2o'] * df['ppt']
    
    # ─────────────────────────────────────────────────────
    # 3F. ELEVATION × CLIMATE INTERACTIONS
    # Higher elevation = cooler, different rain patterns
    # ─────────────────────────────────────────────────────
    
    if 'elevation_m' in df.columns:
        if 'tmax' in df.columns:
            # Temperature at elevation — captures lapse rate
            df['elev_x_tmax'] = df['elevation_m'] * df['tmax']
        
        if 'ppt' in df.columns:
            # Orographic rainfall interaction
            df['elev_x_ppt'] = df['elevation_m'] * df['ppt']
    
    # ─────────────────────────────────────────────────────
    # 3G. LANDSAT × CLIMATE INTERACTIONS
    # Satellite appearance under different climate conditions
    # ─────────────────────────────────────────────────────
    
    if has_landsat and 'ppt' in df.columns:
        # Water index × rainfall — wet periods change reflectance
        if 'MNDWI' in df.columns:
            df['mndwi_x_ppt'] = df['MNDWI'] * df['ppt']
        if 'NDMI' in df.columns:
            df['ndmi_x_ppt'] = df['NDMI'] * df['ppt']
    
    if has_landsat and 'tmax' in df.columns:
        # SWIR × temperature — evaporation effect on surface
        df['swir16_x_tmax'] = df['swir16'] * df['tmax']
    
    n_new = len(df.columns) - n_before
    print(f"  [{dataset_name}] Added {n_new} engineered features → total columns: {len(df.columns)}")
    
    return df


# Apply to both datasets
train_df = engineer_features(train_df, 'TRAIN')
val_df = engineer_features(val_df, 'VAL')


# ============================================================
# [4] CLEAN UP AND IDENTIFY FEATURE COLUMNS
# ============================================================
print(f"\n[4] Identifying feature columns...")

TARGETS = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
META_COLS = ['Latitude', 'Longitude', 'Sample Date'] + TARGETS

# All columns that are NOT metadata/targets are features
feature_cols = [c for c in train_df.columns if c not in META_COLS]

# Categorize features for reporting
landsat_features = [c for c in feature_cols if any(c.startswith(p) for p in 
    ['nir', 'green', 'swir', 'NDMI', 'MNDWI', 'ratio_', 'ndsi_', 'ndvi_', 
     'bsi_', 'ndbi_', 'water_', 'spectral_'])]
terra_features = [c for c in feature_cols if any(c.startswith(p) for p in 
    ['ppt', 'tmax', 'tmin', 'soil_lag', 'aet', 'def', 'q', 'vpd', 'pdsi', 
     'pet', 'srad', 'temp_range', 'aridity']) and 'soil_' not in c or
    c.startswith(('soil_lag', 'soil_moisture'))]
# Fix: separate soil_moisture from soil properties
terra_features = [c for c in feature_cols if 
    (c.split('_')[0] in ['ppt', 'tmax', 'tmin', 'aet', 'def', 'q', 'vpd', 'pdsi', 'pet', 'srad']) or
    c in ['temp_range', 'aridity_index', 'ppt_3m_avg', 'ppt_3m_sum', 'tmax_3m_avg', 'tmin_3m_avg',
           'soil', 'soil_lag1', 'soil_lag2', 'soil_lag3', 'soil_moisture_change']]
elev_features = [c for c in feature_cols if any(c.startswith(p) for p in 
    ['elevation', 'log_elevation', 'elev_', 'slope', 'aspect', 'terrain_'])]
soil_prop_features = [c for c in feature_cols if c.startswith('soil_') and 
    c not in terra_features and 'lag' not in c and 'moisture' not in c]
temporal_features = [c for c in feature_cols if c in 
    ['month', 'year', 'day_of_year', 'month_sin', 'month_cos', 'doy_sin', 
     'doy_cos', 'season', 'is_wet_season']]
interaction_features = [c for c in feature_cols if '_x_' in c]
derived_features = [c for c in feature_cols if c not in 
    landsat_features + terra_features + elev_features + soil_prop_features + 
    temporal_features + interaction_features]

print(f"\n  Feature breakdown:")
print(f"    Landsat spectral:      {len(landsat_features):3d} features")
print(f"    TerraClimate + lags:   {len(terra_features):3d} features")
print(f"    Elevation/Slope:       {len(elev_features):3d} features")
print(f"    Soil properties:       {len(soil_prop_features):3d} features")
print(f"    Temporal:              {len(temporal_features):3d} features")
print(f"    Interactions:          {len(interaction_features):3d} features")
print(f"    Other derived:         {len(derived_features):3d} features")
print(f"    {'─' * 45}")
print(f"    TOTAL:                 {len(feature_cols):3d} features")


# ============================================================
# [5] DATA QUALITY REPORT
# ============================================================
print(f"\n[5] Data quality report...")

for name, df in [('Train', train_df), ('Val', val_df)]:
    n_samples = len(df)
    print(f"\n  {name} ({n_samples} samples):")
    
    # NaN analysis by feature group
    groups = {
        'Landsat': landsat_features,
        'TerraClimate': terra_features,
        'Elevation/Slope': elev_features,
        'Soil': soil_prop_features,
        'Temporal': temporal_features,
        'Interactions': interaction_features,
    }
    
    for group_name, cols in groups.items():
        if not cols:
            continue
        existing_cols = [c for c in cols if c in df.columns]
        if not existing_cols:
            continue
        nan_pct = df[existing_cols].isna().mean().mean() * 100
        worst_col = df[existing_cols].isna().mean().idxmax()
        worst_pct = df[existing_cols].isna().mean().max() * 100
        print(f"    {group_name:20s}: avg {nan_pct:5.1f}% NaN, worst: {worst_col} ({worst_pct:.1f}%)")

# Target distribution
print(f"\n  Target distributions (train):")
for target in TARGETS:
    vals = train_df[target]
    print(f"    {target}:")
    print(f"      Range: [{vals.min():.2f}, {vals.max():.2f}]")
    print(f"      Mean: {vals.mean():.2f}, Median: {vals.median():.2f}")
    print(f"      Skewness: {vals.skew():.2f}")
    if vals.skew() > 2:
        print(f"      → HIGHLY SKEWED — consider log transform")


# ============================================================
# [6] SAVE MODEL-READY DATASETS
# ============================================================
print(f"\n[6] Saving model-ready datasets...")

# Save full datasets
train_path = os.path.join(EXTRACTED_DIR, 'v4_train_features.csv')
val_path = os.path.join(EXTRACTED_DIR, 'v4_val_features.csv')

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)

print(f"  Saved: {train_path}")
print(f"    Shape: {train_df.shape}")
print(f"  Saved: {val_path}")
print(f"    Shape: {val_df.shape}")

# Also save feature column list for the model script
feature_list_path = os.path.join(EXTRACTED_DIR, 'v4_feature_columns.txt')
with open(feature_list_path, 'w') as f:
    f.write('\n'.join(feature_cols))
print(f"  Saved: {feature_list_path} ({len(feature_cols)} features)")


# ============================================================
# [7] PREVIEW
# ============================================================
print(f"\n{'=' * 70}")
print(f"V4 FEATURE ENGINEERING COMPLETE")
print(f"{'=' * 70}")

print(f"\n  Train shape: {train_df.shape}")
print(f"  Val shape:   {val_df.shape}")
print(f"  Features:    {len(feature_cols)}")

print(f"\n  Sample feature values (first 3 train rows):")
sample_features = ['elevation_m', 'slope_deg', 'ppt', 'tmax', 'soil_phh2o', 
                   'NDMI', 'ratio_nir_green', 'month_sin', 'ppt_surge',
                   'soil_buffer_capacity', 'slope_x_ppt']
available_sample = [c for c in sample_features if c in train_df.columns]
if available_sample:
    print(train_df[available_sample].head(3).to_string())

print(f"""
NEXT STEPS:
  1. If slope_data.csv doesn't exist yet:
     Run: python scripts/07_extract_slope.py  (2-5 min)
     Then re-run this script

  2. Run the V4 model:
     python scripts/09_model_v4.py
     
  Features ready for modeling:
    - {len(landsat_features)} spectral features (bands + ratios + indices)
    - {len(terra_features)} climate features (current + 3 lags + derived)
    - {len(elev_features)} terrain features (elevation + slope + roughness)
    - {len(soil_prop_features)} soil features (pH, clay, CEC, SOC, etc.)
    - {len(temporal_features)} temporal features (month, season, cyclical)
    - {len(interaction_features)} cross-domain interactions
    = {len(feature_cols)} total features
""")

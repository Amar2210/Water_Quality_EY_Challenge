"""
============================================================
EY Water Quality Challenge - V5 Feature Engineering
============================================================
WHAT'S NEW IN V5:
  - Sentinel-2 L2A bands and indices (10m resolution!)
  - Red edge bands for chlorophyll / eutrophication detection
  - Cross-sensor features (Sentinel-2 x Climate x Terrain)
  - All V4 features preserved and enhanced

SENTINEL-2 COVERAGE:
  - Sentinel-2A launched June 2015
  - Training data spans 2011-2023: ~40-60% of samples have S2 data
  - Validation data spans 2012-2015: most/all may be pre-S2 (all NaN)
  - XGBoost handles NaN natively -> no imputation needed
  - has_sentinel2 flag lets the model learn when S2 is available

DATA SOURCES:
  1. Landsat features (nir, green, swir16, swir22, NDMI, MNDWI)
  2. TerraClimate expanded (50 climate vars + lags)
  3. Elevation data (elevation_m)
  4. Slope data (slope_deg, aspect_deg, terrain_roughness, elev_range_500m)
  5. Soil data (soil_phh2o, soil_clay, soil_sand, soil_soc, soil_cec, soil_nitrogen)
  6. Sentinel-2 data (11 bands + 9 indices) -- NEW IN V5

OUTPUT:
  data/extracted/v5_train_features.csv
  data/extracted/v5_val_features.csv
  data/extracted/v5_feature_columns.txt

RUNTIME: ~10 seconds (no API calls, just math)
============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'Jupyter Notebook Package')
EXTRACTED_DIR = os.path.join(BASE_DIR, '..', 'data', 'extracted')

print("=" * 70)
print("EY WATER QUALITY - V5 FEATURE ENGINEERING")
print("V4 features + Sentinel-2 bands/indices + cross-sensor interactions")
print("=" * 70)


# ============================================================
# [1] LOAD ALL DATA SOURCES
# ============================================================
print("\n" + "=" * 70)
print("[1] LOADING ALL DATA SOURCES")
print("=" * 70)

# --- Targets (training only) ---
wq_train = pd.read_csv(os.path.join(DATA_DIR, 'water_quality_training_dataset.csv'))
submission_template = pd.read_csv(os.path.join(DATA_DIR, 'submission_template.csv'))
print(f"  Training samples:    {len(wq_train)}")
print(f"  Validation samples:  {len(submission_template)}")

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

# --- Slope ---
slope_path = os.path.join(EXTRACTED_DIR, 'slope_data.csv')
has_slope = os.path.exists(slope_path)
if has_slope:
    slope = pd.read_csv(slope_path)
    print(f"  Slope: {len(slope)} locations, valid: {slope['slope_deg'].notna().sum()}")
else:
    slope = None
    print(f"  Slope: NOT FOUND (continuing without)")

# --- Soil ---
soil = pd.read_csv(os.path.join(EXTRACTED_DIR, 'soil_data.csv'))
soil_cols = [c for c in soil.columns if c.startswith('soil_')]
print(f"  Soil: {len(soil)} locations, properties: {soil_cols}")

# --- Sentinel-2 (NEW in V5) ---
s2_train_path = os.path.join(EXTRACTED_DIR, 'sentinel2_training.csv')
s2_val_path = os.path.join(EXTRACTED_DIR, 'sentinel2_validation.csv')

has_s2 = os.path.exists(s2_train_path) and os.path.exists(s2_val_path)
if has_s2:
    s2_train_raw = pd.read_csv(s2_train_path)
    s2_val_raw = pd.read_csv(s2_val_path)
    # Identify Sentinel-2 specific columns (exclude join keys and targets)
    TARGETS = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
    META_KEYS = ['Latitude', 'Longitude', 'Sample Date']
    s2_band_cols = [c for c in s2_train_raw.columns if c.startswith('s2_')]
    # Only keep join keys + s2 columns for merging
    s2_train = s2_train_raw[META_KEYS + s2_band_cols].copy()
    s2_val = s2_val_raw[META_KEYS + s2_band_cols].copy()

    # Coverage stats
    s2_any_valid_train = s2_train[s2_band_cols].notna().any(axis=1).sum()
    s2_all_nan_train = s2_train[s2_band_cols].isna().all(axis=1).sum()
    s2_any_valid_val = s2_val[s2_band_cols].notna().any(axis=1).sum()
    s2_all_nan_val = s2_val[s2_band_cols].isna().all(axis=1).sum()

    print(f"  Sentinel-2 train: {s2_train.shape}")
    print(f"    Rows with S2 data: {s2_any_valid_train} ({100*s2_any_valid_train/len(s2_train):.1f}%)")
    print(f"    Rows all-NaN:      {s2_all_nan_train} ({100*s2_all_nan_train/len(s2_train):.1f}%)")
    print(f"  Sentinel-2 val:   {s2_val.shape}")
    print(f"    Rows with S2 data: {s2_any_valid_val} ({100*s2_any_valid_val/len(s2_val):.1f}%)")
    print(f"    Rows all-NaN:      {s2_all_nan_val} ({100*s2_all_nan_val/len(s2_val):.1f}%)")
    print(f"  S2 columns: {s2_band_cols}")
else:
    s2_train = None
    s2_val = None
    s2_band_cols = []
    print(f"  Sentinel-2: NOT FOUND (run extract_sentinel2.py first)")
    print(f"  Continuing without Sentinel-2 features...")


# ============================================================
# [2] MERGE ALL SOURCES
# ============================================================
print("\n" + "=" * 70)
print("[2] MERGING ALL DATA SOURCES")
print("=" * 70)

join_keys = ['Latitude', 'Longitude', 'Sample Date']
loc_keys = ['Latitude', 'Longitude']


def round_coords(df, decimals=6):
    """Round lat/lon to avoid floating point merge mismatches."""
    df = df.copy()
    df['Latitude'] = df['Latitude'].round(decimals)
    df['Longitude'] = df['Longitude'].round(decimals)
    return df


# Round coordinates on ALL dataframes for clean merging
wq_train = round_coords(wq_train)
submission_template = round_coords(submission_template)
landsat_train = round_coords(landsat_train)
landsat_val = round_coords(landsat_val)
terra_train = round_coords(terra_train)
terra_val = round_coords(terra_val)
elevation = round_coords(elevation)
soil = round_coords(soil)
if has_slope:
    slope = round_coords(slope)
if has_s2:
    s2_train = round_coords(s2_train)
    s2_val = round_coords(s2_val)

# --- Build training set ---
print("\n  Merging training data...")
train_df = wq_train.copy()
n0 = len(train_df)

train_df = train_df.merge(landsat_train, on=join_keys, how='left')
print(f"    + Landsat:      {train_df.shape[1]} cols ({len(train_df)} rows)")

train_df = train_df.merge(terra_train, on=join_keys, how='left')
print(f"    + TerraClimate: {train_df.shape[1]} cols ({len(train_df)} rows)")

train_df = train_df.merge(elevation, on=loc_keys, how='left')
print(f"    + Elevation:    {train_df.shape[1]} cols ({len(train_df)} rows)")

train_df = train_df.merge(soil, on=loc_keys, how='left')
print(f"    + Soil:         {train_df.shape[1]} cols ({len(train_df)} rows)")

if has_slope:
    train_df = train_df.merge(slope, on=loc_keys, how='left')
    print(f"    + Slope:        {train_df.shape[1]} cols ({len(train_df)} rows)")

if has_s2:
    train_df = train_df.merge(s2_train, on=join_keys, how='left')
    print(f"    + Sentinel-2:   {train_df.shape[1]} cols ({len(train_df)} rows)")

# Check for row inflation from merge
if len(train_df) != n0:
    print(f"  WARNING: Train row count changed from {n0} to {len(train_df)} after merge!")
    print(f"           Dropping duplicates...")
    train_df = train_df.drop_duplicates(subset=join_keys, keep='first')
    print(f"           After dedup: {len(train_df)} rows")

# --- Build validation set ---
print("\n  Merging validation data...")
val_df = submission_template.copy()
n0_val = len(val_df)

val_df = val_df.merge(landsat_val, on=join_keys, how='left')
val_df = val_df.merge(terra_val, on=join_keys, how='left')
val_df = val_df.merge(elevation, on=loc_keys, how='left')
val_df = val_df.merge(soil, on=loc_keys, how='left')
if has_slope:
    val_df = val_df.merge(slope, on=loc_keys, how='left')
if has_s2:
    val_df = val_df.merge(s2_val, on=join_keys, how='left')

if len(val_df) != n0_val:
    print(f"  WARNING: Val row count changed from {n0_val} to {len(val_df)} after merge!")
    val_df = val_df.drop_duplicates(subset=join_keys, keep='first')
    print(f"           After dedup: {len(val_df)} rows")

print(f"\n  Train after merge: {train_df.shape}")
print(f"  Val after merge:   {val_df.shape}")

# Clean up duplicate columns from merge
for df_ref, name in [(train_df, 'train'), (val_df, 'val')]:
    dupe_cols = [c for c in df_ref.columns if c.endswith('_x') or c.endswith('_y')]
    if dupe_cols:
        print(f"  Cleaning duplicate columns in {name}: {dupe_cols}")
        for col in list(dupe_cols):
            if col.endswith('_y'):
                df_ref.drop(columns=[col], inplace=True)
            elif col.endswith('_x'):
                df_ref.rename(columns={col: col.replace('_x', '')}, inplace=True)


# ============================================================
# [3] FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 70)
print("[3] ENGINEERING FEATURES")
print("=" * 70)


def engineer_features(df, dataset_name=''):
    """
    Apply ALL feature engineering. Includes:
      - V4 features (temporal, landsat ratios, climate, terrain, soil, interactions)
      - V5 NEW: Sentinel-2 derived features
      - V5 NEW: Cross-sensor features
    """
    n_before = len(df.columns)

    # ==========================================================
    # 3A. TEMPORAL FEATURES
    # ==========================================================
    df['parsed_date'] = pd.to_datetime(df['Sample Date'], dayfirst=True, errors='coerce')
    df['month'] = df['parsed_date'].dt.month
    df['year'] = df['parsed_date'].dt.year
    df['day_of_year'] = df['parsed_date'].dt.dayofyear

    # Cyclical encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    # Southern Hemisphere seasons
    df['season'] = df['month'].map({
        12: 0, 1: 0, 2: 0,   # Summer
        3: 1, 4: 1, 5: 1,    # Autumn
        6: 2, 7: 2, 8: 2,    # Winter
        9: 3, 10: 3, 11: 3   # Spring
    })
    df['is_wet_season'] = df['month'].isin([10, 11, 12, 1, 2, 3]).astype(int)

    df.drop(columns=['parsed_date'], inplace=True, errors='ignore')

    # ==========================================================
    # 3B. LANDSAT BAND RATIOS & INDICES (from V4)
    # ==========================================================
    has_landsat = 'nir' in df.columns and 'green' in df.columns

    if has_landsat:
        def safe_ratio(a, b, fill=0.0):
            return np.where(b != 0, a / b, fill)

        def safe_norm_diff(a, b, fill=0.0):
            denom = a + b
            return np.where(denom != 0, (a - b) / denom, fill)

        nir = df['nir'].values.astype(float)
        green = df['green'].values.astype(float)
        swir16 = df['swir16'].values.astype(float)
        swir22 = df['swir22'].values.astype(float)

        # Band ratios
        df['ratio_nir_green'] = safe_ratio(nir, green)
        df['ratio_swir16_nir'] = safe_ratio(swir16, nir)
        df['ratio_swir22_swir16'] = safe_ratio(swir22, swir16)
        df['ratio_green_swir16'] = safe_ratio(green, swir16)
        df['ratio_nir_swir22'] = safe_ratio(nir, swir22)

        # Normalized difference indices
        df['ndsi_swir'] = safe_norm_diff(swir16, swir22)
        df['ndvi_proxy'] = safe_norm_diff(nir, green)
        bsi_num = (swir16 + green) - (nir + swir22)
        bsi_den = (swir16 + green) + (nir + swir22)
        df['bsi_proxy'] = np.where(bsi_den != 0, bsi_num / bsi_den, 0.0)
        df['ndbi_proxy'] = safe_norm_diff(swir16, nir)

        # Aggregate spectral features
        df['water_brightness'] = nir + green + swir16 + swir22
        band_stack = np.column_stack([nir, green, swir16, swir22])
        df['spectral_std'] = np.nanstd(band_stack, axis=1)
        df['spectral_mean'] = np.nanmean(band_stack, axis=1)
        df['spectral_cv'] = np.where(
            df['spectral_mean'] != 0,
            df['spectral_std'] / df['spectral_mean'],
            0.0
        )

    # ==========================================================
    # 3C. CLIMATE-DERIVED FEATURES (from V4)
    # ==========================================================
    if 'ppt' in df.columns:
        df['is_dry_month'] = (df['ppt'] < 20).astype(int)
        if 'ppt_lag1' in df.columns:
            df['ppt_change_1m'] = df['ppt'] - df['ppt_lag1']
            df['ppt_surge'] = np.maximum(df['ppt_change_1m'], 0)
        if 'ppt_lag2' in df.columns:
            df['ppt_change_2m'] = df['ppt'] - df['ppt_lag2']
        if all(c in df.columns for c in ['ppt', 'ppt_lag1', 'ppt_lag2']):
            df['drought_3m'] = (
                (df['ppt'] < 30) & (df['ppt_lag1'] < 30) & (df['ppt_lag2'] < 30)
            ).astype(int)

    if 'tmax' in df.columns and 'ppt' in df.columns:
        df['evap_rain_ratio'] = np.where(
            df['ppt'] > 0,
            df.get('pet', df['tmax'] * 5) / df['ppt'],
            np.nan
        )

    if 'pet' in df.columns and 'aet' in df.columns:
        df['evap_deficit_ratio'] = np.where(
            df['pet'] > 0, df['aet'] / df['pet'], np.nan
        )

    if 'vpd' in df.columns and 'vpd_lag1' in df.columns:
        df['vpd_change_1m'] = df['vpd'] - df['vpd_lag1']

    if 'soil' in df.columns and 'soil_lag1' in df.columns:
        df['soil_moisture_change'] = df['soil'] - df['soil_lag1']

    if 'q' in df.columns:
        if 'q_lag1' in df.columns:
            df['runoff_change'] = df['q'] - df['q_lag1']
        if 'ppt' in df.columns:
            df['runoff_coeff'] = np.where(df['ppt'] > 0, df['q'] / df['ppt'], 0)

    if 'pdsi' in df.columns and 'pdsi_lag1' in df.columns:
        df['pdsi_trend'] = df['pdsi'] - df['pdsi_lag1']

    # ==========================================================
    # 3D. TERRAIN-DERIVED FEATURES (from V4)
    # ==========================================================
    if 'elevation_m' in df.columns:
        df['elev_bin'] = pd.cut(
            df['elevation_m'],
            bins=[-np.inf, 200, 600, 1400, np.inf],
            labels=[0, 1, 2, 3]
        ).astype(float)
        df['log_elevation'] = np.log1p(df['elevation_m'].clip(lower=0))

    if 'slope_deg' in df.columns and 'ppt' in df.columns:
        df['slope_x_ppt'] = df['slope_deg'] * df['ppt']
    if 'slope_deg' in df.columns and 'soil_clay' in df.columns:
        df['slope_x_clay'] = df['slope_deg'] * df['soil_clay']

    # ==========================================================
    # 3E. SOIL-GEOLOGY INTERACTIONS (from V4)
    # ==========================================================
    if 'soil_phh2o' in df.columns and 'soil_cec' in df.columns:
        df['soil_buffer_capacity'] = df['soil_phh2o'] * df['soil_cec']
    if 'soil_sand' in df.columns and 'soil_clay' in df.columns:
        df['soil_sand_clay_ratio'] = np.where(
            df['soil_clay'] > 0, df['soil_sand'] / df['soil_clay'], np.nan
        )
    if 'soil_soc' in df.columns and 'soil_nitrogen' in df.columns:
        df['soil_cn_ratio'] = np.where(
            df['soil_nitrogen'] > 0, df['soil_soc'] / df['soil_nitrogen'], np.nan
        )
    if 'soil' in df.columns and 'soil_clay' in df.columns:
        df['moisture_x_clay'] = df['soil'] * df['soil_clay']
    if 'soil_phh2o' in df.columns and 'ppt' in df.columns:
        df['soilph_x_ppt'] = df['soil_phh2o'] * df['ppt']

    # ==========================================================
    # 3F. ELEVATION x CLIMATE (from V4)
    # ==========================================================
    if 'elevation_m' in df.columns:
        if 'tmax' in df.columns:
            df['elev_x_tmax'] = df['elevation_m'] * df['tmax']
        if 'ppt' in df.columns:
            df['elev_x_ppt'] = df['elevation_m'] * df['ppt']

    # ==========================================================
    # 3G. LANDSAT x CLIMATE (from V4)
    # ==========================================================
    if has_landsat and 'ppt' in df.columns:
        if 'MNDWI' in df.columns:
            df['mndwi_x_ppt'] = df['MNDWI'] * df['ppt']
        if 'NDMI' in df.columns:
            df['ndmi_x_ppt'] = df['NDMI'] * df['ppt']
    if has_landsat and 'tmax' in df.columns:
        df['swir16_x_tmax'] = df['swir16'] * df['tmax']

    # ==========================================================
    # 3H. SENTINEL-2 DERIVED FEATURES (NEW IN V5)
    # ==========================================================
    s2_cols_present = [c for c in df.columns if c.startswith('s2_')]

    if s2_cols_present:
        print(f"    [{dataset_name}] Engineering Sentinel-2 features...")

        # Binary flag: does this sample have any S2 data?
        s2_raw_bands = [c for c in s2_cols_present if c in [
            's2_blue', 's2_green', 's2_red', 's2_re1', 's2_re2', 's2_re3',
            's2_nir', 's2_nir8a', 's2_wv', 's2_swir1', 's2_swir2'
        ]]
        if s2_raw_bands:
            df['has_sentinel2'] = df[s2_raw_bands].notna().any(axis=1).astype(int)
        else:
            df['has_sentinel2'] = 0

        eps = 1e-8

        # --- Chlorophyll index (red edge - red) / red edge ---
        # Chlorophyll absorbs red, reflects red edge -> eutrophication proxy -> DRP
        if 's2_re1' in df.columns and 's2_red' in df.columns:
            df['s2_chlorophyll_index'] = (df['s2_re1'] - df['s2_red']) / (df['s2_re1'] + eps)

        # --- Turbidity index: red / green ---
        # Turbid water: more red scattering, less green penetration
        if 's2_red' in df.columns and 's2_green' in df.columns:
            df['s2_turbidity_index'] = df['s2_red'] / (df['s2_green'] + eps)

        # --- Blue to Red-Edge ratio (water clarity) ---
        if 's2_blue' in df.columns and 's2_re1' in df.columns:
            df['s2_bre_ratio'] = df['s2_blue'] / (df['s2_re1'] + eps)

        # --- NIR - Red Edge 1 difference (vegetation/algae detector) ---
        if 's2_nir' in df.columns and 's2_re1' in df.columns:
            df['s2_nir_re_diff'] = df['s2_nir'] - df['s2_re1']

        # --- SWIR/NIR ratio (moisture content) ---
        if 's2_swir1' in df.columns and 's2_nir' in df.columns:
            df['s2_swir_nir_ratio'] = df['s2_swir1'] / (df['s2_nir'] + eps)

        # --- Red edge slope (vegetation health gradient) ---
        if all(c in df.columns for c in ['s2_re3', 's2_re1']):
            df['s2_re_slope'] = (df['s2_re3'] - df['s2_re1']) / 3.0

        # --- Overall water brightness (visible bands) ---
        if all(c in df.columns for c in ['s2_blue', 's2_green', 's2_red']):
            df['s2_brightness'] = (df['s2_blue'] + df['s2_green'] + df['s2_red']) / 3.0

        # --- Visible/NIR ratio (water vs vegetation) ---
        if 's2_brightness' in df.columns and 's2_nir' in df.columns:
            df['s2_vis_nir_ratio'] = df['s2_brightness'] / (df['s2_nir'] + eps)

        # --- Green/Red ratio (clearer water = higher) ---
        if 's2_green' in df.columns and 's2_red' in df.columns:
            df['s2_green_red_ratio'] = df['s2_green'] / (df['s2_red'] + eps)

        # --- SWIR difference (mineral/sediment indicator) ---
        if 's2_swir1' in df.columns and 's2_swir2' in df.columns:
            df['s2_swir_diff'] = df['s2_swir1'] - df['s2_swir2']

        # --- Red edge normalized: (RE2 - RE1) / (RE2 + RE1) ---
        if 's2_re2' in df.columns and 's2_re1' in df.columns:
            df['s2_re_nd'] = (df['s2_re2'] - df['s2_re1']) / (df['s2_re2'] + df['s2_re1'] + eps)

    # ==========================================================
    # 3I. CROSS-SENSOR FEATURES: SENTINEL-2 x CLIMATE x TERRAIN (NEW V5)
    # ==========================================================
    if s2_cols_present:
        print(f"    [{dataset_name}] Engineering cross-sensor features...")

        # S2 NDVI x precipitation -> vegetation response to rainfall
        if 's2_ndvi' in df.columns and 'ppt' in df.columns:
            df['s2_ndvi_x_ppt'] = df['s2_ndvi'] * df['ppt']

        # S2 turbidity x slope -> erosion-driven turbidity
        if 's2_turbidity_index' in df.columns and 'slope_deg' in df.columns:
            df['s2_turbidity_x_slope'] = df['s2_turbidity_index'] * df['slope_deg']

        # S2 chlorophyll x temperature -> algal growth potential
        if 's2_chlorophyll_index' in df.columns and 'tmax' in df.columns:
            df['s2_chlorophyll_x_temp'] = df['s2_chlorophyll_index'] * df['tmax']

        # S2 NDWI x precipitation -> water body extent and rainfall
        if 's2_ndwi' in df.columns and 'ppt' in df.columns:
            df['s2_ndwi_x_ppt'] = df['s2_ndwi'] * df['ppt']

        # S2 brightness x elevation -> altitude optical effect
        if 's2_brightness' in df.columns and 'elevation_m' in df.columns:
            df['s2_brightness_x_elev'] = df['s2_brightness'] * df['elevation_m']

        # S2 MNDWI x soil moisture -> combined water indicator
        if 's2_mndwi' in df.columns and 'soil' in df.columns:
            df['s2_mndwi_x_soilmoist'] = df['s2_mndwi'] * df['soil']

        # S2 NDRE x soil nitrogen -> nutrient availability + vegetation response
        if 's2_ndre' in df.columns and 'soil_nitrogen' in df.columns:
            df['s2_ndre_x_nitrogen'] = df['s2_ndre'] * df['soil_nitrogen']

        # S2 FAI x temperature -> floating algae in warm conditions
        if 's2_fai' in df.columns and 'tmax' in df.columns:
            df['s2_fai_x_temp'] = df['s2_fai'] * df['tmax']

    # ==========================================================
    # 3J. LANDSAT x SENTINEL-2 CROSS-FEATURES (NEW V5)
    # ==========================================================
    if has_landsat and s2_cols_present:
        # NDMI (Landsat) vs NDWI (Sentinel-2) difference -> resolution sensitivity
        if 'NDMI' in df.columns and 's2_ndwi' in df.columns:
            df['landsat_s2_water_diff'] = df['NDMI'] - df['s2_ndwi']

        # Landsat MNDWI vs S2 MNDWI -> sensor comparison
        if 'MNDWI' in df.columns and 's2_mndwi' in df.columns:
            df['landsat_s2_mndwi_diff'] = df['MNDWI'] - df['s2_mndwi']

    n_new = len(df.columns) - n_before
    print(f"  [{dataset_name}] Added {n_new} engineered features -> total columns: {len(df.columns)}")

    return df


# Apply to both datasets
train_df = engineer_features(train_df, 'TRAIN')
val_df = engineer_features(val_df, 'VAL')


# ============================================================
# [4] IDENTIFY FEATURE COLUMNS
# ============================================================
print("\n" + "=" * 70)
print("[4] FEATURE COLUMN IDENTIFICATION")
print("=" * 70)

TARGETS = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
META_COLS = ['Latitude', 'Longitude', 'Sample Date'] + TARGETS

feature_cols = [c for c in train_df.columns if c not in META_COLS]

# Categorize
landsat_features = [c for c in feature_cols if any(c.startswith(p) for p in
    ['nir', 'green', 'swir', 'NDMI', 'MNDWI', 'ratio_', 'ndsi_', 'ndvi_',
     'bsi_', 'ndbi_', 'water_brightness', 'spectral_'])]
s2_features = [c for c in feature_cols if c.startswith('s2_') or c == 'has_sentinel2']
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
interaction_features = [c for c in feature_cols if '_x_' in c and not c.startswith('s2_')]
cross_sensor_features = [c for c in feature_cols if
    (c.startswith('s2_') and '_x_' in c) or c.startswith('landsat_s2_')]
derived_features = [c for c in feature_cols if c not in
    landsat_features + s2_features + terra_features + elev_features +
    soil_prop_features + temporal_features + interaction_features + cross_sensor_features]

print(f"\n  Feature breakdown:")
print(f"    Landsat spectral:        {len(landsat_features):3d} features")
print(f"    Sentinel-2 (NEW):        {len(s2_features):3d} features")
print(f"    TerraClimate + lags:     {len(terra_features):3d} features")
print(f"    Elevation/Slope:         {len(elev_features):3d} features")
print(f"    Soil properties:         {len(soil_prop_features):3d} features")
print(f"    Temporal:                {len(temporal_features):3d} features")
print(f"    V4 Interactions:         {len(interaction_features):3d} features")
print(f"    Cross-sensor (NEW):      {len(cross_sensor_features):3d} features")
print(f"    Other derived:           {len(derived_features):3d} features")
print(f"    {'=' * 50}")
print(f"    TOTAL:                   {len(feature_cols):3d} features")

if s2_features:
    print(f"\n  Sentinel-2 feature list:")
    for f in sorted(s2_features):
        print(f"    - {f}")
if cross_sensor_features:
    print(f"\n  Cross-sensor feature list:")
    for f in sorted(cross_sensor_features):
        print(f"    - {f}")


# ============================================================
# [5] DATA QUALITY REPORT
# ============================================================
print("\n" + "=" * 70)
print("[5] DATA QUALITY REPORT")
print("=" * 70)

for name, df_ref in [('Train', train_df), ('Val', val_df)]:
    n_samples = len(df_ref)
    print(f"\n  {name} ({n_samples} samples):")

    groups = {
        'Landsat': landsat_features,
        'Sentinel-2': s2_features,
        'TerraClimate': terra_features,
        'Elevation/Slope': elev_features,
        'Soil': soil_prop_features,
        'Temporal': temporal_features,
        'Interactions': interaction_features,
        'Cross-sensor': cross_sensor_features,
    }

    for group_name, cols in groups.items():
        if not cols:
            continue
        existing_cols = [c for c in cols if c in df_ref.columns]
        if not existing_cols:
            continue
        nan_pct = df_ref[existing_cols].isna().mean().mean() * 100
        worst_col = df_ref[existing_cols].isna().mean().idxmax()
        worst_pct = df_ref[existing_cols].isna().mean().max() * 100
        print(f"    {group_name:20s}: avg {nan_pct:5.1f}% NaN, worst: {worst_col} ({worst_pct:.1f}%)")

# Target distribution
print(f"\n  Target distributions (train):")
for target in TARGETS:
    if target in train_df.columns:
        vals = train_df[target]
        print(f"    {target}:")
        print(f"      Range: [{vals.min():.2f}, {vals.max():.2f}]")
        print(f"      Mean: {vals.mean():.2f}, Median: {vals.median():.2f}")
        print(f"      Skewness: {vals.skew():.2f}")
        if vals.skew() > 2:
            print(f"      -> HIGHLY SKEWED -- consider log transform")


# ============================================================
# [6] SAVE MODEL-READY DATASETS
# ============================================================
print("\n" + "=" * 70)
print("[6] SAVING MODEL-READY DATASETS")
print("=" * 70)

train_path = os.path.join(EXTRACTED_DIR, 'v5_train_features.csv')
val_path = os.path.join(EXTRACTED_DIR, 'v5_val_features.csv')

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)

print(f"  Saved: {train_path}")
print(f"    Shape: {train_df.shape}")
print(f"  Saved: {val_path}")
print(f"    Shape: {val_df.shape}")

# Save feature column list
feature_list_path = os.path.join(EXTRACTED_DIR, 'v5_feature_columns.txt')
with open(feature_list_path, 'w') as f:
    f.write('\n'.join(feature_cols))
print(f"  Saved: {feature_list_path} ({len(feature_cols)} features)")


# ============================================================
# [7] SUMMARY
# ============================================================
print(f"\n{'=' * 70}")
print(f"V5 FEATURE ENGINEERING COMPLETE")
print(f"{'=' * 70}")

print(f"""
  Train shape: {train_df.shape}
  Val shape:   {val_df.shape}
  Total features: {len(feature_cols)}

  What's new in V5 vs V4:
    + {len(s2_features)} Sentinel-2 features (bands + indices + derived)
    + {len(cross_sensor_features)} cross-sensor features (S2 x climate/terrain)
    + has_sentinel2 binary flag for NaN-aware modeling
    + Coordinate rounding for clean merges

  Sentinel-2 coverage:
    Train: {train_df['has_sentinel2'].sum() if 'has_sentinel2' in train_df.columns else 'N/A'}/{len(train_df)} samples with data ({100*train_df['has_sentinel2'].mean() if 'has_sentinel2' in train_df.columns else 0:.1f}%)
    Val:   {val_df['has_sentinel2'].sum() if 'has_sentinel2' in val_df.columns else 'N/A'}/{len(val_df)} samples with data ({100*val_df['has_sentinel2'].mean() if 'has_sentinel2' in val_df.columns else 0:.1f}%)

  NEXT STEP:
    python scripts/11_model_v5.py
""")

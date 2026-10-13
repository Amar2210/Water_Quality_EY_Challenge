"""
============================================================
EY Water Quality Challenge - V6 Feature Engineering
============================================================
WHAT'S NEW IN V6:
  - Full Landsat extraction (7 bands) replaces EY's 4-band data
  - NEW bands: blue, red, coastal, surface temperature (lwir11)
  - Real NDVI (from actual red band, not proxy from green)
  - Surface temperature -> direct EC correlation
  - 10 Landsat indices (vs 2 from EY)
  - Drops Sentinel-2 (97.5% NaN, useless)
  - Keeps all V4/V5 climate, soil, terrain features
  - New cross-feature interactions with blue, red, thermal

DATA SOURCES:
  1. Full Landsat (NEW) -> blue, green, red, nir, swir16, swir22,
     coastal, surface_temp + 10 indices (85.7% coverage)
  2. EY Landsat (original) -> used as BACKUP when full Landsat is NaN
  3. TerraClimate expanded (50 climate vars + lags)
  4. Elevation, Slope, Soil (static features)

OUTPUT:
  data/extracted/v6_train_features.csv
  data/extracted/v6_val_features.csv
  data/extracted/v6_feature_columns.txt
============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, '..', 'ey_challenge_notebooks')
EXTRACTED_DIR = os.path.join(BASE_DIR, '..', 'data', 'extracted')
TARGETS = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']

print("=" * 70)
print("EY WATER QUALITY - V6 FEATURE ENGINEERING")
print("Full Landsat bands + all V4 climate/terrain/soil features")
print("=" * 70)


# ============================================================
# [1] LOAD ALL DATA SOURCES
# ============================================================
print("\n" + "=" * 70)
print("[1] LOADING ALL DATA SOURCES")
print("=" * 70)

# --- Targets ---
wq_train = pd.read_csv(os.path.join(DATA_DIR, 'water_quality_training_dataset.csv'))
submission_template = pd.read_csv(os.path.join(DATA_DIR, 'submission_template.csv'))
print(f"  Training samples:    {len(wq_train)}")
print(f"  Validation samples:  {len(submission_template)}")

# --- Full Landsat (NEW in V6) ---
ls_train = pd.read_csv(os.path.join(EXTRACTED_DIR, 'landsat_full_training.csv'))
ls_val = pd.read_csv(os.path.join(EXTRACTED_DIR, 'landsat_full_validation.csv'))
ls_sr_cols = ['ls_blue', 'ls_green', 'ls_red', 'ls_nir', 'ls_swir16', 'ls_swir22']
ls_train_nan = ls_train[ls_sr_cols].isna().all(axis=1).sum()
ls_val_nan = ls_val[ls_sr_cols].isna().all(axis=1).sum()
print(f"  Full Landsat train:  {ls_train.shape}, all-NaN rows: {ls_train_nan} ({100*ls_train_nan/len(ls_train):.1f}%)")
print(f"  Full Landsat val:    {ls_val.shape}, all-NaN rows: {ls_val_nan} ({100*ls_val_nan/len(ls_val):.1f}%)")

# --- EY original Landsat (backup) ---
ey_landsat_train = pd.read_csv(os.path.join(DATA_DIR, 'landsat_features_training.csv'))
ey_landsat_val = pd.read_csv(os.path.join(DATA_DIR, 'landsat_features_validation.csv'))
print(f"  EY Landsat train:    {ey_landsat_train.shape}, NaN rows: {ey_landsat_train['nir'].isna().sum()}")

# --- TerraClimate expanded ---
terra_train = pd.read_csv(os.path.join(EXTRACTED_DIR, 'terraclimate_expanded_training.csv'))
terra_val = pd.read_csv(os.path.join(EXTRACTED_DIR, 'terraclimate_expanded_validation.csv'))
print(f"  TerraClimate train:  {terra_train.shape}")

# --- Elevation ---
elevation = pd.read_csv(os.path.join(EXTRACTED_DIR, 'elevation_data.csv'))
print(f"  Elevation:           {len(elevation)} locations")

# --- Slope ---
slope_path = os.path.join(EXTRACTED_DIR, 'slope_data.csv')
has_slope = os.path.exists(slope_path)
if has_slope:
    slope = pd.read_csv(slope_path)
    print(f"  Slope:               {len(slope)} locations")
else:
    slope = None
    print(f"  Slope:               NOT FOUND")

# --- Soil ---
soil = pd.read_csv(os.path.join(EXTRACTED_DIR, 'soil_data.csv'))
print(f"  Soil:                {len(soil)} locations")


# ============================================================
# [2] MERGE ALL SOURCES
# ============================================================
print("\n" + "=" * 70)
print("[2] MERGING ALL DATA SOURCES")
print("=" * 70)

join_keys = ['Latitude', 'Longitude', 'Sample Date']
loc_keys = ['Latitude', 'Longitude']


def round_coords(df, decimals=6):
    df = df.copy()
    df['Latitude'] = df['Latitude'].round(decimals)
    df['Longitude'] = df['Longitude'].round(decimals)
    return df


# Round all
wq_train = round_coords(wq_train)
submission_template = round_coords(submission_template)
ls_train = round_coords(ls_train)
ls_val = round_coords(ls_val)
ey_landsat_train = round_coords(ey_landsat_train)
ey_landsat_val = round_coords(ey_landsat_val)
terra_train = round_coords(terra_train)
terra_val = round_coords(terra_val)
elevation = round_coords(elevation)
soil = round_coords(soil)
if has_slope:
    slope = round_coords(slope)


def build_merged(base_df, ls_df, ey_ls_df, terra_df, name=''):
    """Merge all data sources onto the base dataframe."""
    df = base_df.copy()
    n0 = len(df)

    # Full Landsat (new extraction)
    # Drop platform/meta cols we don't want as features yet
    ls_feature_cols = [c for c in ls_df.columns if c.startswith('ls_') and c not in ['ls_platform']]
    ls_merge = ls_df[join_keys + ls_feature_cols + (['ls_platform'] if 'ls_platform' in ls_df.columns else [])].copy()
    df = df.merge(ls_merge, on=join_keys, how='left')
    print(f"    [{name}] + Full Landsat:  {df.shape[1]} cols")

    # EY original Landsat (with prefix 'ey_' to avoid collision)
    ey_rename = {c: f'ey_{c}' for c in ey_ls_df.columns if c not in join_keys}
    ey_ls_renamed = ey_ls_df.rename(columns=ey_rename)
    df = df.merge(ey_ls_renamed, on=join_keys, how='left')
    print(f"    [{name}] + EY Landsat:    {df.shape[1]} cols")

    # TerraClimate
    df = df.merge(terra_df, on=join_keys, how='left')
    print(f"    [{name}] + TerraClimate:  {df.shape[1]} cols")

    # Elevation
    df = df.merge(elevation, on=loc_keys, how='left')
    print(f"    [{name}] + Elevation:     {df.shape[1]} cols")

    # Soil
    df = df.merge(soil, on=loc_keys, how='left')
    print(f"    [{name}] + Soil:          {df.shape[1]} cols")

    # Slope
    if has_slope:
        df = df.merge(slope, on=loc_keys, how='left')
        print(f"    [{name}] + Slope:         {df.shape[1]} cols")

    # Deduplicate if needed
    if len(df) != n0:
        print(f"    [{name}] WARNING: Row inflation {n0} -> {len(df)}, deduplicating...")
        df = df.drop_duplicates(subset=join_keys, keep='first')

    # Clean _x/_y merge artifacts
    for col in list(df.columns):
        if col.endswith('_y'):
            df.drop(columns=[col], inplace=True)
        elif col.endswith('_x'):
            df.rename(columns={col: col.replace('_x', '')}, inplace=True)

    return df


print("\n  Building training set...")
train_df = build_merged(wq_train, ls_train, ey_landsat_train, terra_train, name='TRAIN')

print("\n  Building validation set...")
val_df = build_merged(submission_template, ls_val, ey_landsat_val, terra_val, name='VAL')

print(f"\n  Train after merge: {train_df.shape}")
print(f"  Val after merge:   {val_df.shape}")


# ============================================================
# [3] FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 70)
print("[3] ENGINEERING FEATURES")
print("=" * 70)


def engineer_features(df, dataset_name=''):
    """All V6 feature engineering."""
    n_before = df.shape[1]
    eps = 1e-8

    # ----------------------------------------------------------
    # 3A. TEMPORAL FEATURES
    # ----------------------------------------------------------
    df['parsed_date'] = pd.to_datetime(df['Sample Date'], dayfirst=True, errors='coerce')
    df['month'] = df['parsed_date'].dt.month
    df['year'] = df['parsed_date'].dt.year
    df['day_of_year'] = df['parsed_date'].dt.dayofyear
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['season'] = df['month'].map({
        12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
        6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3
    })
    df['is_wet_season'] = df['month'].isin([10, 11, 12, 1, 2, 3]).astype(int)
    df.drop(columns=['parsed_date'], inplace=True, errors='ignore')

    # ----------------------------------------------------------
    # 3B. FULL LANDSAT FEATURES (NEW IN V6)
    # ----------------------------------------------------------
    print(f"    [{dataset_name}] Engineering Full Landsat features...")

    # Flag: which Landsat source this sample uses
    has_full_ls = df['ls_blue'].notna()
    df['has_full_landsat'] = has_full_ls.astype(int)

    # Platform encoding (L5=0, L7=1, L8=2)
    if 'ls_platform' in df.columns:
        df['ls_platform_code'] = df['ls_platform'].map({
            'landsat-5': 0, 'landsat-7': 1, 'landsat-8': 2, 'landsat-9': 3
        }).fillna(-1)
        df.drop(columns=['ls_platform'], inplace=True, errors='ignore')

    # Clip the exploding ratios (identified in quality check)
    for ratio_col in ['ls_red_blue_ratio', 'ls_nir_red_ratio']:
        if ratio_col in df.columns:
            df[ratio_col] = df[ratio_col].clip(upper=50)

    # ---- Additional Landsat indices using blue & red (THE KEY V6 VALUE) ----

    blue = df['ls_blue'].values.astype(float)
    green = df['ls_green'].values.astype(float)
    red = df['ls_red'].values.astype(float)
    nir = df['ls_nir'].values.astype(float)
    swir16 = df['ls_swir16'].values.astype(float)
    swir22 = df['ls_swir22'].values.astype(float)

    def safe_nd(a, b):
        """Normalized difference: (a-b)/(a+b), NaN-safe."""
        denom = a + b
        result = np.where(np.abs(denom) > eps, (a - b) / denom, np.nan)
        return result

    def safe_ratio(a, b, clip_max=50):
        """a/b, NaN-safe, clipped."""
        result = np.where(np.abs(b) > eps, a / b, np.nan)
        return np.clip(result, -clip_max, clip_max)

    # Water Turbidity indices (blue+red are critical here)
    df['ls_NSMI'] = safe_nd(red + green, blue)  # Normalized Suspended Material
    df['ls_clarity_index'] = safe_ratio(blue, red + green)  # Water clarity
    df['ls_sediment_index'] = safe_ratio(red, blue)  # Sediment load

    # Chlorophyll-a proxies (uses red absorption)
    df['ls_chlor_ratio'] = safe_ratio(green - red, green + red)  # Chlorophyll proxy

    # Enhanced water indices using blue band
    df['ls_AWEI_nsh'] = 4 * (green - swir16) - (0.25 * nir + 2.75 * swir22)  # Auto Water Extraction
    df['ls_WRI'] = safe_ratio(green + red, nir + swir16)  # Water Ratio Index

    # Spectral statistics including blue & red
    all_bands = np.column_stack([blue, green, red, nir, swir16, swir22])
    df['ls_spectral_mean_6'] = np.nanmean(all_bands, axis=1)
    df['ls_spectral_std_6'] = np.nanstd(all_bands, axis=1)
    df['ls_spectral_cv_6'] = safe_ratio(
        np.nanstd(all_bands, axis=1),
        np.nanmean(all_bands, axis=1)
    )

    # Visible band statistics (blue + green + red only)
    vis_bands = np.column_stack([blue, green, red])
    df['ls_vis_mean'] = np.nanmean(vis_bands, axis=1)
    df['ls_vis_std'] = np.nanstd(vis_bands, axis=1)

    # Brightness indices
    df['ls_total_brightness'] = np.nansum(all_bands, axis=1)
    df['ls_vis_brightness'] = np.nansum(vis_bands, axis=1)

    # ---- Surface Temperature features (NEW — critical for EC) ----
    if 'ls_st_kelvin' in df.columns:
        st = df['ls_st_kelvin'].values.astype(float)
        df['ls_st_celsius'] = st - 273.15
        df['has_surface_temp'] = df['ls_st_kelvin'].notna().astype(int)

        # Temperature x spectral interactions (for EC prediction)
        df['ls_st_x_swir16'] = df['ls_st_celsius'] * df['ls_swir16']
        df['ls_st_x_nir'] = df['ls_st_celsius'] * df['ls_nir']
        df['ls_st_x_NDMI'] = df['ls_st_celsius'] * df['ls_NDMI']

    # ---- Coastal aerosol features (Landsat 8 only) ----
    if 'ls_coastal' in df.columns:
        df['has_coastal'] = df['ls_coastal'].notna().astype(int)
        df['ls_coastal_blue_ratio'] = safe_ratio(
            df['ls_coastal'].values,
            df['ls_blue'].values
        )

    # ----------------------------------------------------------
    # 3C. HYBRID: FILL MISSING FULL LANDSAT WITH EY LANDSAT
    # ----------------------------------------------------------
    # When our extraction returned NaN but EY has data, create backup features
    # EY columns: ey_nir, ey_green, ey_swir16, ey_swir22, ey_NDMI, ey_MNDWI
    ey_cols = [c for c in df.columns if c.startswith('ey_')]
    if ey_cols:
        print(f"    [{dataset_name}] Creating hybrid Landsat features (full + EY backup)...")

        # Unified NIR, Green, SWIR16, SWIR22 — prefer full, fallback to EY (scaled to reflectance)
        # EY values are raw DN (0-65535 range), our values are reflectance (0-1)
        # EY Landsat C2 L2: reflectance = DN * 0.0000275 - 0.2
        EY_SCALE = 0.0000275
        EY_OFFSET = -0.2

        band_map = {
            'nir': ('ls_nir', 'ey_nir'),
            'green': ('ls_green', 'ey_green'),
            'swir16': ('ls_swir16', 'ey_swir16'),
            'swir22': ('ls_swir22', 'ey_swir22'),
        }

        for band_name, (full_col, ey_col) in band_map.items():
            if full_col in df.columns and ey_col in df.columns:
                # Convert EY DN to reflectance
                ey_refl = df[ey_col] * EY_SCALE + EY_OFFSET
                ey_refl = ey_refl.clip(0, 1)
                # Unified: prefer full, fallback to EY
                df[f'unified_{band_name}'] = df[full_col].fillna(ey_refl)

        # Unified NDMI and MNDWI
        if 'ey_NDMI' in df.columns:
            df['unified_NDMI'] = df['ls_NDMI'].fillna(df['ey_NDMI'])
        if 'ey_MNDWI' in df.columns:
            df['unified_MNDWI'] = df['ls_MNDWI'].fillna(df['ey_MNDWI'])

        # Unified NDVI (full Landsat has real NDVI, EY doesn't)
        # For EY-only samples, compute proxy NDVI from ey_nir and ey_green
        if 'ey_nir' in df.columns and 'ey_green' in df.columns:
            ey_ndvi_proxy = (df['ey_nir'] - df['ey_green']) / (df['ey_nir'] + df['ey_green'] + eps)
            df['unified_NDVI'] = df['ls_NDVI'].fillna(ey_ndvi_proxy)

    # ----------------------------------------------------------
    # 3D. LANDSAT BAND RATIOS (from EY's original bands — V4 legacy)
    # ----------------------------------------------------------
    # Keep V4 ratios using EY's original bands for backward compatibility
    for col_pair in [('ey_nir', 'ey_green', 'ratio_nir_green'),
                     ('ey_swir16', 'ey_nir', 'ratio_swir16_nir'),
                     ('ey_swir22', 'ey_swir16', 'ratio_swir22_swir16'),
                     ('ey_green', 'ey_swir16', 'ratio_green_swir16'),
                     ('ey_nir', 'ey_swir22', 'ratio_nir_swir22')]:
        a_col, b_col, out_col = col_pair
        if a_col in df.columns and b_col in df.columns:
            df[out_col] = safe_ratio(df[a_col].values, df[b_col].values)

    # V4 legacy indices from EY bands
    if 'ey_nir' in df.columns and 'ey_green' in df.columns:
        nir_ey = df['ey_nir'].values.astype(float)
        green_ey = df['ey_green'].values.astype(float)
        swir16_ey = df['ey_swir16'].values.astype(float)
        swir22_ey = df['ey_swir22'].values.astype(float)

        df['ndvi_proxy'] = safe_nd(nir_ey, green_ey)
        df['ndsi_swir'] = safe_nd(swir16_ey, swir22_ey)
        bsi_num = (swir16_ey + green_ey) - (nir_ey + swir22_ey)
        bsi_den = (swir16_ey + green_ey) + (nir_ey + swir22_ey)
        df['bsi_proxy'] = np.where(np.abs(bsi_den) > eps, bsi_num / bsi_den, np.nan)
        df['ndbi_proxy'] = safe_nd(swir16_ey, nir_ey)

        band_stack = np.column_stack([nir_ey, green_ey, swir16_ey, swir22_ey])
        df['water_brightness'] = np.nansum(band_stack, axis=1)
        df['spectral_std'] = np.nanstd(band_stack, axis=1)
        df['spectral_mean'] = np.nanmean(band_stack, axis=1)
        df['spectral_cv'] = safe_ratio(
            np.nanstd(band_stack, axis=1),
            np.nanmean(band_stack, axis=1)
        )

    # ----------------------------------------------------------
    # 3E. CLIMATE-DERIVED FEATURES (from V4/V5)
    # ----------------------------------------------------------
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

    if 'pet' in df.columns and 'ppt' in df.columns:
        df['evap_rain_ratio'] = np.where(df['ppt'] > 0, df['pet'] / df['ppt'], np.nan)
    if 'pet' in df.columns and 'aet' in df.columns:
        df['evap_deficit_ratio'] = np.where(df['pet'] > 0, df['aet'] / df['pet'], np.nan)
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

    # ----------------------------------------------------------
    # 3F. TERRAIN FEATURES (from V4)
    # ----------------------------------------------------------
    if 'elevation_m' in df.columns:
        df['elev_bin'] = pd.cut(
            df['elevation_m'], bins=[-np.inf, 200, 600, 1400, np.inf],
            labels=[0, 1, 2, 3]
        ).astype(float)
        df['log_elevation'] = np.log1p(df['elevation_m'].clip(lower=0))

    # ----------------------------------------------------------
    # 3G. INTERACTION FEATURES (V4 + NEW V6 interactions)
    # ----------------------------------------------------------
    # V4 interactions
    if 'slope_deg' in df.columns and 'ppt' in df.columns:
        df['slope_x_ppt'] = df['slope_deg'] * df['ppt']
    if 'slope_deg' in df.columns and 'soil_clay' in df.columns:
        df['slope_x_clay'] = df['slope_deg'] * df['soil_clay']
    if 'elevation_m' in df.columns and 'tmax' in df.columns:
        df['elev_x_tmax'] = df['elevation_m'] * df['tmax']
    if 'elevation_m' in df.columns and 'ppt' in df.columns:
        df['elev_x_ppt'] = df['elevation_m'] * df['ppt']
    if 'soil' in df.columns and 'soil_clay' in df.columns:
        df['moisture_x_clay'] = df['soil'] * df['soil_clay']
    if 'soil_phh2o' in df.columns and 'ppt' in df.columns:
        df['soilph_x_ppt'] = df['soil_phh2o'] * df['ppt']

    # Soil derived
    if 'soil_phh2o' in df.columns and 'soil_cec' in df.columns:
        df['soil_buffer_capacity'] = df['soil_phh2o'] * df['soil_cec']
    if 'soil_sand' in df.columns and 'soil_clay' in df.columns:
        df['soil_sand_clay_ratio'] = np.where(df['soil_clay'] > 0, df['soil_sand'] / df['soil_clay'], np.nan)
    if 'soil_soc' in df.columns and 'soil_nitrogen' in df.columns:
        df['soil_cn_ratio'] = np.where(df['soil_nitrogen'] > 0, df['soil_soc'] / df['soil_nitrogen'], np.nan)

    # Landsat x Climate (using EY's original bands for backward compat)
    if 'ey_MNDWI' in df.columns and 'ppt' in df.columns:
        df['mndwi_x_ppt'] = df['ey_MNDWI'] * df['ppt']
    if 'ey_NDMI' in df.columns and 'ppt' in df.columns:
        df['ndmi_x_ppt'] = df['ey_NDMI'] * df['ppt']
    if 'ey_swir16' in df.columns and 'tmax' in df.columns:
        df['swir16_x_tmax'] = df['ey_swir16'] * df['tmax']

    # ----- NEW V6 INTERACTIONS -----

    # Surface temperature x climate (thermal-conductivity chain)
    if 'ls_st_celsius' in df.columns:
        if 'ppt' in df.columns:
            df['st_x_ppt'] = df['ls_st_celsius'] * df['ppt']
        if 'vpd' in df.columns:
            df['st_x_vpd'] = df['ls_st_celsius'] * df['vpd']
        if 'elevation_m' in df.columns:
            df['st_x_elev'] = df['ls_st_celsius'] * df['elevation_m']
        if 'soil_clay' in df.columns:
            df['st_x_clay'] = df['ls_st_celsius'] * df['soil_clay']

    # Blue band x slope (erosion -> turbidity -> blue absorption)
    if 'ls_blue' in df.columns and 'slope_deg' in df.columns:
        df['blue_x_slope'] = df['ls_blue'] * df['slope_deg']

    # NDVI x precipitation (vegetation response to rainfall)
    if 'ls_NDVI' in df.columns and 'ppt' in df.columns:
        df['ndvi_x_ppt'] = df['ls_NDVI'] * df['ppt']

    # NDVI x temperature (growing conditions)
    if 'ls_NDVI' in df.columns and 'tmax' in df.columns:
        df['ndvi_x_tmax'] = df['ls_NDVI'] * df['tmax']

    # Turbidity x precipitation (rainfall-driven sediment)
    if 'ls_NDTI' in df.columns and 'ppt' in df.columns:
        df['ndti_x_ppt'] = df['ls_NDTI'] * df['ppt']

    # Water clarity x soil type
    if 'ls_clarity_index' in df.columns and 'soil_sand' in df.columns:
        df['clarity_x_sand'] = df['ls_clarity_index'] * df['soil_sand']

    n_after = df.shape[1]
    print(f"    [{dataset_name}] Features: {n_before} -> {n_after} (+{n_after - n_before} new)")

    return df


train_df = engineer_features(train_df, 'TRAIN')
val_df = engineer_features(val_df, 'VAL')


# ============================================================
# [4] SELECT FEATURE COLUMNS
# ============================================================
print("\n" + "=" * 70)
print("[4] SELECTING FEATURE COLUMNS")
print("=" * 70)

# Columns to exclude from features
exclude_cols = set(TARGETS + ['Latitude', 'Longitude', 'Sample Date'])
# Also exclude any raw EY columns (we keep derived ones)
# Actually keep ey_ columns since they're used in ratios and as backup
# But drop ls_platform (already encoded as ls_platform_code)

feature_cols = [c for c in train_df.columns if c not in exclude_cols]

# Remove any remaining non-numeric columns
non_numeric = []
for col in feature_cols:
    if train_df[col].dtype == 'object':
        non_numeric.append(col)
feature_cols = [c for c in feature_cols if c not in non_numeric]
if non_numeric:
    print(f"  Dropped non-numeric: {non_numeric}")

print(f"  Total feature columns: {len(feature_cols)}")

# NaN summary
nan_rates = train_df[feature_cols].isna().mean().sort_values(ascending=False)
high_nan = nan_rates[nan_rates > 0.9]
if len(high_nan) > 0:
    print(f"\n  Features with >90% NaN (will be dropped):")
    for col, rate in high_nan.items():
        print(f"    {col}: {rate*100:.1f}% NaN")
    feature_cols = [c for c in feature_cols if c not in high_nan.index]
    print(f"  After dropping high-NaN: {len(feature_cols)} features")

# Print feature groups
groups = {
    'Full Landsat (new)': [c for c in feature_cols if c.startswith('ls_')],
    'EY Landsat (original)': [c for c in feature_cols if c.startswith('ey_')],
    'Unified Landsat': [c for c in feature_cols if c.startswith('unified_')],
    'TerraClimate': [c for c in feature_cols if any(c.startswith(p) for p in ['ppt', 'tmax', 'tmin', 'soil_lag', 'aet', 'def', 'q', 'vpd', 'pdsi', 'pet', 'srad'])
                      and not c.startswith('soil_')],
    'Soil': [c for c in feature_cols if c.startswith('soil_')],
    'Terrain': [c for c in feature_cols if c in ['elevation_m', 'slope_deg', 'aspect_deg', 'terrain_roughness',
                'elev_range_500m', 'elev_bin', 'log_elevation']],
    'Temporal': [c for c in feature_cols if c in ['month', 'year', 'day_of_year', 'month_sin', 'month_cos',
                'doy_sin', 'doy_cos', 'season', 'is_wet_season']],
    'Interactions': [c for c in feature_cols if '_x_' in c],
}

print(f"\n  Feature breakdown:")
accounted = set()
for group_name, cols in groups.items():
    if cols:
        print(f"    {group_name:25s}: {len(cols)} features")
        accounted.update(cols)
other = [c for c in feature_cols if c not in accounted]
if other:
    print(f"    {'Other/Derived':25s}: {len(other)} features")


# ============================================================
# [5] SAVE
# ============================================================
print("\n" + "=" * 70)
print("[5] SAVING V6 FEATURES")
print("=" * 70)

# Save features + targets for training
train_out = train_df[['Latitude', 'Longitude', 'Sample Date'] + TARGETS + feature_cols]
val_out = val_df[['Latitude', 'Longitude', 'Sample Date'] + feature_cols]

train_path = os.path.join(EXTRACTED_DIR, 'v6_train_features.csv')
val_path = os.path.join(EXTRACTED_DIR, 'v6_val_features.csv')
feat_path = os.path.join(EXTRACTED_DIR, 'v6_feature_columns.txt')

train_out.to_csv(train_path, index=False)
val_out.to_csv(val_path, index=False)
with open(feat_path, 'w') as f:
    for col in feature_cols:
        f.write(col + '\n')

print(f"  Train: {train_out.shape} -> {train_path}")
print(f"  Val:   {val_out.shape} -> {val_path}")
print(f"  Feature list: {len(feature_cols)} columns -> {feat_path}")

print(f"""
{'='*70}
V6 FEATURE ENGINEERING COMPLETE
{'='*70}

Key improvements over V5:
  1. Real NDVI from actual red band (not green proxy)
  2. Blue band -> turbidity, clarity, sediment indices
  3. Surface temperature -> direct EC correlation
  4. 7 Landsat bands instead of 4 (85.7% coverage)
  5. Hybrid Landsat: full extraction + EY backup
  6. Dropped Sentinel-2 (97.5% NaN, added no value)
  7. New thermal x climate interactions for EC
  8. New blue x terrain interactions for turbidity

Next: Run 12_model_v6.py
{'='*70}
""")

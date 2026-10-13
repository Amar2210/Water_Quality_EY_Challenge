"""
============================================================
EY Water Quality Challenge - V8 Feature Engineering
============================================================
WHAT'S NEW IN V8:
  Adds two CRITICAL new data sources on top of V6:
  
  1. LAND COVER (ESA CCI 2013, 300m) — % cropland, urban, forest,
     bare, water, etc. at 1km, 5km, 10km buffer zones
     WHY: Agricultural runoff drives DRP. Urban areas drive EC.
     Natural forest = low EC. Mining/bare areas = high EC.
     
  2. GEOLOGY (Macrostrat API) — rock type, lithology, age
     WHY: Bedrock is THE primary natural driver of alkalinity.
     Limestone → high alkalinity. Granite → low alkalinity.
     Without this, model can't distinguish limestone rivers
     from granite rivers.
  
  ALSO: Land cover x climate interactions, geology x spectral
  interactions — these give the model the spatial context it
  needs to generalize to unseen regions.

DATA SOURCES (all):
  1. Full Landsat (blue, green, red, nir, swir16, swir22, thermal)
  2. EY Landsat (backup for NaN rows)
  3. TerraClimate expanded (50+ climate vars + lags)
  4. Elevation, Slope, Soil (static)
  5. Land Cover (NEW) — ESA CCI 2013, 3 buffer sizes
  6. Geology (NEW) — Macrostrat lithology + soil proxies

OUTPUT:
  data/extracted/v8_train_features.csv
  data/extracted/v8_val_features.csv
  data/extracted/v8_feature_columns.txt
============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'ey_challenge_notebooks')
EXTRACTED_DIR = os.path.join(BASE_DIR, '..', 'data', 'extracted')
TARGETS = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']

print("=" * 70)
print("EY WATER QUALITY - V8 FEATURE ENGINEERING")
print("V6 features + LAND COVER + GEOLOGY")
print("=" * 70)


# ============================================================
# [1] LOAD ALL DATA SOURCES
# ============================================================
print("\n" + "=" * 70)
print("[1] LOADING ALL DATA SOURCES")
print("=" * 70)

# --- Base datasets ---
wq_train = pd.read_csv(os.path.join(DATA_DIR, 'water_quality_training_dataset.csv'))
submission_template = pd.read_csv(os.path.join(DATA_DIR, 'submission_template.csv'))
print(f"  Training samples:    {len(wq_train)}")
print(f"  Validation samples:  {len(submission_template)}")

# --- Full Landsat ---
ls_train = pd.read_csv(os.path.join(EXTRACTED_DIR, 'landsat_full_training.csv'))
ls_val = pd.read_csv(os.path.join(EXTRACTED_DIR, 'landsat_full_validation.csv'))
print(f"  Full Landsat train:  {ls_train.shape}")

# --- EY original Landsat ---
ey_landsat_train = pd.read_csv(os.path.join(DATA_DIR, 'landsat_features_training.csv'))
ey_landsat_val = pd.read_csv(os.path.join(DATA_DIR, 'landsat_features_validation.csv'))
print(f"  EY Landsat train:    {ey_landsat_train.shape}")

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

# --- Land Cover (NEW in V8) ---
landcover = pd.read_csv(os.path.join(EXTRACTED_DIR, 'landcover_data.csv'))
print(f"  Land Cover:          {len(landcover)} locations, {landcover.shape[1]} cols")
lc_nan = landcover.iloc[:, 2:].isna().all(axis=1).sum()
print(f"    All-NaN locations: {lc_nan} ({100*lc_nan/len(landcover):.1f}%)")

# --- Geology (NEW in V8) ---
geology = pd.read_csv(os.path.join(EXTRACTED_DIR, 'geology_data.csv'))
print(f"  Geology:             {len(geology)} locations, {geology.shape[1]} cols")
geo_nan = geology.iloc[:, 2:].isna().all(axis=1).sum()
print(f"    All-NaN locations: {geo_nan} ({100*geo_nan/len(geology):.1f}%)")


# ============================================================
# [2] MERGE ALL SOURCES
# ============================================================
print("\n" + "=" * 70)
print("[2] MERGING ALL DATA SOURCES")
print("=" * 70)

join_keys = ['Latitude', 'Longitude', 'Sample Date']
loc_keys = ['Latitude', 'Longitude']
eps = 1e-8


def round_coords(df, decimals=6):
    df = df.copy()
    df['Latitude'] = df['Latitude'].round(decimals)
    df['Longitude'] = df['Longitude'].round(decimals)
    return df


# Round each source once before merging.
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
landcover = round_coords(landcover)
geology = round_coords(geology)


def build_merged(base_df, ls_df, ey_ls_df, terra_df, name=''):
    """Merge all data sources onto the base dataframe."""
    df = base_df.copy()
    n0 = len(df)

    # Full Landsat
    ls_feature_cols = [c for c in ls_df.columns if c.startswith('ls_')]
    ls_merge = ls_df[join_keys + ls_feature_cols].copy()
    df = df.merge(ls_merge, on=join_keys, how='left')
    print(f"    [{name}] + Full Landsat:  {df.shape[1]} cols")

    # EY original Landsat
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

    # ---- NEW IN V8: Land Cover ----
    # Drop text columns before merge (dominant class names)
    lc_numeric = landcover.drop(columns=[c for c in landcover.columns if 'dominant' in c], errors='ignore')
    df = df.merge(lc_numeric, on=loc_keys, how='left')
    print(f"    [{name}] + Land Cover:    {df.shape[1]} cols")

    # ---- NEW IN V8: Geology ----
    # Drop text/source columns, keep numeric + binary flags
    geo_drop = ['geo_source', 'geo_lith_raw', 'geo_formation']
    geo_numeric = geology.drop(columns=[c for c in geo_drop if c in geology.columns], errors='ignore')
    # Encode categorical geology columns
    if 'geo_lith_category' in geo_numeric.columns:
        # One-hot encode lithology category
        lith_dummies = pd.get_dummies(geo_numeric['geo_lith_category'], prefix='lith', dummy_na=False)
        geo_numeric = pd.concat([geo_numeric, lith_dummies], axis=1)
        geo_numeric.drop(columns=['geo_lith_category'], inplace=True)
    if 'geo_rock_type' in geo_numeric.columns:
        rock_dummies = pd.get_dummies(geo_numeric['geo_rock_type'], prefix='rock', dummy_na=False)
        geo_numeric = pd.concat([geo_numeric, rock_dummies], axis=1)
        geo_numeric.drop(columns=['geo_rock_type'], inplace=True)
    if 'geo_proxy_ph_class' in geo_numeric.columns:
        ph_dummies = pd.get_dummies(geo_numeric['geo_proxy_ph_class'], prefix='ph_class', dummy_na=False)
        geo_numeric = pd.concat([geo_numeric, ph_dummies], axis=1)
        geo_numeric.drop(columns=['geo_proxy_ph_class'], inplace=True)
    
    # Handle duplicate soil columns from geology (soil_cec, soil_clay etc already in soil)
    # Rename geology's soil columns to avoid collision
    geo_soil_cols = [c for c in geo_numeric.columns 
                     if c.startswith('soil_') and c in df.columns and c not in loc_keys]
    for col in geo_soil_cols:
        geo_numeric.rename(columns={col: f'geo_{col}'}, inplace=True)
    
    df = df.merge(geo_numeric, on=loc_keys, how='left')
    print(f"    [{name}] + Geology:       {df.shape[1]} cols")

    # Deduplicate
    if len(df) != n0:
        print(f"    [{name}] WARNING: Row inflation {n0} -> {len(df)}, deduplicating...")
        df = df.drop_duplicates(subset=join_keys, keep='first')

    # Clean merge artifacts
    for col in list(df.columns):
        if col.endswith('_y'):
            df.drop(columns=[col], inplace=True)
        elif col.endswith('_x'):
            df.rename(columns={col: col.replace('_x', '')}, inplace=True)

    # Remove any duplicate columns from merges
    df = df.loc[:, ~df.columns.duplicated()]

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


def safe_nd(a, b):
    denom = a + b
    return np.where(np.abs(denom) > eps, (a - b) / denom, np.nan)


def safe_ratio(a, b, clip_max=50):
    result = np.where(np.abs(b) > eps, a / b, np.nan)
    return np.clip(result, -clip_max, clip_max)


def engineer_features(df, dataset_name=''):
    """All V8 feature engineering (V6 base + land cover + geology)."""
    n_before = df.shape[1]

    # ----------------------------------------------------------
    # 3A. TEMPORAL FEATURES (same as V6)
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
    # V7 additions
    df['is_cool_dry'] = df['month'].isin([5, 6, 7, 8, 9, 10]).astype(int)
    df['is_el_nino_drought'] = (
        (df['parsed_date'] >= pd.Timestamp('2014-10-01')) & 
        (df['parsed_date'] <= pd.Timestamp('2015-12-31'))
    ).astype(int)
    df.drop(columns=['parsed_date'], inplace=True, errors='ignore')

    # ----------------------------------------------------------
    # 3B. FULL LANDSAT FEATURES (same as V6)
    # ----------------------------------------------------------
    print(f"    [{dataset_name}] Engineering Full Landsat features...")

    has_full_ls = df['ls_blue'].notna() if 'ls_blue' in df.columns else pd.Series(False, index=df.index)
    df['has_full_landsat'] = has_full_ls.astype(int)

    if 'ls_platform' in df.columns:
        # Handle potential duplicate columns from merge
        if isinstance(df['ls_platform'], pd.DataFrame):
            # Multiple ls_platform columns — take first one
            platform_series = df['ls_platform'].iloc[:, 0]
        else:
            platform_series = df['ls_platform']
        platform_map = {'landsat-5': 0, 'landsat-7': 1, 'landsat-8': 2, 'landsat-9': 3}
        df['ls_platform_code'] = platform_series.map(platform_map).fillna(-1).astype(float)
        # Drop all ls_platform columns
        df = df.loc[:, ~df.columns.duplicated()]
        df.drop(columns=['ls_platform'], inplace=True, errors='ignore')

    for ratio_col in ['ls_red_blue_ratio', 'ls_nir_red_ratio']:
        if ratio_col in df.columns:
            df[ratio_col] = df[ratio_col].clip(upper=50)

    blue = df.get('ls_blue', pd.Series(np.nan, index=df.index)).values.astype(float)
    green = df.get('ls_green', pd.Series(np.nan, index=df.index)).values.astype(float)
    red = df.get('ls_red', pd.Series(np.nan, index=df.index)).values.astype(float)
    nir = df.get('ls_nir', pd.Series(np.nan, index=df.index)).values.astype(float)
    swir16 = df.get('ls_swir16', pd.Series(np.nan, index=df.index)).values.astype(float)
    swir22 = df.get('ls_swir22', pd.Series(np.nan, index=df.index)).values.astype(float)

    df['ls_NSMI'] = safe_nd(red + green, blue)
    df['ls_clarity_index'] = safe_ratio(blue, red + green)
    df['ls_sediment_index'] = safe_ratio(red, blue)
    df['ls_chlor_ratio'] = safe_ratio(green - red, green + red)
    df['ls_AWEI_nsh'] = 4 * (green - swir16) - (0.25 * nir + 2.75 * swir22)
    df['ls_WRI'] = safe_ratio(green + red, nir + swir16)

    all_bands = np.column_stack([blue, green, red, nir, swir16, swir22])
    df['ls_spectral_mean_6'] = np.nanmean(all_bands, axis=1)
    df['ls_spectral_std_6'] = np.nanstd(all_bands, axis=1)
    df['ls_spectral_cv_6'] = safe_ratio(np.nanstd(all_bands, axis=1), np.nanmean(all_bands, axis=1))

    vis_bands = np.column_stack([blue, green, red])
    df['ls_vis_mean'] = np.nanmean(vis_bands, axis=1)
    df['ls_vis_std'] = np.nanstd(vis_bands, axis=1)
    df['ls_total_brightness'] = np.nansum(all_bands, axis=1)
    df['ls_vis_brightness'] = np.nansum(vis_bands, axis=1)

    if 'ls_st_kelvin' in df.columns:
        st = df['ls_st_kelvin'].values.astype(float)
        df['ls_st_celsius'] = st - 273.15
        df['has_surface_temp'] = df['ls_st_kelvin'].notna().astype(int)
        df['ls_st_x_swir16'] = df['ls_st_celsius'] * df['ls_swir16']
        df['ls_st_x_nir'] = df['ls_st_celsius'] * df['ls_nir']
        df['ls_st_x_NDMI'] = df['ls_st_celsius'] * df.get('ls_NDMI', 0)

    if 'ls_coastal' in df.columns:
        df['has_coastal'] = df['ls_coastal'].notna().astype(int)
        df['ls_coastal_blue_ratio'] = safe_ratio(df['ls_coastal'].values, df['ls_blue'].values)

    # ----------------------------------------------------------
    # 3C. HYBRID LANDSAT (same as V6)
    # ----------------------------------------------------------
    ey_cols = [c for c in df.columns if c.startswith('ey_')]
    if ey_cols:
        EY_SCALE = 0.0000275
        EY_OFFSET = -0.2
        band_map = {
            'nir': ('ls_nir', 'ey_nir'), 'green': ('ls_green', 'ey_green'),
            'swir16': ('ls_swir16', 'ey_swir16'), 'swir22': ('ls_swir22', 'ey_swir22'),
        }
        for band_name, (full_col, ey_col) in band_map.items():
            if full_col in df.columns and ey_col in df.columns:
                ey_refl = (df[ey_col] * EY_SCALE + EY_OFFSET).clip(0, 1)
                df[f'unified_{band_name}'] = df[full_col].fillna(ey_refl)
        if 'ey_NDMI' in df.columns and 'ls_NDMI' in df.columns:
            df['unified_NDMI'] = df['ls_NDMI'].fillna(df['ey_NDMI'])
        if 'ey_MNDWI' in df.columns and 'ls_MNDWI' in df.columns:
            df['unified_MNDWI'] = df['ls_MNDWI'].fillna(df['ey_MNDWI'])
        if 'ey_nir' in df.columns and 'ey_green' in df.columns:
            ey_ndvi = (df['ey_nir'] - df['ey_green']) / (df['ey_nir'] + df['ey_green'] + eps)
            df['unified_NDVI'] = df.get('ls_NDVI', pd.Series(np.nan, index=df.index)).fillna(ey_ndvi)

    # ----------------------------------------------------------
    # 3D. LEGACY LANDSAT RATIOS (same as V6)
    # ----------------------------------------------------------
    for a_col, b_col, out_col in [
        ('ey_nir', 'ey_green', 'ratio_nir_green'),
        ('ey_swir16', 'ey_nir', 'ratio_swir16_nir'),
        ('ey_swir22', 'ey_swir16', 'ratio_swir22_swir16'),
        ('ey_green', 'ey_swir16', 'ratio_green_swir16'),
        ('ey_nir', 'ey_swir22', 'ratio_nir_swir22'),
    ]:
        if a_col in df.columns and b_col in df.columns:
            df[out_col] = safe_ratio(df[a_col].values, df[b_col].values)

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
        df['spectral_cv'] = safe_ratio(np.nanstd(band_stack, axis=1), np.nanmean(band_stack, axis=1))

    # ----------------------------------------------------------
    # 3E. CLIMATE-DERIVED FEATURES (same as V6)
    # ----------------------------------------------------------
    if 'ppt' in df.columns:
        df['is_dry_month'] = (df['ppt'] < 20).astype(int)
        if 'ppt_lag1' in df.columns:
            df['ppt_change_1m'] = df['ppt'] - df['ppt_lag1']
            df['ppt_surge'] = np.maximum(df['ppt_change_1m'], 0)
        if 'ppt_lag2' in df.columns:
            df['ppt_change_2m'] = df['ppt'] - df['ppt_lag2']
        if all(c in df.columns for c in ['ppt', 'ppt_lag1', 'ppt_lag2']):
            df['drought_3m'] = ((df['ppt'] < 30) & (df['ppt_lag1'] < 30) & (df['ppt_lag2'] < 30)).astype(int)

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
    # 3F. TERRAIN FEATURES (same as V6)
    # ----------------------------------------------------------
    if 'elevation_m' in df.columns:
        df['elev_bin'] = pd.cut(
            df['elevation_m'], bins=[-np.inf, 200, 600, 1400, np.inf], labels=[0, 1, 2, 3]
        ).astype(float)
        df['log_elevation'] = np.log1p(df['elevation_m'].clip(lower=0))

    # ----------------------------------------------------------
    # 3G. V6 INTERACTION FEATURES
    # ----------------------------------------------------------
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
    if 'soil_phh2o' in df.columns and 'soil_cec' in df.columns:
        df['soil_buffer_capacity'] = df['soil_phh2o'] * df['soil_cec']
    if 'soil_sand' in df.columns and 'soil_clay' in df.columns:
        df['soil_sand_clay_ratio'] = np.where(df['soil_clay'] > 0, df['soil_sand'] / df['soil_clay'], np.nan)
    if 'soil_soc' in df.columns and 'soil_nitrogen' in df.columns:
        df['soil_cn_ratio'] = np.where(df['soil_nitrogen'] > 0, df['soil_soc'] / df['soil_nitrogen'], np.nan)
    if 'ey_MNDWI' in df.columns and 'ppt' in df.columns:
        df['mndwi_x_ppt'] = df['ey_MNDWI'] * df['ppt']
    if 'ey_NDMI' in df.columns and 'ppt' in df.columns:
        df['ndmi_x_ppt'] = df['ey_NDMI'] * df['ppt']
    if 'ey_swir16' in df.columns and 'tmax' in df.columns:
        df['swir16_x_tmax'] = df['ey_swir16'] * df['tmax']

    # V6 thermal interactions
    if 'ls_st_celsius' in df.columns:
        if 'ppt' in df.columns:
            df['st_x_ppt'] = df['ls_st_celsius'] * df['ppt']
        if 'vpd' in df.columns:
            df['st_x_vpd'] = df['ls_st_celsius'] * df['vpd']
        if 'elevation_m' in df.columns:
            df['st_x_elev'] = df['ls_st_celsius'] * df['elevation_m']
        if 'soil_clay' in df.columns:
            df['st_x_clay'] = df['ls_st_celsius'] * df['soil_clay']
    if 'ls_blue' in df.columns and 'slope_deg' in df.columns:
        df['blue_x_slope'] = df['ls_blue'] * df['slope_deg']
    if 'ls_NDVI' in df.columns and 'ppt' in df.columns:
        df['ndvi_x_ppt'] = df['ls_NDVI'] * df['ppt']
    if 'ls_NDVI' in df.columns and 'tmax' in df.columns:
        df['ndvi_x_tmax'] = df['ls_NDVI'] * df['tmax']
    if 'ls_NDTI' in df.columns and 'ppt' in df.columns:
        df['ndti_x_ppt'] = df['ls_NDTI'] * df['ppt']
    if 'ls_clarity_index' in df.columns and 'soil_sand' in df.columns:
        df['clarity_x_sand'] = df['ls_clarity_index'] * df['soil_sand']

    # ==========================================================
    # 3H. LAND COVER FEATURES (NEW IN V8)
    # ==========================================================
    print(f"    [{dataset_name}] Engineering Land Cover features...")

    # The raw percentages (lc_1km_pct_*, lc_5km_pct_*, lc_10km_pct_*)
    # and derived features (human_impact, natural_veg, urban_ratio, agri_ratio)
    # are already in the dataframe from the merge.
    
    # Additional land cover derived features:

    # Multi-scale contrast: how much does land cover change from 1km to 10km?
    # High contrast = point is at an edge (e.g., city edge, farm border)
    for cls in ['cropland', 'urban', 'forest', 'bare']:
        c1 = f'lc_1km_pct_{cls}'
        c10 = f'lc_10km_pct_{cls}'
        if c1 in df.columns and c10 in df.columns:
            df[f'lc_contrast_{cls}'] = df[c1] - df[c10]
    
    # Land cover diversity (Shannon entropy at 5km)
    lc_pct_5km_cols = [f'lc_5km_pct_{c}' for c in 
                       ['cropland', 'forest', 'shrubland', 'grassland', 'sparse_veg',
                        'wetland', 'urban', 'bare', 'water'] 
                       if f'lc_5km_pct_{c}' in df.columns]
    if lc_pct_5km_cols:
        pct_vals = df[lc_pct_5km_cols].values / 100.0  # Convert to proportions
        pct_vals = np.clip(pct_vals, 1e-10, 1.0)  # Avoid log(0)
        df['lc_5km_shannon'] = -np.nansum(pct_vals * np.log(pct_vals), axis=1)
    
    # Land cover x climate interactions (THE KEY INSIGHT)
    # Agricultural land + rainfall = fertilizer runoff → high DRP
    if 'lc_5km_pct_cropland' in df.columns and 'ppt' in df.columns:
        df['crop_x_ppt'] = df['lc_5km_pct_cropland'] * df['ppt']
        df['crop_x_ppt_lag1'] = df['lc_5km_pct_cropland'] * df.get('ppt_lag1', 0)
    
    # Urban area + rainfall = urban runoff → high EC
    if 'lc_5km_pct_urban' in df.columns and 'ppt' in df.columns:
        df['urban_x_ppt'] = df['lc_5km_pct_urban'] * df['ppt']
    
    # Human impact + temperature = evaporation concentrates pollutants
    if 'lc_5km_human_impact' in df.columns and 'tmax' in df.columns:
        df['human_x_tmax'] = df['lc_5km_human_impact'] * df['tmax']
    
    # Natural vegetation + soil moisture = natural filtering
    if 'lc_5km_natural_veg' in df.columns and 'soil' in df.columns:
        df['natveg_x_soilmoist'] = df['lc_5km_natural_veg'] * df['soil']
    
    # Bare land x slope = erosion risk → sediment + minerals → EC
    if 'lc_5km_pct_bare' in df.columns and 'slope_deg' in df.columns:
        df['bare_x_slope'] = df['lc_5km_pct_bare'] * df['slope_deg']
    
    # Cropland x soil properties = nutrient leaching potential
    if 'lc_5km_pct_cropland' in df.columns and 'soil_nitrogen' in df.columns:
        df['crop_x_soil_n'] = df['lc_5km_pct_cropland'] * df['soil_nitrogen']
    if 'lc_5km_pct_cropland' in df.columns and 'soil_phh2o' in df.columns:
        df['crop_x_soil_ph'] = df['lc_5km_pct_cropland'] * df['soil_phh2o']
    
    # Land cover x spectral (urban reflectance signature)
    if 'lc_5km_pct_urban' in df.columns and 'ls_swir16' in df.columns:
        df['urban_x_swir16'] = df['lc_5km_pct_urban'] * df['ls_swir16']
    
    # Wetland presence + precipitation = natural nutrient filter
    if 'lc_5km_water_wetland' in df.columns and 'ppt' in df.columns:
        df['wetland_x_ppt'] = df['lc_5km_water_wetland'] * df['ppt']

    # ==========================================================
    # 3I. GEOLOGY FEATURES (NEW IN V8)
    # ==========================================================
    print(f"    [{dataset_name}] Engineering Geology features...")

    # The binary geology flags (geo_is_carbonate, geo_is_sedimentary, etc.)
    # and impact scores (geo_ec_impact, geo_alk_impact) are already in
    # the dataframe from the merge.
    
    # Geology age features
    if 'geo_age_top_ma' in df.columns:
        df['geo_age_range'] = df['geo_age_bottom_ma'] - df['geo_age_top_ma']
        df['geo_log_age'] = np.log1p(df['geo_age_top_ma'].clip(lower=0))
    
    # Geology x climate interactions
    # Carbonate rock + rainfall = dissolution → high alkalinity
    if 'geo_is_carbonate' in df.columns and 'ppt' in df.columns:
        df['carb_x_ppt'] = df['geo_is_carbonate'] * df['ppt']
    
    # Geology impact x temperature (weathering rate increases with temp)
    if 'geo_alk_impact' in df.columns and 'tmax' in df.columns:
        df['alk_impact_x_tmax'] = df['geo_alk_impact'] * df['tmax']
    if 'geo_ec_impact' in df.columns and 'tmax' in df.columns:
        df['ec_impact_x_tmax'] = df['geo_ec_impact'] * df['tmax']
    
    # Geology x elevation (different rock types at different elevations)
    if 'geo_ec_impact' in df.columns and 'elevation_m' in df.columns:
        df['ec_impact_x_elev'] = df['geo_ec_impact'] * df['elevation_m']
    
    # Geology x land cover (agricultural land on carbonate = very high alk)
    if 'geo_is_carbonate' in df.columns and 'lc_5km_pct_cropland' in df.columns:
        df['carb_x_crop'] = df['geo_is_carbonate'] * df['lc_5km_pct_cropland']
    
    # Geology x spectral (certain rock types have distinct spectral signatures)
    if 'geo_ec_impact' in df.columns and 'ls_swir16' in df.columns:
        df['ec_impact_x_swir'] = df['geo_ec_impact'] * df['ls_swir16']
    
    # Geology proxy: CEC per clay (mineral richness indicator)
    if 'geo_proxy_cec_per_clay' in df.columns:
        df['geo_proxy_cec_per_clay'] = df['geo_proxy_cec_per_clay'].clip(upper=10)
    if 'geo_proxy_mineral_index' in df.columns:
        df['geo_proxy_mineral_index'] = df['geo_proxy_mineral_index'].clip(upper=100)

    # ==========================================================
    # 3J. COMBINED SPATIAL CONTEXT FEATURES (NEW IN V8)
    # ==========================================================
    print(f"    [{dataset_name}] Engineering combined spatial context features...")

    # Total anthropogenic pressure index
    # Combines: urban %, cropland %, geology mining/mineral impact, slope
    urban_5km = df.get('lc_5km_pct_urban', 0)
    crop_5km = df.get('lc_5km_pct_cropland', 0) 
    bare_5km = df.get('lc_5km_pct_bare', 0)
    ec_impact = df.get('geo_ec_impact', 3)  # Default to neutral
    
    df['anthro_pressure'] = (
        urban_5km * 3 +    # Urban has strongest impact
        crop_5km * 2 +     # Agriculture has moderate impact
        bare_5km * 1.5     # Bare land (mining) also contributes
    ) * ec_impact / 3      # Scaled by geology impact

    # Natural buffering capacity
    # Forest + wetland + soil CEC → indicates natural filtering
    forest_5km = df.get('lc_5km_pct_forest', 0)
    wetland_pct = df.get('lc_5km_pct_wetland', 0)
    soil_cec = df.get('soil_cec', 0)
    df['natural_buffer'] = forest_5km + wetland_pct * 2 + soil_cec / 10

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

exclude_cols = set(TARGETS + ['Latitude', 'Longitude', 'Sample Date'])

feature_cols = [c for c in train_df.columns if c not in exclude_cols]

# Remove non-numeric columns
non_numeric = [col for col in feature_cols if train_df[col].dtype == 'object']
feature_cols = [c for c in feature_cols if c not in non_numeric]
if non_numeric:
    print(f"  Dropped non-numeric: {non_numeric}")

# Drop features with >90% NaN
nan_rates = train_df[feature_cols].isna().mean().sort_values(ascending=False)
high_nan = nan_rates[nan_rates > 0.9]
if len(high_nan) > 0:
    print(f"\n  Features with >90% NaN (dropping {len(high_nan)}):")
    for col, rate in list(high_nan.items())[:10]:
        print(f"    {col}: {rate*100:.1f}% NaN")
    if len(high_nan) > 10:
        print(f"    ... and {len(high_nan) - 10} more")
    feature_cols = [c for c in feature_cols if c not in high_nan.index]

print(f"\n  Total feature columns: {len(feature_cols)}")

# Print feature groups
groups = {
    'Full Landsat (new)': [c for c in feature_cols if c.startswith('ls_')],
    'EY Landsat (original)': [c for c in feature_cols if c.startswith('ey_')],
    'Unified Landsat': [c for c in feature_cols if c.startswith('unified_')],
    'TerraClimate': [c for c in feature_cols if any(c.startswith(p) for p in 
                      ['ppt', 'tmax', 'tmin', 'aet', 'def_', 'q', 'vpd', 'pdsi', 'pet', 'srad', 'ro'])
                      and not c.startswith('soil_') and '_x_' not in c],
    'Soil': [c for c in feature_cols if c.startswith('soil_') and 'geo_' not in c],
    'Terrain': [c for c in feature_cols if c in ['elevation_m', 'slope_deg', 'aspect_deg',
                'terrain_roughness', 'elev_range_500m', 'elev_bin', 'log_elevation']],
    'Temporal': [c for c in feature_cols if c in ['month', 'year', 'day_of_year', 'month_sin',
                'month_cos', 'doy_sin', 'doy_cos', 'season', 'is_wet_season', 
                'is_cool_dry', 'is_el_nino_drought']],
    'Land Cover (NEW)': [c for c in feature_cols if c.startswith('lc_')],
    'Geology (NEW)': [c for c in feature_cols if c.startswith('geo_') or c.startswith('lith_') or c.startswith('rock_') or c.startswith('ph_class_')],
    'Interactions': [c for c in feature_cols if '_x_' in c],
}

print(f"\n  Feature breakdown:")
accounted = set()
for group_name, cols in groups.items():
    if cols:
        marker = " ★" if "NEW" in group_name else ""
        print(f"    {group_name:30s}: {len(cols):3d} features{marker}")
        accounted.update(cols)
other = [c for c in feature_cols if c not in accounted]
if other:
    print(f"    {'Other/Derived':30s}: {len(other):3d} features")


# ============================================================
# [5] SAVE
# ============================================================
print("\n" + "=" * 70)
print("[5] SAVING V8 FEATURES")
print("=" * 70)

train_out = train_df[['Latitude', 'Longitude', 'Sample Date'] + TARGETS + feature_cols]
val_out = val_df[['Latitude', 'Longitude', 'Sample Date'] + feature_cols]

train_path = os.path.join(EXTRACTED_DIR, 'v8_train_features.csv')
val_path = os.path.join(EXTRACTED_DIR, 'v8_val_features.csv')
feat_path = os.path.join(EXTRACTED_DIR, 'v8_feature_columns.txt')

train_out.to_csv(train_path, index=False)
val_out.to_csv(val_path, index=False)
with open(feat_path, 'w') as f:
    for col in feature_cols:
        f.write(col + '\n')

print(f"  Train: {train_out.shape} -> {train_path}")
print(f"  Val:   {val_out.shape} -> {val_path}")
print(f"  Features: {len(feature_cols)} -> {feat_path}")

print(f"""
{'='*70}
V8 FEATURE ENGINEERING COMPLETE
{'='*70}

NEW in V8 over V6:
  ★ Land Cover (ESA CCI 2013):
    - % cropland, forest, urban, bare, water at 1km/5km/10km
    - Human impact index, natural vegetation index
    - Multi-scale contrast features
    - Shannon diversity of land cover
    
  ★ Geology (Macrostrat):
    - Rock type, lithology binary flags
    - EC impact score, alkalinity impact score
    - Geological age features
    
  ★ Cross-domain interactions:
    - Cropland × rainfall (fertilizer runoff → DRP)
    - Urban × rainfall (urban runoff → EC)
    - Carbonate × rainfall (dissolution → alkalinity)
    - Geology impact × temperature (weathering rate)
    - Bare land × slope (erosion risk)
    - Anthropogenic pressure index
    - Natural buffering capacity

Next: Run python scripts/20_feature_engineering_v9.py
{'='*70}
""")

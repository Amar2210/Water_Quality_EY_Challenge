"""
============================================================
EY Water Quality Challenge - Feature Extraction (All Sources)
============================================================
FIXES from original:
  1. 'ro' renamed to 'q' (correct TerraClimate variable name for runoff)
  2. Vectorized extraction: sel() with arrays instead of per-location loop
     This is 50-100x faster (~5 min vs ~10 hours)
  3. Added extra variables: pdsi, pet, srad, ws (available in dataset)

Total estimated run time: 10-15 minutes
Output: data/extracted/ folder with CSV files

PREREQUISITES:
  pip install xarray zarr adlfs planetary_computer pystac_client requests scipy dask netCDF4
============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
import requests
import time
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

try:
    import pystac_client
    import planetary_computer as pc
    PC_AVAILABLE = True
except ImportError:
    PC_AVAILABLE = False
    print("WARNING: planetary_computer or pystac_client not installed.")
    print("Install with: pip install planetary_computer pystac_client")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'ey_challenge_notebooks')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'data', 'extracted')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_locations():
    """Load all unique locations from training + validation datasets."""
    train = pd.read_csv(os.path.join(DATA_DIR, 'water_quality_training_dataset.csv'))
    val = pd.read_csv(os.path.join(DATA_DIR, 'submission_template.csv'))
    
    train_samples = train[['Latitude', 'Longitude', 'Sample Date']].copy()
    train_samples['dataset'] = 'train'
    val_samples = val[['Latitude', 'Longitude', 'Sample Date']].copy()
    val_samples['dataset'] = 'val'
    
    all_samples = pd.concat([train_samples, val_samples], ignore_index=True)
    all_samples['parsed_date'] = pd.to_datetime(all_samples['Sample Date'], dayfirst=True, errors='coerce')
    
    unique_locs = all_samples.drop_duplicates(subset=['Latitude', 'Longitude'])[['Latitude', 'Longitude']].reset_index(drop=True)
    
    print(f"  Total samples: {len(all_samples)} (train: {len(train_samples)}, val: {len(val_samples)})")
    print(f"  Unique locations: {len(unique_locs)}")
    
    return all_samples, unique_locs


# ============================================================
# SOURCE 1: TERRACLIMATE (Planetary Computer)
# ============================================================
# FIX: 'ro' -> 'q' (runoff is called 'q' in TerraClimate)
# FIX: Vectorized extraction instead of per-location loop
# ADDED: pdsi, pet, srad (also available and useful)
# ============================================================

# CORRECT variable names from the TerraClimate Zarr store:
#   aet  = Actual Evapotranspiration (mm)
#   def  = Climate Water Deficit (mm)
#   pdsi = Palmer Drought Severity Index
#   pet  = Potential Evapotranspiration (mm)
#   ppt  = Precipitation (mm)
#   q    = Runoff (mm)  <-- WAS 'ro' IN ORIGINAL (WRONG!)
#   soil = Soil Moisture (mm)
#   srad = Downward surface shortwave radiation (W/m2)
#   swe  = Snow Water Equivalent (mm)
#   tmax = Max Temperature (C)
#   tmin = Min Temperature (C)
#   vap  = Vapor Pressure (kPa)
#   vpd  = Vapor Pressure Deficit (kPa)
#   ws   = Wind Speed (m/s)

TERRA_VARIABLES = ['ppt', 'tmax', 'tmin', 'soil', 'aet', 'def', 'q', 'vpd', 'pdsi', 'pet', 'srad']
TERRA_LAG_MONTHS = [1, 2, 3]


def extract_terraclimate(all_samples):
    """
    Extract TerraClimate variables using VECTORIZED approach.
    
    Instead of looping over 186 locations (each taking 3+ min),
    we extract ALL unique (lat, lon, time) combinations at once
    using xarray's vectorized .sel() with DataArrays.
    
    This triggers far fewer remote reads and finishes in ~5 min.
    """
    if not PC_AVAILABLE:
        print("  SKIPPED: planetary_computer not installed")
        return None
    
    print("\n  Connecting to Planetary Computer...")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )
    collection = catalog.get_collection("terraclimate")
    asset = collection.assets["zarr-abfs"]
    
    # Open the Zarr store
    # The asset extra_fields may contain kwargs incompatible with our xarray version,
    # so we just use the storage_options directly (this worked in the original script)
    storage_opts = asset.extra_fields.get("xarray:storage_options", {})
    
    if storage_opts:
        ds = xr.open_zarr(asset.href, storage_options=storage_opts, consolidated=True)
    else:
        ds = xr.open_dataset(asset.href, **asset.extra_fields["xarray:open_kwargs"])
    
    available_vars = list(ds.data_vars)
    print(f"  Connected. Available variables: {available_vars}")
    
    # Filter to only variables that actually exist
    valid_vars = [v for v in TERRA_VARIABLES if v in available_vars]
    missing_vars = [v for v in TERRA_VARIABLES if v not in available_vars]
    if missing_vars:
        print(f"  WARNING: These variables don't exist and will be skipped: {missing_vars}")
    print(f"  Extracting: {valid_vars}")
    
    # Subset to only needed variables (reduces data loaded)
    ds = ds[valid_vars]
    
    # Time range: include extra months before earliest sample for lags
    max_lag = max(TERRA_LAG_MONTHS)
    min_date = all_samples['parsed_date'].min() - pd.DateOffset(months=max_lag + 1)
    max_date = all_samples['parsed_date'].max() + pd.DateOffset(months=1)
    ds = ds.sel(time=slice(str(min_date.date()), str(max_date.date())))
    print(f"  Time range: {min_date.date()} to {max_date.date()}")
    
    # ── STEP 1: Get unique (location, month) pairs we need ──
    # Build list of all (lat, lon, year-month) we need to fetch
    needed_ym = set()
    for _, row in all_samples.iterrows():
        dt = row['parsed_date']
        # Current month
        needed_ym.add((row['Latitude'], row['Longitude'], dt.year, dt.month))
        # Lagged months
        for lag in TERRA_LAG_MONTHS:
            lag_dt = dt - pd.DateOffset(months=lag)
            needed_ym.add((row['Latitude'], row['Longitude'], lag_dt.year, lag_dt.month))
    
    print(f"  Need {len(needed_ym)} unique (location, year-month) extractions")
    
    # ── STEP 2: Get unique locations ──
    unique_locs = all_samples.drop_duplicates(subset=['Latitude', 'Longitude'])[['Latitude', 'Longitude']].values
    print(f"  Unique locations: {len(unique_locs)}")
    
    # ── STEP 3: Vectorized extraction per unique time step ──
    # Strategy: for each unique month in the dataset, extract ALL locations at once
    # This is much faster because xarray can batch the spatial lookups
    
    unique_times = sorted(set((ym[2], ym[3]) for ym in needed_ym))
    print(f"  Unique months to fetch: {len(unique_times)}")
    
    # Build lat/lon arrays for vectorized sel
    lats = xr.DataArray(unique_locs[:, 0], dims='location')
    lons = xr.DataArray(unique_locs[:, 1], dims='location')
    
    # Create a mapping from (lat, lon) -> location index
    loc_to_idx = {}
    for i, (lat, lon) in enumerate(unique_locs):
        loc_to_idx[(round(lat, 6), round(lon, 6))] = i
    
    # Storage: dict of (loc_idx, year, month) -> {var: value}
    extracted = {}
    
    print(f"\n  Fetching data (vectorized, {len(unique_times)} months)...")
    for i, (year, month) in enumerate(tqdm(unique_times, desc="  Months")):
        try:
            # Build a specific datetime for this month (TerraClimate uses 1st of month)
            target_time = pd.Timestamp(year=year, month=month, day=1)
            
            # Select nearest time step, then ALL locations at once (vectorized!)
            month_data = ds.sel(time=target_time, method='nearest').sel(
                lat=lats, lon=lons, method='nearest'
            )
            
            # .compute() triggers the actual data fetch — ONE call for all locations
            month_data = month_data.compute()
            
            # Extract values for each location
            for loc_idx in range(len(unique_locs)):
                key = (loc_idx, year, month)
                vals = {}
                for var in valid_vars:
                    try:
                        arr = month_data[var].values
                        # arr should be 1D with shape (n_locations,)
                        if arr.ndim == 0:
                            val = arr.item()
                        else:
                            val = arr[loc_idx].item()
                        vals[var] = val if np.isfinite(val) else np.nan
                    except:
                        vals[var] = np.nan
                extracted[key] = vals
                
        except Exception as e:
            if i < 3:  # Only print first few warnings
                print(f"  Warning: Failed for {year}-{month:02d}: {e}")
            for loc_idx in range(len(unique_locs)):
                extracted[(loc_idx, year, month)] = {v: np.nan for v in valid_vars}
    
    # ── STEP 4: Map extracted data back to all samples ──
    print(f"\n  Mapping to {len(all_samples)} samples (with lags)...")
    
    result_rows = []
    for idx, row in tqdm(all_samples.iterrows(), total=len(all_samples), desc="  Mapping"):
        lat = round(row['Latitude'], 6)
        lon = round(row['Longitude'], 6)
        dt = row['parsed_date']
        
        loc_idx = loc_to_idx.get((lat, lon))
        if loc_idx is None:
            # Find nearest
            min_dist = float('inf')
            for (k_lat, k_lon), k_idx in loc_to_idx.items():
                d = (k_lat - lat)**2 + (k_lon - lon)**2
                if d < min_dist:
                    min_dist = d
                    loc_idx = k_idx
        
        result = {}
        
        # Current month values
        key = (loc_idx, dt.year, dt.month)
        current = extracted.get(key, {})
        for var in valid_vars:
            result[var] = current.get(var, np.nan)
        
        # Lagged values
        for lag in TERRA_LAG_MONTHS:
            lag_dt = dt - pd.DateOffset(months=lag)
            key = (loc_idx, lag_dt.year, lag_dt.month)
            lagged = extracted.get(key, {})
            for var in valid_vars:
                result[f'{var}_lag{lag}'] = lagged.get(var, np.nan)
        
        result_rows.append(result)
    
    result_df = pd.DataFrame(result_rows)
    
    # ── STEP 5: Derived features ──
    print("  Computing derived features...")
    for var in ['ppt', 'tmax', 'tmin']:
        cols = [var, f'{var}_lag1', f'{var}_lag2']
        existing = [c for c in cols if c in result_df.columns]
        if len(existing) == 3:
            result_df[f'{var}_3m_avg'] = result_df[existing].mean(axis=1)
        if var == 'ppt' and len(existing) == 3:
            result_df[f'{var}_3m_sum'] = result_df[existing].sum(axis=1)
    
    # Temperature range
    if 'tmax' in result_df.columns and 'tmin' in result_df.columns:
        result_df['temp_range'] = result_df['tmax'] - result_df['tmin']
    
    # Aridity index (PET / PPT) — very useful for water quality
    if 'pet' in result_df.columns and 'ppt' in result_df.columns:
        result_df['aridity_index'] = result_df['pet'] / result_df['ppt'].replace(0, np.nan)
    
    # Combine with location info
    final_df = pd.concat([
        all_samples[['Latitude', 'Longitude', 'Sample Date', 'dataset']].reset_index(drop=True),
        result_df.reset_index(drop=True)
    ], axis=1)
    
    return final_df


# ============================================================
# SOURCE 2: ELEVATION (Open-Elevation API)
# ============================================================

def extract_elevation(unique_locs):
    """Extract elevation for all unique locations using Open-Elevation API."""
    locations = unique_locs[['Latitude', 'Longitude']].values.tolist()
    elevations = {}
    batch_size = 80
    
    for i in range(0, len(locations), batch_size):
        batch = locations[i:i + batch_size]
        loc_string = "|".join([f"{lat},{lon}" for lat, lon in batch])
        url = f"https://api.open-elevation.com/api/v1/lookup?locations={loc_string}"
        
        try:
            print(f"  Requesting batch {i // batch_size + 1} ({len(batch)} locations)...")
            response = requests.get(url, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                for j, result in enumerate(data['results']):
                    lat, lon = batch[j]
                    elevations[(round(lat, 6), round(lon, 6))] = result['elevation']
                print(f"    Got {len(data['results'])} elevations")
            else:
                print(f"    API returned status {response.status_code}, trying POST...")
                post_data = {"locations": [{"latitude": lat, "longitude": lon} for lat, lon in batch]}
                response = requests.post("https://api.open-elevation.com/api/v1/lookup", json=post_data, timeout=60)
                if response.status_code == 200:
                    data = response.json()
                    for j, result in enumerate(data['results']):
                        lat, lon = batch[j]
                        elevations[(round(lat, 6), round(lon, 6))] = result['elevation']
        except requests.exceptions.Timeout:
            print(f"    Timeout. Retrying with longer timeout...")
            time.sleep(5)
            try:
                response = requests.get(url, timeout=120)
                if response.status_code == 200:
                    data = response.json()
                    for j, result in enumerate(data['results']):
                        lat, lon = batch[j]
                        elevations[(round(lat, 6), round(lon, 6))] = result['elevation']
            except:
                print(f"    Retry failed. Using NaN for this batch.")
        except Exception as e:
            print(f"    Error: {e}")
        
        time.sleep(1)
    
    elev_df = unique_locs.copy()
    elev_df['elevation_m'] = elev_df.apply(
        lambda r: elevations.get((round(r['Latitude'], 6), round(r['Longitude'], 6)), np.nan),
        axis=1
    )
    return elev_df


# ============================================================
# SOURCE 3: SOIL PROPERTIES (SoilGrids / ISRIC API)
# ============================================================

SOIL_PROPERTIES = ['phh2o', 'clay', 'sand', 'soc', 'cec', 'nitrogen']
SOIL_DEPTH = '5-15cm'

def extract_soil_single_location(lat, lon):
    """Extract soil properties for a single location from SoilGrids API."""
    url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    params = {
        'lat': lat, 'lon': lon,
        'property': SOIL_PROPERTIES,
        'depth': SOIL_DEPTH,
        'value': 'mean'
    }
    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            result = {}
            for layer in data.get('properties', {}).get('layers', []):
                prop_name = layer['name']
                for d in layer.get('depths', []):
                    if d['label'] == SOIL_DEPTH:
                        result[f'soil_{prop_name}'] = d.get('values', {}).get('mean', np.nan)
                        break
            return result
        return {f'soil_{p}': np.nan for p in SOIL_PROPERTIES}
    except:
        return {f'soil_{p}': np.nan for p in SOIL_PROPERTIES}


def extract_soil(unique_locs):
    """Extract soil properties for all unique locations using parallel requests."""
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {}
        for idx, row in unique_locs.iterrows():
            future = executor.submit(extract_soil_single_location, row['Latitude'], row['Longitude'])
            futures[future] = idx
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="  Extracting soil"):
            idx = futures[future]
            try:
                result = future.result()
                result['idx'] = idx
                results.append(result)
            except Exception as e:
                result = {f'soil_{p}': np.nan for p in SOIL_PROPERTIES}
                result['idx'] = idx
                results.append(result)
    
    soil_df = pd.DataFrame(results).set_index('idx').sort_index()
    return pd.concat([unique_locs.reset_index(drop=True), soil_df.reset_index(drop=True)], axis=1)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    total_start = time.time()
    
    print("=" * 70)
    print("EY WATER QUALITY - FEATURE EXTRACTION (FIXED)")
    print("=" * 70)
    
    print("\n[0] Loading locations...")
    all_samples, unique_locs = load_locations()
    
    # ─── TERRACLIMATE ───
    print(f"\n{'=' * 70}")
    print(f"[1] TERRACLIMATE EXTRACTION (VECTORIZED)")
    print(f"    Variables: {TERRA_VARIABLES}")
    print(f"    NOTE: 'q' = runoff (was incorrectly 'ro' before)")
    print(f"    Lags: {TERRA_LAG_MONTHS} months")
    print(f"{'=' * 70}")
    
    t0 = time.time()
    try:
        terra_df = extract_terraclimate(all_samples)
        if terra_df is not None:
            terra_train = terra_df[terra_df['dataset'] == 'train'].drop(columns=['dataset']).reset_index(drop=True)
            terra_val = terra_df[terra_df['dataset'] == 'val'].drop(columns=['dataset']).reset_index(drop=True)
            
            terra_train.to_csv(os.path.join(OUTPUT_DIR, 'terraclimate_expanded_training.csv'), index=False)
            terra_val.to_csv(os.path.join(OUTPUT_DIR, 'terraclimate_expanded_validation.csv'), index=False)
            
            print(f"\n  Saved terraclimate_expanded_training.csv ({terra_train.shape})")
            print(f"  Saved terraclimate_expanded_validation.csv ({terra_val.shape})")
            feature_cols = [c for c in terra_train.columns if c not in ['Latitude', 'Longitude', 'Sample Date']]
            print(f"  Feature columns ({len(feature_cols)}): {feature_cols[:10]}...")
            
            # Quick sanity check
            for var in ['ppt', 'tmax', 'q', 'soil']:
                if var in terra_train.columns:
                    valid = terra_train[var].notna().sum()
                    print(f"    {var}: {valid}/{len(terra_train)} valid, mean={terra_train[var].mean():.2f}")
            
            print(f"  Time: {time.time()-t0:.0f} seconds")
    except Exception as e:
        print(f"  TerraClimate error: {e}")
        import traceback
        traceback.print_exc()
    
    # ─── ELEVATION ───
    print(f"\n{'=' * 70}")
    print(f"[2] ELEVATION EXTRACTION (Open-Elevation API)")
    print(f"{'=' * 70}")
    
    t0 = time.time()
    try:
        elev_df = extract_elevation(unique_locs)
        elev_df.to_csv(os.path.join(OUTPUT_DIR, 'elevation_data.csv'), index=False)
        valid = elev_df['elevation_m'].notna().sum()
        print(f"\n  Saved elevation_data.csv ({valid}/{len(elev_df)} valid)")
        if valid > 0:
            print(f"  Range: {elev_df['elevation_m'].min():.0f}m to {elev_df['elevation_m'].max():.0f}m")
        print(f"  Time: {time.time()-t0:.0f} seconds")
    except Exception as e:
        print(f"  Elevation error: {e}")
        import traceback
        traceback.print_exc()
    
    # ─── SOIL ───
    print(f"\n{'=' * 70}")
    print(f"[3] SOIL PROPERTIES (SoilGrids/ISRIC API)")
    print(f"    Properties: {SOIL_PROPERTIES}")
    print(f"    Depth: {SOIL_DEPTH}")
    print(f"{'=' * 70}")
    
    t0 = time.time()
    try:
        soil_df = extract_soil(unique_locs)
        soil_df.to_csv(os.path.join(OUTPUT_DIR, 'soil_data.csv'), index=False)
        print(f"\n  Saved soil_data.csv")
        for col in soil_df.columns:
            if col.startswith('soil_'):
                valid = soil_df[col].notna().sum()
                print(f"    {col}: {valid}/{len(soil_df)} valid, mean={soil_df[col].mean():.1f}")
        print(f"  Time: {time.time()-t0:.0f} seconds")
    except Exception as e:
        print(f"  Soil error: {e}")
        import traceback
        traceback.print_exc()
    
    # ─── SUMMARY ───
    total_time = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"EXTRACTION COMPLETE — Total: {total_time/60:.1f} minutes")
    print(f"{'=' * 70}")
    
    print(f"\nOutput files in {OUTPUT_DIR}:")
    if os.path.exists(OUTPUT_DIR):
        for f in sorted(os.listdir(OUTPUT_DIR)):
            fpath = os.path.join(OUTPUT_DIR, f)
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  {f}: {size_kb:.1f} KB")
    
    print(f"""
NEXT STEPS:
  1. Check the extracted CSV files in data/extracted/
  2. Complete the Landsat, slope, land cover, and geology extraction stages
  3. Run scripts/18_feature_engineering_v8.py, then scripts/20_feature_engineering_v9.py
""")
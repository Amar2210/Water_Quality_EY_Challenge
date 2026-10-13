"""
============================================================
EY Water Quality Challenge - ESA WorldCover Land Use Extraction
============================================================
Extracts land cover percentages within buffer zones (1km, 5km, 10km)
around each unique river sampling location.

SOURCE: ESA WorldCover 2021 (10m resolution) via Planetary Computer
  Collection: 'esa-worldcover'

LAND COVER CLASSES:
  10 = Tree cover          → Natural/forest (low EC)
  20 = Shrubland           → Semi-natural
  30 = Grassland           → Grazing/natural
  40 = Cropland            → Agriculture (fertilizer → DRP, moderate EC)
  50 = Built-up            → Urban (high EC from salts, chemicals, runoff)
  60 = Bare/sparse veg     → Mining, erosion, degraded land (high EC)
  70 = Snow and ice        → N/A for South Africa
  80 = Permanent water     → Water bodies
  90 = Herbaceous wetland  → Wetlands (nutrient filtering)
  95 = Mangroves           → Coastal
 100 = Moss and lichen     → Rare

WHY THIS MATTERS FOR EC:
  - A river surrounded by 80% cropland has very different EC than
    one surrounded by 80% natural forest
  - Built-up/urban areas introduce dissolved salts, chemicals
  - Mining/bare areas expose minerals → high EC
  - This is SPATIAL information that transfers to unseen locations

OUTPUT: data/extracted/worldcover_data.csv
  One row per unique location with columns for each class percentage
  at each buffer radius.

ESTIMATED TIME: 10-30 minutes (186 unique locations, parallel)
============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import numpy as np
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

try:
    import pystac_client
    import planetary_computer as pc
    import rioxarray
    import xarray as xr
    from shapely.geometry import Point, box
    PC_AVAILABLE = True
except ImportError as e:
    PC_AVAILABLE = False
    print(f"ERROR: Missing dependency: {e}")
    print("Install with: pip install planetary_computer pystac_client rioxarray shapely")
    sys.exit(1)

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, '..', 'ey_challenge_notebooks')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'data', 'extracted')
os.makedirs(OUTPUT_DIR, exist_ok=True)

BUFFER_RADII_M = [1000, 5000, 10000]  # 1km, 5km, 10km

# WorldCover class names
LC_CLASSES = {
    10: 'tree_cover',
    20: 'shrubland',
    30: 'grassland',
    40: 'cropland',
    50: 'built_up',
    60: 'bare_sparse',
    70: 'snow_ice',
    80: 'water',
    90: 'herbaceous_wetland',
    95: 'mangroves',
    100: 'moss_lichen',
}

MAX_WORKERS = 8
MAX_RETRIES = 3

OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'worldcover_data.csv')
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, 'worldcover_checkpoint.csv')


def load_unique_locations():
    """Load all unique locations from training + validation."""
    train = pd.read_csv(os.path.join(DATA_DIR, 'water_quality_training_dataset.csv'))
    val = pd.read_csv(os.path.join(DATA_DIR, 'submission_template.csv'))
    
    all_locs = pd.concat([
        train[['Latitude', 'Longitude']],
        val[['Latitude', 'Longitude']]
    ]).drop_duplicates().reset_index(drop=True)
    
    print(f"  Unique locations: {len(all_locs)}")
    return all_locs


def meters_to_degrees(meters, latitude):
    """Approximate conversion from meters to degrees at given latitude."""
    lat_deg = meters / 111320.0
    lon_deg = meters / (111320.0 * np.cos(np.radians(latitude)))
    return lat_deg, lon_deg


def extract_worldcover_for_location(lat, lon, catalog):
    """
    Extract land cover percentages at multiple buffer radii for one location.
    
    Returns dict with columns like:
      lc_1km_pct_tree_cover, lc_1km_pct_cropland, lc_5km_pct_built_up, etc.
    """
    result = {'Latitude': lat, 'Longitude': lon}
    
    for radius_m in BUFFER_RADII_M:
        radius_label = f"{radius_m // 1000}km"
        
        # Create bounding box in degrees
        lat_buf, lon_buf = meters_to_degrees(radius_m, lat)
        bbox = [lon - lon_buf, lat - lat_buf, lon + lon_buf, lat + lat_buf]
        
        try:
            # Search for WorldCover tile covering this location
            search = catalog.search(
                collections=['esa-worldcover'],
                bbox=bbox,
                query={"esa_worldcover:product_version": {"eq": "V200"}},  # 2021 version
            )
            items = list(search.items())
            
            if not items:
                # Try without version filter
                search = catalog.search(
                    collections=['esa-worldcover'],
                    bbox=bbox,
                )
                items = list(search.items())
            
            if not items:
                # No data — fill with NaN
                for cls_val, cls_name in LC_CLASSES.items():
                    result[f'lc_{radius_label}_pct_{cls_name}'] = np.nan
                continue
            
            # Sign the asset URL for Planetary Computer
            item = items[0]
            signed_item = pc.sign(item)
            
            # Read the land cover raster
            asset = signed_item.assets.get('map', signed_item.assets.get('data', None))
            if asset is None:
                # Try first available asset
                asset_key = [k for k in signed_item.assets.keys() if k not in ['tilejson', 'rendered_preview']]
                if asset_key:
                    asset = signed_item.assets[asset_key[0]]
                else:
                    for cls_val, cls_name in LC_CLASSES.items():
                        result[f'lc_{radius_label}_pct_{cls_name}'] = np.nan
                    continue
            
            href = asset.href
            
            # Read clipped to our bounding box
            da = rioxarray.open_rasterio(href)
            
            # Clip to bounding box
            da_clipped = da.rio.clip_box(
                minx=bbox[0], miny=bbox[1], 
                maxx=bbox[2], maxy=bbox[3]
            )
            
            # Get the land cover values
            values = da_clipped.values.flatten()
            values = values[values > 0]  # remove nodata (0)
            
            if len(values) == 0:
                for cls_val, cls_name in LC_CLASSES.items():
                    result[f'lc_{radius_label}_pct_{cls_name}'] = np.nan
                continue
            
            total_pixels = len(values)
            
            # Calculate percentage for each class
            for cls_val, cls_name in LC_CLASSES.items():
                count = np.sum(values == cls_val)
                pct = (count / total_pixels) * 100.0
                result[f'lc_{radius_label}_pct_{cls_name}'] = round(pct, 2)
            
        except Exception as e:
            # On error, fill NaN for this radius
            for cls_val, cls_name in LC_CLASSES.items():
                result[f'lc_{radius_label}_pct_{cls_name}'] = np.nan
            result[f'lc_{radius_label}_error'] = str(e)[:100]
    
    # Compute derived features (across all radii)
    for radius_m in BUFFER_RADII_M:
        r = f"{radius_m // 1000}km"
        # Human impact index = built_up + cropland + bare (proxy for anthropogenic influence)
        built = result.get(f'lc_{r}_pct_built_up', 0) or 0
        crop = result.get(f'lc_{r}_pct_cropland', 0) or 0
        bare = result.get(f'lc_{r}_pct_bare_sparse', 0) or 0
        natural = result.get(f'lc_{r}_pct_tree_cover', 0) or 0
        grass = result.get(f'lc_{r}_pct_grassland', 0) or 0
        shrub = result.get(f'lc_{r}_pct_shrubland', 0) or 0
        
        result[f'lc_{r}_human_impact'] = round(built + crop + bare, 2)
        result[f'lc_{r}_natural_veg'] = round(natural + grass + shrub, 2)
        # Urban density ratio (built-up relative to all non-water land)
        total_land = 100.0 - (result.get(f'lc_{r}_pct_water', 0) or 0)
        result[f'lc_{r}_urban_ratio'] = round(built / max(total_land, 1), 4)
    
    return result


def extract_with_retry(lat, lon, catalog, max_retries=MAX_RETRIES):
    """Extract with retry logic."""
    for attempt in range(max_retries):
        try:
            return extract_worldcover_for_location(lat, lon, catalog)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                result = {'Latitude': lat, 'Longitude': lon}
                for radius_m in BUFFER_RADII_M:
                    r = f"{radius_m // 1000}km"
                    for cls_name in LC_CLASSES.values():
                        result[f'lc_{r}_pct_{cls_name}'] = np.nan
                result['error'] = str(e)[:200]
                return result


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print(f"""
{'='*70}
ESA WORLDCOVER LAND USE EXTRACTION
{'='*70}
Source: ESA WorldCover 2021 (10m resolution)
Buffers: {BUFFER_RADII_M} meters
Workers: {MAX_WORKERS}
{'='*70}
""")
    
    # Load locations
    print("[1] Loading locations...")
    locations = load_unique_locations()
    
    # Check for existing checkpoint
    done_locs = set()
    existing_results = []
    if os.path.exists(CHECKPOINT_FILE):
        checkpoint = pd.read_csv(CHECKPOINT_FILE)
        for _, row in checkpoint.iterrows():
            done_locs.add((round(row['Latitude'], 6), round(row['Longitude'], 6)))
            existing_results.append(row.to_dict())
        print(f"  Resuming: {len(done_locs)} locations already done")
    
    # Filter to remaining locations
    remaining = []
    for _, row in locations.iterrows():
        key = (round(row['Latitude'], 6), round(row['Longitude'], 6))
        if key not in done_locs:
            remaining.append((row['Latitude'], row['Longitude']))
    
    print(f"  Remaining: {len(remaining)} locations to extract")
    
    if len(remaining) == 0:
        print("  All locations already extracted!")
        if existing_results:
            result_df = pd.DataFrame(existing_results)
            result_df.to_csv(OUTPUT_FILE, index=False)
            print(f"  Saved: {OUTPUT_FILE}")
        sys.exit(0)
    
    # Connect to Planetary Computer
    print("\n[2] Connecting to Planetary Computer...")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )
    print("  Connected ✓")
    
    # Test with first location
    print("\n[3] Testing extraction with first location...")
    test_lat, test_lon = remaining[0]
    test_result = extract_with_retry(test_lat, test_lon, catalog)
    
    # Show sample results
    print(f"  Test location: ({test_lat}, {test_lon})")
    for radius_m in BUFFER_RADII_M:
        r = f"{radius_m // 1000}km"
        print(f"  {r} buffer:")
        for cls_name in ['tree_cover', 'cropland', 'built_up', 'bare_sparse', 'grassland', 'water']:
            val = test_result.get(f'lc_{r}_pct_{cls_name}', 'N/A')
            if isinstance(val, float):
                print(f"    {cls_name:20s}: {val:6.2f}%")
            else:
                print(f"    {cls_name:20s}: {val}")
        hi = test_result.get(f'lc_{r}_human_impact', 'N/A')
        print(f"    {'human_impact':20s}: {hi}")
    
    if 'error' in test_result:
        print(f"\n  ⚠ Test had error: {test_result['error']}")
        print("  Will continue anyway — some locations may fail.")
    
    # Full extraction with parallel processing
    print(f"\n[4] Extracting all {len(remaining)} locations (parallel, {MAX_WORKERS} workers)...")
    
    all_results = list(existing_results)  # Start with checkpoint data
    
    batch_size = 20  # Checkpoint every 20 locations
    
    for batch_start in range(0, len(remaining), batch_size):
        batch = remaining[batch_start:batch_start + batch_size]
        batch_results = []
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(extract_with_retry, lat, lon, catalog): (lat, lon)
                for lat, lon in batch
            }
            
            pbar = tqdm(as_completed(futures), total=len(futures),
                       desc=f"  Batch {batch_start//batch_size + 1}/"
                            f"{(len(remaining) + batch_size - 1)//batch_size}")
            
            for future in pbar:
                try:
                    result = future.result(timeout=120)
                    batch_results.append(result)
                except Exception as e:
                    lat, lon = futures[future]
                    err_result = {'Latitude': lat, 'Longitude': lon, 'error': str(e)[:200]}
                    batch_results.append(err_result)
                    pbar.set_postfix({'errors': 1})
            
            pbar.close()
        
        all_results.extend(batch_results)
        
        # Checkpoint
        checkpoint_df = pd.DataFrame(all_results)
        checkpoint_df.to_csv(CHECKPOINT_FILE, index=False)
        
        done_count = len(all_results)
        total = len(locations)
        print(f"  Progress: {done_count}/{total} locations ({100*done_count/total:.1f}%)")
    
    # Save final output
    print(f"\n[5] Saving results...")
    result_df = pd.DataFrame(all_results)
    
    # Remove error columns from final output
    error_cols = [c for c in result_df.columns if 'error' in c.lower()]
    n_errors = result_df[error_cols].notna().any(axis=1).sum() if error_cols else 0
    result_df_clean = result_df.drop(columns=error_cols, errors='ignore')
    
    result_df_clean.to_csv(OUTPUT_FILE, index=False)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"  Locations: {len(result_df_clean)}")
    print(f"  Errors: {n_errors}")
    print(f"  Output: {OUTPUT_FILE}")
    
    # Show coverage summary
    for radius_m in BUFFER_RADII_M:
        r = f"{radius_m // 1000}km"
        col = f'lc_{r}_pct_built_up'
        if col in result_df_clean.columns:
            coverage = result_df_clean[col].notna().sum()
            print(f"  {r} buffer coverage: {coverage}/{len(result_df_clean)} ({100*coverage/len(result_df_clean):.1f}%)")
    
    # Show mean land cover breakdown at 5km
    print(f"\n  Mean land cover at 5km buffer:")
    for cls_name in ['tree_cover', 'shrubland', 'grassland', 'cropland', 'built_up', 'bare_sparse', 'water']:
        col = f'lc_5km_pct_{cls_name}'
        if col in result_df_clean.columns:
            mean_val = result_df_clean[col].mean()
            print(f"    {cls_name:20s}: {mean_val:6.2f}%")
    
    hi_col = 'lc_5km_human_impact'
    if hi_col in result_df_clean.columns:
        print(f"    {'human_impact':20s}: {result_df_clean[hi_col].mean():6.2f}%")
    
    # Clean up checkpoint
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print(f"\n  Checkpoint file removed.")
    
    print(f"\n  Done! Run the model script next.")

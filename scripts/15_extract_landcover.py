"""
============================================================
EY Water Quality Challenge - Land Cover Extraction (v2)
============================================================
Extracts land cover percentages within buffer zones (1km, 5km, 10km)
around each unique river sampling location.

SOURCE: ESA CCI Land Cover (300m resolution) via Planetary Computer
  Collection: 'esa-cci-lc' (COG tiles)
  Year: 2013 (midpoint of our 2011-2015 study period)
  
WHY ESA CCI INSTEAD OF ESA WORLDCOVER:
  - WorldCover is 10m resolution but only available for 2020/2021
  - Our training data is 2011-2015 — using 2021 land cover introduces
    temporal mismatch (urban expansion, new mines, etc.)
  - ESA CCI provides ANNUAL maps from 1992-2020 at 300m resolution
  - 300m is still plenty for our 1km/5km/10km buffer analysis
  - Using 2013 gives us land cover contemporaneous with our data

ESA CCI LAND COVER CLASSES (UN FAO LCCS):
  10  = Cropland, rainfed
  11  = Herbaceous cover
  12  = Tree or shrub cover
  20  = Cropland, irrigated or post-flooding
  30  = Mosaic cropland/natural vegetation
  40  = Mosaic natural vegetation/cropland
  50  = Tree cover, broadleaved, evergreen
  60  = Tree cover, broadleaved, deciduous, closed
  61  = Tree cover, broadleaved, deciduous, closed (>40%)
  62  = Tree cover, broadleaved, deciduous, open (15-40%)
  70  = Tree cover, needleleaved, evergreen
  80  = Tree cover, needleleaved, deciduous
  90  = Tree cover, mixed leaf type
  100 = Mosaic tree/shrub and herbaceous
  110 = Mosaic herbaceous/tree and shrub
  120 = Shrubland
  130 = Grassland
  140 = Lichens and mosses
  150 = Sparse vegetation
  160 = Tree cover, flooded (fresh or brackish)
  170 = Tree cover, flooded, saline water (mangrove)
  180 = Shrub or herbaceous cover, flooded
  190 = Urban areas
  200 = Bare areas
  210 = Water bodies
  220 = Permanent snow and ice

OUTPUT: data/extracted/landcover_data.csv
ESTIMATED TIME: 5-30 minutes
============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import numpy as np
import pandas as pd
import time
from collections import Counter

try:
    import pystac_client
    import planetary_computer as pc
    import rioxarray
    import xarray as xr
    from rasterio.errors import RasterioIOError
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    print("Install with: pip install planetary_computer pystac_client rioxarray xarray")
    sys.exit(1)

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'ey_challenge_notebooks')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'data', 'extracted')
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_YEAR = 2013  # Midpoint of 2011-2015 study period
BUFFER_RADII_M = [1000, 5000, 10000]  # 1km, 5km, 10km

# ============================================================
# SIMPLIFIED LAND COVER CLASS MAPPING
# Maps ESA CCI pixel values → simplified class name
# ============================================================
LC_SIMPLIFIED = {}

# Cropland (rainfed, irrigated, mosaic)
for v in [10, 11, 12, 20, 30]:
    LC_SIMPLIFIED[v] = 'cropland'

# Forest / Tree cover (all types including mosaic with >50% tree)
for v in [40, 50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 100]:
    LC_SIMPLIFIED[v] = 'forest'

# Shrubland (including mosaic herbaceous)
for v in [110, 120, 121, 122]:
    LC_SIMPLIFIED[v] = 'shrubland'

# Grassland
LC_SIMPLIFIED[130] = 'grassland'

# Sparse vegetation (including lichens/mosses)
for v in [140, 150, 151, 152, 153]:
    LC_SIMPLIFIED[v] = 'sparse_veg'

# Wetland (flooded forests, mangroves, herbaceous wetland)
for v in [160, 170, 180]:
    LC_SIMPLIFIED[v] = 'wetland'

# Urban / Built-up
LC_SIMPLIFIED[190] = 'urban'

# Bare areas (including rock, sand, etc.)
for v in [200, 201, 202]:
    LC_SIMPLIFIED[v] = 'bare'

# Water bodies
LC_SIMPLIFIED[210] = 'water'

# Snow and ice
LC_SIMPLIFIED[220] = 'snow_ice'

SIMPLIFIED_CLASSES = ['cropland', 'forest', 'shrubland', 'grassland', 'sparse_veg',
                      'wetland', 'urban', 'bare', 'water', 'snow_ice']

OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'landcover_data.csv')
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, 'landcover_checkpoint.csv')


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_unique_locations():
    """Load all unique lat/lon from training + validation datasets."""
    train = pd.read_csv(os.path.join(DATA_DIR, 'water_quality_training_dataset.csv'))
    val = pd.read_csv(os.path.join(DATA_DIR, 'submission_template.csv'))
    
    all_locs = pd.concat([
        train[['Latitude', 'Longitude']],
        val[['Latitude', 'Longitude']]
    ]).drop_duplicates().reset_index(drop=True)
    
    print(f"  Unique locations: {len(all_locs)}")
    print(f"  Lat range: {all_locs['Latitude'].min():.4f} to {all_locs['Latitude'].max():.4f}")
    print(f"  Lon range: {all_locs['Longitude'].min():.4f} to {all_locs['Longitude'].max():.4f}")
    return all_locs


def meters_to_degrees(meters, latitude):
    """Convert buffer radius in meters to approximate degrees."""
    lat_deg = meters / 111320.0
    lon_deg = meters / (111320.0 * np.cos(np.radians(abs(latitude))))
    return lat_deg, lon_deg


def extract_landcover_for_location(lat, lon, lc_data, lc_x_coords, lc_y_coords):
    """
    Extract land cover percentages at multiple buffer radii for one location.
    
    Args:
        lat, lon: Location coordinates
        lc_data: 2D numpy array of land cover values
        lc_x_coords: 1D array of x (longitude) coordinates
        lc_y_coords: 1D array of y (latitude) coordinates
    
    Returns:
        dict with land cover percentages for each buffer and class
    """
    result = {'Latitude': lat, 'Longitude': lon}
    
    for radius_m in BUFFER_RADII_M:
        radius_label = f"lc_{radius_m // 1000}km"
        
        try:
            lat_buf, lon_buf = meters_to_degrees(radius_m, lat)
            
            # Find pixel indices within the buffer
            x_mask = (lc_x_coords >= lon - lon_buf) & (lc_x_coords <= lon + lon_buf)
            y_mask = (lc_y_coords >= lat - lat_buf) & (lc_y_coords <= lat + lat_buf)
            
            if not np.any(x_mask) or not np.any(y_mask):
                for cls in SIMPLIFIED_CLASSES:
                    result[f'{radius_label}_pct_{cls}'] = np.nan
                continue
            
            # Extract the buffer subset
            y_indices = np.where(y_mask)[0]
            x_indices = np.where(x_mask)[0]
            
            subset = lc_data[y_indices[0]:y_indices[-1]+1, x_indices[0]:x_indices[-1]+1]
            values = subset.flatten()
            
            # Remove nodata (0 or NaN)
            values = values[values > 0]
            values = values[~np.isnan(values)]
            
            if len(values) == 0:
                for cls in SIMPLIFIED_CLASSES:
                    result[f'{radius_label}_pct_{cls}'] = np.nan
                continue
            
            total = len(values)
            
            # Count each simplified class
            class_counts = Counter()
            for v in values:
                cls = LC_SIMPLIFIED.get(int(v), 'other')
                if cls != 'other':
                    class_counts[cls] += 1
            
            # Calculate percentages
            for cls in SIMPLIFIED_CLASSES:
                pct = (class_counts.get(cls, 0) / total) * 100.0
                result[f'{radius_label}_pct_{cls}'] = round(pct, 2)
            
            # Also store dominant class and pixel count
            result[f'{radius_label}_n_pixels'] = total
            if class_counts:
                result[f'{radius_label}_dominant'] = class_counts.most_common(1)[0][0]
            
        except Exception as e:
            for cls in SIMPLIFIED_CLASSES:
                result[f'{radius_label}_pct_{cls}'] = np.nan
            print(f"    WARNING: Error at ({lat:.4f}, {lon:.4f}) r={radius_m}m: {e}")
    
    # ---- Derived features (across all radii) ----
    for radius_m in BUFFER_RADII_M:
        r = f"lc_{radius_m // 1000}km"
        
        urban = result.get(f'{r}_pct_urban', 0) or 0
        crop = result.get(f'{r}_pct_cropland', 0) or 0
        bare = result.get(f'{r}_pct_bare', 0) or 0
        forest = result.get(f'{r}_pct_forest', 0) or 0
        grass = result.get(f'{r}_pct_grassland', 0) or 0
        shrub = result.get(f'{r}_pct_shrubland', 0) or 0
        water = result.get(f'{r}_pct_water', 0) or 0
        wetland = result.get(f'{r}_pct_wetland', 0) or 0
        
        # Human impact index = urban + cropland + bare (mining/degraded)
        result[f'{r}_human_impact'] = round(urban + crop + bare, 2)
        
        # Natural vegetation = forest + grassland + shrubland
        result[f'{r}_natural_veg'] = round(forest + grass + shrub, 2)
        
        # Urban density ratio (excluding water)
        total_land = 100.0 - water
        result[f'{r}_urban_ratio'] = round(urban / max(total_land, 1), 4)
        
        # Agriculture intensity ratio
        result[f'{r}_agri_ratio'] = round(crop / max(total_land, 1), 4)
        
        # Wetland proximity (wetland + water combined)
        result[f'{r}_water_wetland'] = round(water + wetland, 2)
    
    return result


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    t_start = time.time()
    
    print(f"""
{'='*70}
ESA CCI LAND COVER EXTRACTION (v2)
{'='*70}
Source:  ESA CCI Land Cover {TARGET_YEAR} (300m resolution)
Buffers: {[f"{r//1000}km" for r in BUFFER_RADII_M]}
Classes: {SIMPLIFIED_CLASSES}
{'='*70}
""")
    
    # ---- Step 1: Load locations ----
    print("[1/5] Loading locations...")
    locations = load_unique_locations()
    
    # ---- Step 2: Check for checkpoint ----
    done_locs = set()
    existing_results = []
    if os.path.exists(CHECKPOINT_FILE):
        checkpoint = pd.read_csv(CHECKPOINT_FILE)
        for _, row in checkpoint.iterrows():
            done_locs.add((round(row['Latitude'], 6), round(row['Longitude'], 6)))
            existing_results.append(row.to_dict())
        print(f"  Resuming from checkpoint: {len(done_locs)} locations already done")
    
    remaining = []
    for idx, row in locations.iterrows():
        key = (round(row['Latitude'], 6), round(row['Longitude'], 6))
        if key not in done_locs:
            remaining.append(idx)
    
    print(f"  Remaining: {len(remaining)} locations to extract")
    
    if len(remaining) == 0:
        print("  All locations already extracted!")
        if existing_results:
            pd.DataFrame(existing_results).to_csv(OUTPUT_FILE, index=False)
            print(f"  Saved: {OUTPUT_FILE}")
        sys.exit(0)
    
    # ---- Step 3: Connect to Planetary Computer ----
    print(f"\n[2/5] Connecting to Planetary Computer STAC API...")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )
    print("  Connected.")
    
    # South Africa bounding box (with extra margin for 10km buffers)
    lat_min = locations['Latitude'].min() - 0.15
    lat_max = locations['Latitude'].max() + 0.15
    lon_min = locations['Longitude'].min() - 0.15
    lon_max = locations['Longitude'].max() + 0.15
    sa_bbox = [lon_min, lat_min, lon_max, lat_max]
    print(f"  Search bbox: {[round(x, 2) for x in sa_bbox]}")
    
    # ---- Step 4: Search and load ESA CCI LC ----
    print(f"\n[3/5] Searching for ESA CCI LC {TARGET_YEAR} tiles...")
    
    lc_data = None
    lc_x = None
    lc_y = None
    source_name = None
    
    # --- Attempt 1: COG tiles (esa-cci-lc) ---
    try:
        search = catalog.search(
            collections=['esa-cci-lc'],
            bbox=sa_bbox,
            datetime=f"{TARGET_YEAR}-01-01/{TARGET_YEAR}-12-31",
        )
        items = list(search.items())
        print(f"  esa-cci-lc: Found {len(items)} tiles for {TARGET_YEAR}")
        
        if items:
            # ESA CCI LC COG tiles — need to mosaic multiple tiles
            print(f"  Loading {len(items)} COG tiles...")
            
            # Check available assets in first item
            sample_item = items[0]
            print(f"  Sample item ID: {sample_item.id}")
            print(f"  Assets: {list(sample_item.assets.keys())}")
            
            # Find the land cover classification asset
            lc_asset_key = None
            for key in ['lccs_class', 'data', 'map', 'land-cover']:
                if key in sample_item.assets:
                    lc_asset_key = key
                    break
            
            if lc_asset_key is None:
                # Try first data-like asset
                skip = {'tilejson', 'rendered_preview', 'thumbnail'}
                for key in sample_item.assets:
                    if key not in skip:
                        lc_asset_key = key
                        break
            
            print(f"  Using asset key: '{lc_asset_key}'")
            
            if len(items) == 1:
                # Single tile covering the area
                href = items[0].assets[lc_asset_key].href
                print(f"  Loading single tile: {href[:80]}...")
                raster = rioxarray.open_rasterio(href, chunks='auto')
                
                # Clip to SA
                raster = raster.rio.clip_box(
                    minx=lon_min, miny=lat_min, 
                    maxx=lon_max, maxy=lat_max
                )
                
                if len(raster.shape) == 3:
                    raster = raster[0]  # Take first band
                
                raster = raster.compute()
                lc_data = raster.values
                lc_x = raster.x.values
                lc_y = raster.y.values
                source_name = f"ESA CCI LC {TARGET_YEAR} (COG)"
                
            else:
                # Multiple tiles — mosaic them using rioxarray.merge
                from rioxarray.merge import merge_arrays
                
                print(f"  Loading and merging {len(items)} tiles...")
                arrays = []
                for i, item in enumerate(items):
                    href = item.assets[lc_asset_key].href
                    try:
                        arr = rioxarray.open_rasterio(href)
                        arr = arr.rio.clip_box(
                            minx=lon_min, miny=lat_min,
                            maxx=lon_max, maxy=lat_max
                        )
                        if arr.size > 0:
                            arrays.append(arr)
                            print(f"    Tile {i+1}/{len(items)}: {arr.shape}")
                    except Exception as e:
                        print(f"    Tile {i+1}/{len(items)}: skipped ({e})")
                
                if arrays:
                    merged = merge_arrays(arrays)
                    if len(merged.shape) == 3:
                        merged = merged[0]
                    merged = merged.compute()
                    lc_data = merged.values
                    lc_x = merged.x.values
                    lc_y = merged.y.values
                    source_name = f"ESA CCI LC {TARGET_YEAR} (COG, {len(arrays)} tiles)"
                    
    except Exception as e:
        print(f"  COG approach failed: {e}")
    
    # --- Attempt 2: NetCDF (esa-cci-lc-netcdf) ---
    if lc_data is None:
        print(f"\n  Trying NetCDF collection (esa-cci-lc-netcdf)...")
        try:
            search = catalog.search(
                collections=['esa-cci-lc-netcdf'],
                datetime=f"{TARGET_YEAR}-01-01/{TARGET_YEAR}-12-31",
            )
            nc_items = list(search.items())
            print(f"  Found {len(nc_items)} NetCDF items")
            
            if nc_items:
                nc_item = nc_items[0]
                print(f"  Item: {nc_item.id}")
                print(f"  Assets: {list(nc_item.assets.keys())}")
                
                # Find the NetCDF asset
                nc_href = None
                for key in ['lccs_class', 'netcdf', 'data']:
                    if key in nc_item.assets:
                        nc_href = nc_item.assets[key].href
                        print(f"  Using asset '{key}': {nc_href[:80]}...")
                        break
                
                if nc_href is None:
                    skip = {'tilejson', 'rendered_preview', 'thumbnail'}
                    for key, asset in nc_item.assets.items():
                        if key not in skip:
                            nc_href = asset.href
                            print(f"  Using asset '{key}': {nc_href[:80]}...")
                            break
                
                if nc_href:
                    ds = xr.open_dataset(nc_href, engine='h5netcdf')
                    print(f"  Variables: {list(ds.data_vars)}")
                    print(f"  Dims: {dict(ds.dims)}")
                    
                    # Get the land cover variable
                    lc_var = None
                    for var_name in ['lccs_class', 'land_cover', 'lc', 'lccs-class']:
                        if var_name in ds:
                            lc_var = var_name
                            break
                    
                    if lc_var is None:
                        lc_var = list(ds.data_vars)[0]
                        print(f"  Using first variable: {lc_var}")
                    
                    # Determine coordinate names
                    lat_name = 'lat' if 'lat' in ds.coords else 'latitude'
                    lon_name = 'lon' if 'lon' in ds.coords else 'longitude'
                    
                    # Subset to South Africa
                    lats = ds[lat_name].values
                    if lats[0] > lats[-1]:  # Descending
                        subset = ds[lc_var].sel(
                            **{lat_name: slice(lat_max, lat_min),
                               lon_name: slice(lon_min, lon_max)}
                        )
                    else:  # Ascending
                        subset = ds[lc_var].sel(
                            **{lat_name: slice(lat_min, lat_max),
                               lon_name: slice(lon_min, lon_max)}
                        )
                    
                    # Handle time dimension
                    if 'time' in subset.dims:
                        subset = subset.isel(time=0)
                    
                    subset = subset.compute()
                    lc_data = subset.values.astype(float)
                    lc_x = subset[lon_name].values
                    lc_y = subset[lat_name].values
                    source_name = f"ESA CCI LC {TARGET_YEAR} (NetCDF)"
                    
                    ds.close()
                    
        except Exception as e:
            print(f"  NetCDF approach failed: {e}")
            import traceback
            traceback.print_exc()
    
    # --- Attempt 3: Fallback to ESA WorldCover 2021 ---
    if lc_data is None:
        print(f"\n  FALLBACK: Using ESA WorldCover 2021 (10m)...")
        print(f"  NOTE: This is 2021 data, 6-10 years after study period.")
        print(f"  Large-scale patterns should still be similar.")
        
        try:
            search = catalog.search(
                collections=['esa-worldcover'],
                bbox=sa_bbox,
            )
            wc_items = list(search.items())
            print(f"  Found {len(wc_items)} WorldCover tiles")
            
            if wc_items:
                # WorldCover has different class values — remap to our scheme
                WC_REMAP = {
                    10: 'forest',       # Tree cover
                    20: 'shrubland',    # Shrubland
                    30: 'grassland',    # Grassland
                    40: 'cropland',     # Cropland
                    50: 'urban',        # Built-up
                    60: 'bare',         # Bare / sparse vegetation
                    70: 'snow_ice',     # Snow and ice
                    80: 'water',        # Permanent water bodies
                    90: 'wetland',      # Herbaceous wetland
                    95: 'wetland',      # Mangroves
                    100: 'sparse_veg',  # Moss and lichen
                }
                
                # Load tiles covering SA — WorldCover is high res, so be careful
                # Use overview (lower res) to avoid massive memory use
                from rioxarray.merge import merge_arrays
                
                arrays = []
                for i, item in enumerate(wc_items[:20]):  # Limit tiles
                    asset_key = 'map' if 'map' in item.assets else list(
                        k for k in item.assets if k not in {'tilejson', 'rendered_preview', 'thumbnail'}
                    )[0]
                    href = item.assets[asset_key].href
                    try:
                        # Open at reduced resolution (overview level 4 ≈ 160m)
                        arr = rioxarray.open_rasterio(href, overview_level=4)
                        arr = arr.rio.clip_box(
                            minx=lon_min, miny=lat_min,
                            maxx=lon_max, maxy=lat_max
                        )
                        if arr.size > 0:
                            arrays.append(arr)
                            print(f"    Tile {i+1}: loaded ({arr.shape})")
                    except Exception as e:
                        print(f"    Tile {i+1}: skipped ({str(e)[:50]})")
                
                if arrays:
                    merged = merge_arrays(arrays)
                    if len(merged.shape) == 3:
                        merged = merged[0]
                    merged = merged.compute()
                    lc_data = merged.values.astype(float)
                    lc_x = merged.x.values
                    lc_y = merged.y.values
                    
                    # Override the LC_SIMPLIFIED mapping for WorldCover values
                    LC_SIMPLIFIED.clear()
                    for wc_val, cls_name in WC_REMAP.items():
                        LC_SIMPLIFIED[wc_val] = cls_name
                    
                    source_name = "ESA WorldCover 2021 (fallback)"
                    
        except Exception as e:
            print(f"  WorldCover fallback also failed: {e}")
            import traceback
            traceback.print_exc()
    
    # --- Check if we have data ---
    if lc_data is None:
        print("\n  ERROR: Could not load any land cover data!")
        print("  Please check your internet connection and Planetary Computer access.")
        sys.exit(1)
    
    print(f"\n  SUCCESS: Loaded {source_name}")
    print(f"  Raster shape: {lc_data.shape}")
    print(f"  X range: {lc_x.min():.4f} to {lc_x.max():.4f}")
    print(f"  Y range: {lc_y.min():.4f} to {lc_y.max():.4f}")
    print(f"  Pixel size: ~{abs(lc_x[1]-lc_x[0])*111320:.0f}m x ~{abs(lc_y[1]-lc_y[0])*111320:.0f}m")
    
    # Ensure y is sorted (descending for lat, or ascending)
    if lc_y[0] > lc_y[-1]:
        # y is descending (typical for rasters: north at top)
        # We need to flip for our indexing to work
        lc_data = lc_data[::-1, :]
        lc_y = lc_y[::-1]
        print(f"  Flipped Y to ascending order")
    
    # Show unique values
    sample_vals = np.unique(lc_data[lc_data > 0])
    if len(sample_vals) > 20:
        print(f"  Unique values: {sample_vals[:10]}... ({len(sample_vals)} total)")
    else:
        print(f"  Unique values: {sample_vals}")
    
    # ---- Step 5: Extract for each location ----
    print(f"\n[4/5] Extracting land cover for {len(remaining)} locations...")
    
    all_results = list(existing_results)
    errors = 0
    
    for i, idx in enumerate(remaining):
        lat = locations.loc[idx, 'Latitude']
        lon = locations.loc[idx, 'Longitude']
        
        result = extract_landcover_for_location(lat, lon, lc_data, lc_x, lc_y)
        all_results.append(result)
        
        # Progress
        if (i + 1) % 25 == 0 or (i + 1) == len(remaining):
            pct = (i + 1) / len(remaining) * 100
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (len(remaining) - i - 1) if i > 0 else 0
            print(f"    [{i+1:4d}/{len(remaining)}] ({pct:5.1f}%)  "
                  f"Elapsed: {elapsed:.0f}s  ETA: {eta:.0f}s  "
                  f"Loc: ({lat:.4f}, {lon:.4f})")
        
        # Checkpoint every 100 locations
        if (i + 1) % 100 == 0:
            pd.DataFrame(all_results).to_csv(CHECKPOINT_FILE, index=False)
    
    # ---- Step 6: Save results ----
    print(f"\n[5/5] Saving results...")
    
    result_df = pd.DataFrame(all_results)
    
    # Remove internal columns (n_pixels, dominant) — keep for debugging
    # They're actually useful features, keep them
    
    result_df.to_csv(OUTPUT_FILE, index=False)
    
    # ---- Summary ----
    elapsed_total = time.time() - t_start
    
    print(f"""
{'='*70}
EXTRACTION COMPLETE
{'='*70}
  Source:    {source_name}
  Locations: {len(result_df)}
  Time:      {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)
  Output:    {OUTPUT_FILE}
""")
    
    # Coverage stats
    for radius_m in BUFFER_RADII_M:
        r = f"lc_{radius_m // 1000}km"
        col = f'{r}_pct_urban'
        if col in result_df.columns:
            cov = result_df[col].notna().sum()
            total = len(result_df)
            print(f"  {r} coverage: {cov}/{total} ({cov/total*100:.1f}%)")
    
    # Mean land cover at 5km buffer
    print(f"\n  Mean land cover within 5km buffer:")
    print(f"  {'Class':15s}  {'Mean %':>8s}  {'Std %':>8s}")
    print(f"  {'-'*35}")
    for cls in SIMPLIFIED_CLASSES:
        col = f'lc_5km_pct_{cls}'
        if col in result_df.columns:
            m = result_df[col].mean()
            s = result_df[col].std()
            bar = '█' * int(m / 2)
            print(f"  {cls:15s}  {m:7.2f}%  {s:7.2f}%  {bar}")
    
    # Derived features
    for feat in ['human_impact', 'natural_veg', 'water_wetland']:
        col = f'lc_5km_{feat}'
        if col in result_df.columns:
            print(f"  {feat:15s}  {result_df[col].mean():7.2f}%  {result_df[col].std():7.2f}%")
    
    # Cleanup checkpoint
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
    
    print(f"""
{'='*70}
NEXT STEPS
{'='*70}
  1. Check the data: Open {OUTPUT_FILE}
  2. Run geology extraction: python scripts/16_extract_geology.py
  3. Build next model version with land cover features
{'='*70}
""")

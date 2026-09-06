"""
============================================================
EY Water Quality Challenge - Slope Extraction
============================================================
WHY SLOPE?
  Research paper found slope was the #2 most important feature
  for predicting phosphate (DRP), representing ~20% of variable
  importance in XGBoost. Slope controls:
    - Water flow speed (steep = fast runoff, less infiltration)
    - Erosion rate (steep = more sediment + phosphorus transport)
    - Residence time (flat = water sits longer, more dissolution)

METHOD:
  For each location, fetch elevation at 4 nearby points
  (N, S, E, W at ~500m offset) using Open-Elevation API,
  then compute slope as max gradient.

  slope = arctan(rise / run) in degrees
  
  Also computes:
    - aspect (dominant downhill direction)
    - terrain_roughness (std of nearby elevations)
    - elevation already exists, just adds slope-derived features

OUTPUT:
  data/extracted/slope_data.csv
  Columns: Latitude, Longitude, slope_deg, aspect_deg, terrain_roughness

RUNTIME: ~2-5 minutes (depends on API speed)

PREREQUISITES:
  pip install requests numpy pandas tqdm
============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
import time
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'Jupyter Notebook Package')
EXTRACTED_DIR = os.path.join(BASE_DIR, '..', 'data', 'extracted')
os.makedirs(EXTRACTED_DIR, exist_ok=True)

# Offset in degrees (~500m at South Africa's latitude ~30°S)
# 1 degree latitude ≈ 111km, so 500m ≈ 0.0045 degrees
# 1 degree longitude at -30° ≈ 96km, so 500m ≈ 0.0052 degrees
LAT_OFFSET = 0.0045
LON_OFFSET = 0.0052
DISTANCE_M = 500.0  # approximate distance between center and neighbor


def load_unique_locations():
    """Load all unique locations from training + validation."""
    train = pd.read_csv(os.path.join(DATA_DIR, 'water_quality_training_dataset.csv'))
    val = pd.read_csv(os.path.join(DATA_DIR, 'submission_template.csv'))
    
    all_locs = pd.concat([
        train[['Latitude', 'Longitude']],
        val[['Latitude', 'Longitude']]
    ]).drop_duplicates().reset_index(drop=True)
    
    print(f"  Unique locations to process: {len(all_locs)}")
    return all_locs


def fetch_elevations_batch(locations, max_retries=3):
    """
    Fetch elevations for a batch of (lat, lon) pairs.
    Uses POST method with JSON body for reliability.
    Returns dict of (lat, lon) -> elevation.
    """
    loc_list = [{"latitude": lat, "longitude": lon} for lat, lon in locations]
    
    for attempt in range(max_retries):
        try:
            # Try POST first (more reliable for larger batches)
            response = requests.post(
                "https://api.open-elevation.com/api/v1/lookup",
                json={"locations": loc_list},
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                result = {}
                for i, r in enumerate(data['results']):
                    lat, lon = locations[i]
                    result[(round(lat, 8), round(lon, 8))] = r['elevation']
                return result
            
            # Fallback to GET
            loc_string = "|".join([f"{lat},{lon}" for lat, lon in locations])
            response = requests.get(
                f"https://api.open-elevation.com/api/v1/lookup?locations={loc_string}",
                timeout=60
            )
            if response.status_code == 200:
                data = response.json()
                result = {}
                for i, r in enumerate(data['results']):
                    lat, lon = locations[i]
                    result[(round(lat, 8), round(lon, 8))] = r['elevation']
                return result
                
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                print(f"    Retry {attempt+1}/{max_retries} after {wait}s: {e}")
                time.sleep(wait)
            else:
                print(f"    Failed after {max_retries} attempts: {e}")
    
    return {}


def compute_slope_for_locations(unique_locs):
    """
    For each location, get elevations at center + 4 neighbors (N, S, E, W),
    then compute slope, aspect, and terrain roughness.
    
    Slope computation:
      dz_dx = (elev_E - elev_W) / (2 * DISTANCE_M)  [east-west gradient]
      dz_dy = (elev_N - elev_S) / (2 * DISTANCE_M)  [north-south gradient]
      slope = arctan(sqrt(dz_dx² + dz_dy²)) in degrees
      aspect = arctan2(-dz_dy, dz_dx) converted to compass degrees
    """
    
    # Step 1: Build list of ALL points we need to query
    # For each location: center + N + S + E + W = 5 points
    all_query_points = []
    location_point_map = {}  # maps loc_idx -> {center, N, S, E, W} indices in all_query_points
    
    for idx, row in unique_locs.iterrows():
        lat, lon = row['Latitude'], row['Longitude']
        
        points = {
            'center': (lat, lon),
            'N': (lat + LAT_OFFSET, lon),
            'S': (lat - LAT_OFFSET, lon),
            'E': (lat, lon + LON_OFFSET),
            'W': (lat, lon - LON_OFFSET),
        }
        
        point_indices = {}
        for name, (plat, plon) in points.items():
            point_indices[name] = len(all_query_points)
            all_query_points.append((plat, plon))
        
        location_point_map[idx] = point_indices
    
    print(f"  Total elevation points to fetch: {len(all_query_points)}")
    print(f"  ({len(unique_locs)} locations × 5 points each)")
    
    # Step 2: Fetch elevations in batches
    # Open-Elevation API handles ~100 points per request well
    BATCH_SIZE = 80
    all_elevations = {}
    
    n_batches = (len(all_query_points) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"  Fetching in {n_batches} batches of {BATCH_SIZE}...")
    
    for i in tqdm(range(0, len(all_query_points), BATCH_SIZE), desc="  Fetching elevations"):
        batch = all_query_points[i:i + BATCH_SIZE]
        result = fetch_elevations_batch(batch)
        all_elevations.update(result)
        
        # Rate limit: be gentle with the free API
        time.sleep(1.0)
    
    valid_count = sum(1 for v in all_elevations.values() if v is not None)
    print(f"  Got {valid_count}/{len(all_query_points)} valid elevations")
    
    # Step 3: Compute slope, aspect, roughness for each location
    results = []
    
    for idx, row in unique_locs.iterrows():
        lat, lon = row['Latitude'], row['Longitude']
        point_indices = location_point_map[idx]
        
        # Get elevations for each direction
        elevs = {}
        for name, pt_idx in point_indices.items():
            pt = all_query_points[pt_idx]
            key = (round(pt[0], 8), round(pt[1], 8))
            elevs[name] = all_elevations.get(key, None)
        
        # Check if we have all 5 elevations
        if all(v is not None for v in elevs.values()):
            elev_c = elevs['center']
            elev_n = elevs['N']
            elev_s = elevs['S']
            elev_e = elevs['E']
            elev_w = elevs['W']
            
            # Gradient in x (east-west) and y (north-south) directions
            dz_dx = (elev_e - elev_w) / (2 * DISTANCE_M)
            dz_dy = (elev_n - elev_s) / (2 * DISTANCE_M)
            
            # Slope in degrees
            slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
            slope_deg = np.degrees(slope_rad)
            
            # Aspect (compass direction of steepest descent)
            # arctan2(-dz_dy, dz_dx) gives angle from east, convert to compass
            aspect_rad = np.arctan2(-dz_dy, dz_dx)
            aspect_deg = np.degrees(aspect_rad)
            # Convert from math angles to compass (0=N, 90=E, 180=S, 270=W)
            aspect_compass = (90.0 - aspect_deg) % 360.0
            
            # Terrain roughness (std of the 5 elevation values)
            all_5 = [elev_c, elev_n, elev_s, elev_e, elev_w]
            terrain_roughness = np.std(all_5)
            
            # Elevation range in neighborhood
            elev_range = max(all_5) - min(all_5)
            
            results.append({
                'Latitude': lat,
                'Longitude': lon,
                'slope_deg': round(slope_deg, 4),
                'aspect_deg': round(aspect_compass, 2),
                'terrain_roughness': round(terrain_roughness, 4),
                'elev_range_500m': round(elev_range, 2),
            })
        else:
            # Missing data — store NaN
            n_missing = sum(1 for v in elevs.values() if v is None)
            results.append({
                'Latitude': lat,
                'Longitude': lon,
                'slope_deg': np.nan,
                'aspect_deg': np.nan,
                'terrain_roughness': np.nan,
                'elev_range_500m': np.nan,
            })
    
    return pd.DataFrame(results)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    start_time = time.time()
    
    print("=" * 70)
    print("EY WATER QUALITY - SLOPE EXTRACTION")
    print("=" * 70)
    
    # Load locations
    print("\n[1] Loading unique locations...")
    unique_locs = load_unique_locations()
    
    # Check if we already have slope data
    slope_path = os.path.join(EXTRACTED_DIR, 'slope_data.csv')
    if os.path.exists(slope_path):
        existing = pd.read_csv(slope_path)
        print(f"\n  WARNING: slope_data.csv already exists ({len(existing)} rows)")
        print(f"  Columns: {list(existing.columns)}")
        
        # Check if it covers all locations
        existing_locs = set(zip(existing['Latitude'].round(6), existing['Longitude'].round(6)))
        needed_locs = set(zip(unique_locs['Latitude'].round(6), unique_locs['Longitude'].round(6)))
        missing = needed_locs - existing_locs
        
        if len(missing) == 0:
            print(f"  All {len(needed_locs)} locations already covered. Skipping extraction.")
            print(f"  Delete slope_data.csv to re-extract.")
            
            # Print stats
            valid = existing['slope_deg'].notna().sum()
            print(f"\n  Stats:")
            print(f"    Valid slope values: {valid}/{len(existing)}")
            if valid > 0:
                print(f"    Slope range: {existing['slope_deg'].min():.2f}° to {existing['slope_deg'].max():.2f}°")
                print(f"    Mean slope: {existing['slope_deg'].mean():.2f}°")
                print(f"    Terrain roughness range: {existing['terrain_roughness'].min():.2f} to {existing['terrain_roughness'].max():.2f}")
            exit(0)
        else:
            print(f"  Missing {len(missing)} locations. Re-extracting all...")
    
    # Extract slope
    print(f"\n[2] Computing slope from nearby elevations...")
    print(f"    Method: 4-neighbor gradient (N/S/E/W at ~500m)")
    print(f"    API: Open-Elevation (free, no auth)")
    
    slope_df = compute_slope_for_locations(unique_locs)
    
    # Save
    slope_df.to_csv(slope_path, index=False)
    
    # Summary
    elapsed = time.time() - start_time
    valid = slope_df['slope_deg'].notna().sum()
    
    print(f"\n{'=' * 70}")
    print(f"SLOPE EXTRACTION COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Saved: data/extracted/slope_data.csv")
    print(f"  Valid: {valid}/{len(slope_df)} locations")
    print(f"  Time: {elapsed:.0f} seconds")
    
    if valid > 0:
        print(f"\n  Statistics:")
        print(f"    Slope:     min={slope_df['slope_deg'].min():.2f}°, "
              f"max={slope_df['slope_deg'].max():.2f}°, "
              f"mean={slope_df['slope_deg'].mean():.2f}°")
        print(f"    Roughness: min={slope_df['terrain_roughness'].min():.2f}, "
              f"max={slope_df['terrain_roughness'].max():.2f}, "
              f"mean={slope_df['terrain_roughness'].mean():.2f}")
        print(f"    Elev range: min={slope_df['elev_range_500m'].min():.1f}m, "
              f"max={slope_df['elev_range_500m'].max():.1f}m")
        
        # Distribution check
        flat = (slope_df['slope_deg'] < 1).sum()
        moderate = ((slope_df['slope_deg'] >= 1) & (slope_df['slope_deg'] < 5)).sum()
        steep = (slope_df['slope_deg'] >= 5).sum()
        print(f"\n    Flat (<1°): {flat} locations")
        print(f"    Moderate (1-5°): {moderate} locations")
        print(f"    Steep (>5°): {steep} locations")
    
    print(f"\n  NEXT: Run 08_feature_engineering_v4.py to merge all features")

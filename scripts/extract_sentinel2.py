"""
============================================================
EY Water Quality Challenge - Sentinel-2 Full Extraction
============================================================
Extracts ALL key Sentinel-2 L2A bands for every sample
in training and validation datasets via Planetary Computer.

Bands extracted:
  B2  - Blue (490nm)       -> Water penetration, turbidity proxy
  B3  - Green (560nm)      -> Chlorophyll, algae
  B4  - Red (665nm)        -> Chlorophyll absorption (critical)
  B5  - Red Edge 1 (705nm) -> Chlorophyll concentration
  B6  - Red Edge 2 (740nm) -> Vegetation health
  B7  - Red Edge 3 (783nm) -> Canopy structure
  B8  - NIR (842nm)        -> NDVI numerator
  B8a - Narrow NIR (865nm) -> More sensitive than B8
  B9  - Water Vapour (945nm) -> Atmospheric correction proxy
  B11 - SWIR 1 (1610nm)   -> Soil moisture, turbidity
  B12 - SWIR 2 (2190nm)   -> Mineralogy, sediment

Derived Indices (computed from bands):
  NDVI   = (B8 - B4) / (B8 + B4)         -> Vegetation
  NDWI   = (B3 - B8) / (B3 + B8)         -> Water body extent
  MNDWI  = (B3 - B11) / (B3 + B11)       -> Modified water index
  NDRE   = (B8a - B5) / (B8a + B5)       -> Red-edge (chlorophyll)
  NDTI   = (B4 - B3) / (B4 + B3)         -> Turbidity
  FAI    = B8 - (B4 + (B11-B4)*0.4)      -> Floating algae

Strategy:
  For each sample (location + date), we search for the
  nearest cloud-free Sentinel-2 image within a +/-30-day
  window and extract pixel values at that location.

  Uses parallel processing with ThreadPoolExecutor for speed,
  and batch checkpointing for crash recovery.

Output:
  data/extracted/sentinel2_training.csv
  data/extracted/sentinel2_validation.csv
============================================================
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import stackstac
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

# --- PATHS ---------------------------------------------------------------
BASE_DIR   = Path(__file__).parent.parent          # project root
DATA_DIR   = BASE_DIR / "Jupyter Notebook Package"
OUT_DIR    = BASE_DIR / "data" / "extracted"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_OUT  = OUT_DIR / "sentinel2_training.csv"
VAL_OUT    = OUT_DIR / "sentinel2_validation.csv"

# --- SETTINGS -------------------------------------------------------------
BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
# Rename map: STAC band name -> friendly column name
BAND_RENAME = {
    "B02": "s2_blue",
    "B03": "s2_green",
    "B04": "s2_red",
    "B05": "s2_re1",
    "B06": "s2_re2",
    "B07": "s2_re3",
    "B08": "s2_nir",
    "B8A": "s2_nir8a",
    "B09": "s2_wv",
    "B11": "s2_swir1",
    "B12": "s2_swir2",
}

MAX_CLOUD_COVER    = 30      # Only use images with <30% cloud cover
SEARCH_WINDOW_DAYS = 30      # Search +/-30 days around sample date
BUFFER_METERS      = 30      # Pixel buffer around point (S2 = 10m resolution)
RETRY_LIMIT        = 3       # Retries per sample on failure
SLEEP_BETWEEN      = 0.02    # Seconds between requests (reduced for parallel)

# Parallel / batch settings
MAX_WORKERS        = 8       # Number of parallel threads
BATCH_SIZE         = 100     # Save checkpoint every N rows
BATCH_SEARCH_SIZE  = 20      # Number of samples to batch together for STAC search

# --- CONNECT TO PLANETARY COMPUTER ----------------------------------------
print("=" * 70)
print("EY WATER QUALITY - SENTINEL-2 EXTRACTION")
print("=" * 70)
print("\nConnecting to Planetary Computer STAC API...")
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)
print("Connected!")


# --- HELPER: SEARCH AND EXTRACT FOR ONE SAMPLE ----------------------------

def extract_one_sample(lat: float, lon: float, date_str: str) -> dict:
    """
    Given a location (lat, lon) and a sample date string ('YYYY-MM-DD'),
    finds the nearest cloud-free Sentinel-2 image within +/-30 days and
    returns band values as a dictionary.

    Returns a dict of band values, or all NaNs if no suitable image found.
    """
    nan_result = {v: np.nan for v in BAND_RENAME.values()}

    sample_date = pd.Timestamp(date_str)
    date_start  = (sample_date - pd.Timedelta(days=SEARCH_WINDOW_DAYS)).strftime("%Y-%m-%d")
    date_end    = (sample_date + pd.Timedelta(days=SEARCH_WINDOW_DAYS)).strftime("%Y-%m-%d")

    # Tiny bounding box around the point (S2 is 10m, 30m buffer = 3 pixels)
    delta = BUFFER_METERS / 111_320  # degrees per meter at equator
    bbox  = [lon - delta, lat - delta, lon + delta, lat + delta]

    # Search STAC catalog
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{date_start}/{date_end}",
        query={"eo:cloud_cover": {"lt": MAX_CLOUD_COVER}},
        sortby=[{"field": "properties.datetime", "direction": "asc"}],
        max_items=10,
    )

    items = list(search.items())
    if not items:
        return nan_result

    # Sort items by temporal closeness to sample date, then by cloud cover
    def sort_key(item):
        item_date = pd.Timestamp(item.datetime)
        days_diff  = abs((item_date - sample_date).days)
        cloud      = item.properties.get("eo:cloud_cover", 100)
        return (days_diff, cloud)

    items.sort(key=sort_key)
    best_item = items[0]

    # Sign the item for authenticated access
    signed_item = planetary_computer.sign(best_item)

    try:
        # Load only the bands we need using stackstac
        stack = stackstac.stack(
            [signed_item],
            assets=BANDS,
            bounds_latlon=bbox,
            resolution=10,
            rescale=False,
            fill_value=np.nan,
        )

        # Compute the spatial mean across the small buffer window
        arr = stack.mean(dim=["x", "y"]).compute()

        result = {}
        for band_id, col_name in BAND_RENAME.items():
            try:
                # Select band from the DataArray
                val = float(arr.sel(band=band_id).values[0])
                # Sentinel-2 L2A reflectance values are in range 0-10000 (scaled)
                # Convert to 0-1 reflectance
                result[col_name] = val / 10_000.0 if val > 0 else np.nan
            except Exception:
                result[col_name] = np.nan

        return result

    except Exception:
        return nan_result


def extract_one_sample_with_retry(idx: int, lat: float, lon: float, date_str: str) -> tuple:
    """
    Wrapper around extract_one_sample with retry logic.
    Returns (idx, band_vals_dict).
    """
    for attempt in range(RETRY_LIMIT):
        try:
            band_vals = extract_one_sample(lat, lon, date_str)
            return (idx, band_vals)
        except Exception:
            if attempt == RETRY_LIMIT - 1:
                return (idx, {v: np.nan for v in BAND_RENAME.values()})
            time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
    return (idx, {v: np.nan for v in BAND_RENAME.values()})


# --- DERIVED INDICES -------------------------------------------------------

def compute_indices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute spectral water quality indices from the extracted S2 bands.
    All inputs are expected to be in 0-1 reflectance scale.
    """
    eps = 1e-8  # small constant to avoid division by zero

    b3  = df["s2_green"]
    b4  = df["s2_red"]
    b5  = df["s2_re1"]
    b8  = df["s2_nir"]
    b8a = df["s2_nir8a"]
    b11 = df["s2_swir1"]

    # NDVI
    df["s2_ndvi"] = (b8 - b4) / (b8 + b4 + eps)

    # NDWI
    df["s2_ndwi"] = (b3 - b8) / (b3 + b8 + eps)

    # MNDWI
    df["s2_mndwi"] = (b3 - b11) / (b3 + b11 + eps)

    # NDRE
    df["s2_ndre"] = (b8a - b5) / (b8a + b5 + eps)

    # NDTI
    df["s2_ndti"] = (b4 - b3) / (b4 + b3 + eps)

    # FAI
    df["s2_fai"] = b8 - (b4 + (b11 - b4) * 0.4)

    # Band ratios
    df["s2_g_r"] = b3 / (b4 + eps)
    df["s2_re_ratio"] = b5 / (b4 + eps)
    df["s2_swir_ratio"] = b11 / (b8 + eps)

    return df


# --- MAIN EXTRACTION FUNCTION (PARALLEL) -----------------------------------

def run_extraction(input_csv_path: str, output_csv_path: str, split_name: str):
    """
    Reads the input CSV, extracts Sentinel-2 bands for each row using
    parallel threads, computes indices, and saves to output CSV.

    Supports resuming: if output CSV already exists, skips completed rows.
    """
    print(f"\n{'='*70}")
    print(f"  EXTRACTING {split_name.upper()}")
    print(f"{'='*70}")

    df = pd.read_csv(input_csv_path)
    print(f"  Loaded {len(df)} rows from {Path(input_csv_path).name}")

    # -- Column name normalization --
    lat_col  = next((c for c in df.columns if "lat"  in c.lower()), None)
    lon_col  = next((c for c in df.columns if "lon"  in c.lower()), None)
    date_col = next((c for c in df.columns if "date" in c.lower()), None)

    if not all([lat_col, lon_col, date_col]):
        raise ValueError(
            f"Could not find lat/lon/date columns. Found: {list(df.columns)}"
        )

    print(f"  Using columns -> lat: '{lat_col}', lon: '{lon_col}', date: '{date_col}'")

    # -- Resume support --
    completed_idx = set()
    if Path(output_csv_path).exists():
        done_df = pd.read_csv(output_csv_path)
        if "row_idx" in done_df.columns:
            completed_idx = set(done_df["row_idx"].tolist())
        else:
            completed_idx = set(done_df.index.tolist())
        print(f"  Resuming: {len(completed_idx)} rows already extracted, skipping...")
    else:
        done_df = pd.DataFrame()

    # -- Build work list --
    remaining_df = df[~df.index.isin(completed_idx)]
    total_remaining = len(remaining_df)
    print(f"  Rows to extract: {total_remaining}")
    print(f"  Parallel workers: {MAX_WORKERS}")
    print(f"  Batch checkpoint size: {BATCH_SIZE}")

    if total_remaining == 0:
        print("  Nothing to extract, all rows already done.")
        return done_df

    # -- Parallel extraction --
    new_results = []
    pbar = tqdm(total=total_remaining, desc=f"  S2 {split_name}", ncols=90)

    # Process in batches for checkpointing
    remaining_items = list(remaining_df.iterrows())

    for batch_start in range(0, len(remaining_items), BATCH_SIZE):
        batch = remaining_items[batch_start:batch_start + BATCH_SIZE]
        batch_results = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {}
            for idx, row in batch:
                lat  = float(row[lat_col])
                lon  = float(row[lon_col])
                date = str(row[date_col])[:10]  # Ensure YYYY-MM-DD format

                future = executor.submit(
                    extract_one_sample_with_retry, idx, lat, lon, date
                )
                futures[future] = (idx, row)

            for future in as_completed(futures):
                idx, row = futures[future]
                try:
                    result_idx, band_vals = future.result()
                    combined = row.to_dict()
                    combined.update(band_vals)
                    combined["row_idx"] = result_idx
                    batch_results.append(combined)
                except Exception:
                    combined = row.to_dict()
                    combined.update({v: np.nan for v in BAND_RENAME.values()})
                    combined["row_idx"] = idx
                    batch_results.append(combined)

                pbar.update(1)

        new_results.extend(batch_results)

        # Save checkpoint after each batch
        checkpoint_df = pd.DataFrame(new_results)
        if not done_df.empty:
            checkpoint_df = pd.concat([done_df, checkpoint_df], ignore_index=True)
        checkpoint_df.to_csv(output_csv_path, index=False)
        tqdm.write(f"    Checkpoint saved ({len(checkpoint_df)} rows total)")

    pbar.close()

    # -- Merge with already completed rows and save final output --
    final_df = pd.DataFrame(new_results)
    if not done_df.empty:
        final_df = pd.concat([done_df, final_df], ignore_index=True)

    # Compute spectral indices
    print(f"\n  Computing spectral indices...")
    final_df = compute_indices(final_df)

    # Drop helper column
    if "row_idx" in final_df.columns:
        final_df.drop(columns=["row_idx"], inplace=True)

    final_df.to_csv(output_csv_path, index=False)

    # -- Summary --
    s2_cols    = [c for c in final_df.columns if c.startswith("s2_")]
    nan_pct    = final_df[s2_cols].isna().mean().mean() * 100
    print(f"\n  {'_'*50}")
    print(f"  Done. Saved -> {output_csv_path}")
    print(f"  Total rows:     {len(final_df)}")
    print(f"  S2 columns:     {len(s2_cols)} ({', '.join(s2_cols[:5])}...)")
    print(f"  NaN rate:       {nan_pct:.1f}% (cloud cover or no image found)")
    print(f"  {'_'*50}")

    return final_df


# --- ENTRYPOINT ------------------------------------------------------------

if __name__ == "__main__":
    # Check inputs exist
    train_input = DATA_DIR / "water_quality_training_dataset.csv"
    val_input   = DATA_DIR / "submission_template.csv"

    if not train_input.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_input}")
    if not val_input.exists():
        raise FileNotFoundError(f"Validation CSV not found: {val_input}")

    print(f"\n  Input paths:")
    print(f"    Training:   {train_input}")
    print(f"    Validation: {val_input}")
    print(f"  Output paths:")
    print(f"    Training:   {TRAIN_OUT}")
    print(f"    Validation: {VAL_OUT}")
    print(f"\n  Settings:")
    print(f"    Bands:           {', '.join(BANDS)}")
    print(f"    Max cloud cover: {MAX_CLOUD_COVER}%")
    print(f"    Search window:   +/-{SEARCH_WINDOW_DAYS} days")
    print(f"    Buffer:          {BUFFER_METERS}m")
    print(f"    Parallel workers: {MAX_WORKERS}")
    print(f"    Batch size:      {BATCH_SIZE}")

    print(f"\n  Parallel extraction enabled ({MAX_WORKERS} threads).")
    print("  Saves checkpoints every batch -- safe to Ctrl+C and resume.")

    # Extract training data first (larger, takes longer)
    t_start = time.time()
    run_extraction(str(train_input), str(TRAIN_OUT), split_name="training")

    # Then validation data
    run_extraction(str(val_input), str(VAL_OUT), split_name="validation")

    elapsed = (time.time() - t_start) / 3600
    print(f"\n{'='*70}")
    print(f"  ALL DONE in {elapsed:.1f} hours")
    print(f"  Training  -> {TRAIN_OUT}")
    print(f"  Validation -> {VAL_OUT}")
    print(f"{'='*70}")

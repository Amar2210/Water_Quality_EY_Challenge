"""
============================================================
EY Water Quality Challenge - Full Landsat Band Extraction
============================================================
Re-extracts Landsat data with ALL available bands from
Planetary Computer, not just the 4 bands EY provided.

WHAT EY GAVE US (4 bands + 2 indices):
  green, nir08, swir16, swir22, NDMI, MNDWI

WHAT WE'RE EXTRACTING (7 bands + 10 indices):
  BANDS:
    blue    (0.48μm) -> Water clarity, turbidity proxy
    green   (0.56μm) -> Chlorophyll, algae (already had)
    red     (0.65μm) -> Chlorophyll absorption, NDVI
    nir08   (0.86μm) -> Vegetation, NDVI (already had)
    swir16  (1.6μm)  -> Soil moisture, turbidity (already had)
    swir22  (2.2μm)  -> Mineralogy, sediment (already had)
    lwir11  (10.9μm) -> Surface Temperature (Kelvin)

  DERIVED INDICES:
    NDVI  = (nir - red) / (nir + red)       -> Vegetation health
    NDWI  = (green - nir) / (green + nir)   -> Water body detection
    NDMI  = (nir - swir16) / (nir + swir16) -> Moisture content
    MNDWI = (green - swir16) / (green + swir16) -> Modified water
    NDTI  = (red - green) / (red + green)   -> Turbidity proxy
    blue_green_ratio  = blue / green        -> Water clarity
    red_blue_ratio    = red / blue          -> Sediment load
    nir_red_ratio     = nir / red           -> Veg health proxy
    green_swir_ratio  = green / swir22      -> Water composition
    blue_red_ratio    = blue / red          -> Scattering proxy

WHY THIS MATTERS:
  - Blue band: Research shows blue reflectance is the #1
    predictor for turbidity and suspended solids in water
  - Red band: Essential for real NDVI computation. Without
    it, we couldn't compute NDVI which correlates with
    chlorophyll-a and nutrient loading (affects DRP)
  - Thermal band (lwir11): Direct water surface temperature
    measurement. Strongly correlates with Electrical
    Conductance (EC) because warmer water = more dissolved
    ions = higher conductivity
  - Coastal aerosol: Only on Landsat 8 (post-2013), useful
    for atmospheric scattering correction

PARALLELIZATION:
  Uses ThreadPoolExecutor with configurable workers (default 10)
  + batch checkpointing every 200 rows for crash recovery
  + resume support (skips already-extracted rows)
  + retry with exponential backoff

COVERAGE:
  - Landsat 7 ETM+ (1999-present, SLC-off from 2003)
    -> Has: blue, green, red, nir08, swir16, swir22, lwir
    -> Missing: coastal
  - Landsat 8 OLI/TIRS (April 2013-present)
    -> Has: ALL bands including coastal and lwir11
  - Our date range: 2011-2015
    -> 2011-Apr2013: Landsat 7 only
    -> Apr2013-2015: Both Landsat 7 & 8 (prefer L8)

ESTIMATED TIME:
  Serial (EY approach): ~7 hours for 9519 samples
  Parallel (10 workers): ~45-90 minutes
============================================================
"""

import os
import sys
import time
import warnings
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

# --- Check dependencies ---------------------------------------------------
try:
    import planetary_computer as pc
    import pystac_client
    from odc.stac import stac_load
    from pystac.extensions.eo import EOExtension as eo
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install planetary-computer pystac-client odc-stac")
    sys.exit(1)

from tqdm import tqdm

# =========================================================================
# CONFIGURATION
# =========================================================================
BASE_DIR   = Path(__file__).parent.parent
DATA_DIR   = BASE_DIR / "Jupyter Notebook Package"
OUT_DIR    = BASE_DIR / "data" / "extracted"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_OUT  = OUT_DIR / "landsat_full_training.csv"
VAL_OUT    = OUT_DIR / "landsat_full_validation.csv"

# --- Extraction settings ---
# Bands to load via odc.stac (these are the STAC common names)
# For Surface Reflectance bands:
SR_BANDS = ["blue", "green", "red", "nir08", "swir16", "swir22"]
# For Surface Temperature band:
ST_BAND  = "lwir11"
# Coastal aerosol (Landsat 8+ only — will be NaN for Landsat 7)
COASTAL_BAND = "coastal"

# All bands we attempt to load
ALL_BANDS = SR_BANDS + [ST_BAND]
# Note: We load coastal separately since it fails on Landsat 7

MAX_CLOUD_COVER    = 10       # Same as EY's original setting
SEARCH_WINDOW_DAYS = 60       # +/-60 days from sample date for search
BBOX_BUFFER_DEG    = 0.00089831  # ~100m buffer (same as EY notebook)
SCALE_FACTOR_SR    = 0.0000275   # Landsat C2 L2 SR scale factor
OFFSET_SR          = -0.2        # Landsat C2 L2 SR offset
SCALE_FACTOR_ST    = 0.00341802  # Surface Temperature scale
OFFSET_ST          = 149.0       # Surface Temperature offset (result in Kelvin)

# --- Parallel & batch settings ---
MAX_WORKERS        = 10       # Number of parallel threads
BATCH_SIZE         = 200      # Save checkpoint every N rows
RETRY_LIMIT        = 3        # Max retries per sample on failure

# =========================================================================
# STAC CATALOG (created once, reused by all workers)
# =========================================================================
print("=" * 70)
print("EY WATER QUALITY - FULL LANDSAT EXTRACTION")
print("=" * 70)
print(f"\nConnecting to Planetary Computer STAC API...")

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=pc.sign_inplace,
)
print("Connected!\n")

print(f"Settings:")
print(f"  SR Bands:         {', '.join(SR_BANDS)}")
print(f"  ST Band:          {ST_BAND}")
print(f"  Coastal Band:     {COASTAL_BAND} (Landsat 8+ only)")
print(f"  Cloud cover:      < {MAX_CLOUD_COVER}%")
print(f"  Search window:    +/- {SEARCH_WINDOW_DAYS} days")
print(f"  Buffer:           ~100m ({BBOX_BUFFER_DEG} deg)")
print(f"  Parallel workers: {MAX_WORKERS}")
print(f"  Batch checkpoint: every {BATCH_SIZE} rows")
print(f"  Retry limit:      {RETRY_LIMIT}")


# =========================================================================
# COLUMN NAMES FOR OUTPUT
# =========================================================================
# Raw band columns
BAND_COLS = [
    "ls_blue", "ls_green", "ls_red", "ls_nir",
    "ls_swir16", "ls_swir22", "ls_coastal", "ls_st_kelvin"
]

# Derived index columns
INDEX_COLS = [
    "ls_NDVI", "ls_NDWI", "ls_NDMI", "ls_MNDWI", "ls_NDTI",
    "ls_blue_green_ratio", "ls_red_blue_ratio", "ls_nir_red_ratio",
    "ls_green_swir22_ratio", "ls_blue_red_ratio"
]

# Metadata columns
META_COLS = ["ls_platform", "ls_days_offset", "ls_cloud_cover"]

ALL_OUTPUT_COLS = BAND_COLS + INDEX_COLS + META_COLS

# NaN result template
NAN_RESULT = {col: np.nan for col in ALL_OUTPUT_COLS}
NAN_RESULT["ls_platform"] = ""


# =========================================================================
# HELPER: EXTRACT ONE SAMPLE
# =========================================================================

def extract_one_sample(lat: float, lon: float, date_str: str) -> dict:
    """
    Given a location and sample date, find the nearest cloud-free
    Landsat scene and extract ALL band values.

    Strategy:
    1. Search landsat-c2-l2 within +/-60 days of sample date
    2. Sort by: temporal closeness, then cloud cover
    3. Prefer Landsat 8 over Landsat 7 for same-day scenes
    4. Load all bands, compute median over ~100m buffer
    5. Apply scale factors to convert to physical units
    6. Compute derived indices

    Returns dict with all band values, indices, and metadata.
    """
    result = NAN_RESULT.copy()

    try:
        sample_date = pd.to_datetime(date_str, dayfirst=True, errors="coerce")
        if pd.isna(sample_date):
            return result

        # Create bounding box (~100m around point)
        bbox = [
            lon - BBOX_BUFFER_DEG / 2,
            lat - BBOX_BUFFER_DEG / 2,
            lon + BBOX_BUFFER_DEG / 2,
            lat + BBOX_BUFFER_DEG / 2,
        ]

        # Search window
        date_start = (sample_date - timedelta(days=SEARCH_WINDOW_DAYS)).strftime("%Y-%m-%d")
        date_end   = (sample_date + timedelta(days=SEARCH_WINDOW_DAYS)).strftime("%Y-%m-%d")

        # Search STAC catalog
        search = catalog.search(
            collections=["landsat-c2-l2"],
            bbox=bbox,
            datetime=f"{date_start}/{date_end}",
            query={"eo:cloud_cover": {"lt": MAX_CLOUD_COVER}},
        )

        items = list(search.items())
        if not items:
            return result

        # Sort by: (1) temporal closeness, (2) prefer L8 over L7, (3) cloud cover
        sample_date_utc = sample_date.tz_localize("UTC") if sample_date.tzinfo is None else sample_date.tz_convert("UTC")

        def sort_key(item):
            item_date = pd.to_datetime(item.properties["datetime"]).tz_convert("UTC")
            days_diff = abs((item_date - sample_date_utc).days)
            platform  = item.properties.get("platform", "")
            # Prefer Landsat 8/9 over 7 (0 = L8/L9, 1 = L7)
            platform_priority = 0 if platform in ("landsat-8", "landsat-9") else 1
            cloud = item.properties.get("eo:cloud_cover", 100)
            return (days_diff, platform_priority, cloud)

        items.sort(key=sort_key)
        best_item = items[0]

        # Get metadata
        platform = best_item.properties.get("platform", "unknown")
        cloud_cover = best_item.properties.get("eo:cloud_cover", np.nan)
        item_date = pd.to_datetime(best_item.properties["datetime"]).tz_convert("UTC")
        days_offset = (item_date - sample_date_utc).days

        result["ls_platform"] = platform
        result["ls_days_offset"] = days_offset
        result["ls_cloud_cover"] = cloud_cover

        # Sign the item for authenticated access
        signed_item = pc.sign(best_item)

        # ---- Load Surface Reflectance bands ----
        try:
            data = stac_load(
                [signed_item],
                bands=SR_BANDS,
                bbox=bbox,
            ).isel(time=0)

            # Extract median values within buffer (handles nodata/NaN)
            raw_blue   = float(data["blue"].astype("float").median(skipna=True).values)
            raw_green  = float(data["green"].astype("float").median(skipna=True).values)
            raw_red    = float(data["red"].astype("float").median(skipna=True).values)
            raw_nir    = float(data["nir08"].astype("float").median(skipna=True).values)
            raw_swir16 = float(data["swir16"].astype("float").median(skipna=True).values)
            raw_swir22 = float(data["swir22"].astype("float").median(skipna=True).values)

            # Apply scale factor: reflectance = DN * 0.0000275 - 0.2
            # Clamp to [0, 1] range (valid reflectance)
            def to_reflectance(raw):
                if raw == 0 or np.isnan(raw):
                    return np.nan
                refl = raw * SCALE_FACTOR_SR + OFFSET_SR
                return np.clip(refl, 0.0, 1.0)

            blue   = to_reflectance(raw_blue)
            green  = to_reflectance(raw_green)
            red    = to_reflectance(raw_red)
            nir    = to_reflectance(raw_nir)
            swir16 = to_reflectance(raw_swir16)
            swir22 = to_reflectance(raw_swir22)

            result["ls_blue"]   = blue
            result["ls_green"]  = green
            result["ls_red"]    = red
            result["ls_nir"]    = nir
            result["ls_swir16"] = swir16
            result["ls_swir22"] = swir22

        except Exception as e:
            # SR bands failed — leave as NaN
            blue = green = red = nir = swir16 = swir22 = np.nan

        # ---- Load Surface Temperature band ----
        try:
            st_data = stac_load(
                [signed_item],
                bands=[ST_BAND],
                bbox=bbox,
            ).isel(time=0)

            raw_st = float(st_data[ST_BAND].astype("float").median(skipna=True).values)
            if raw_st != 0 and not np.isnan(raw_st):
                # Convert DN to Kelvin: K = DN * 0.00341802 + 149.0
                kelvin = raw_st * SCALE_FACTOR_ST + OFFSET_ST
                # Sanity check: valid water temp range ~250-330 K
                if 200 < kelvin < 400:
                    result["ls_st_kelvin"] = kelvin

        except Exception:
            pass  # ST not available for this scene (L2SR-only)

        # ---- Load Coastal Aerosol band (Landsat 8+ only) ----
        try:
            if platform in ("landsat-8", "landsat-9"):
                ca_data = stac_load(
                    [signed_item],
                    bands=[COASTAL_BAND],
                    bbox=bbox,
                ).isel(time=0)

                raw_coastal = float(ca_data[COASTAL_BAND].astype("float").median(skipna=True).values)
                if raw_coastal != 0 and not np.isnan(raw_coastal):
                    coastal_refl = raw_coastal * SCALE_FACTOR_SR + OFFSET_SR
                    result["ls_coastal"] = np.clip(coastal_refl, 0.0, 1.0)

        except Exception:
            pass  # Coastal not available

        # ---- Compute derived indices ----
        eps = 1e-8  # Prevent division by zero

        # Only compute if we have valid reflectance values
        if not any(np.isnan(x) for x in [nir, red, green, blue, swir16, swir22]
                   if isinstance(x, float)):

            if not (np.isnan(nir) or np.isnan(red)):
                result["ls_NDVI"] = (nir - red) / (nir + red + eps)
                result["ls_nir_red_ratio"] = nir / (red + eps)

            if not (np.isnan(green) or np.isnan(nir)):
                result["ls_NDWI"] = (green - nir) / (green + nir + eps)

            if not (np.isnan(nir) or np.isnan(swir16)):
                result["ls_NDMI"] = (nir - swir16) / (nir + swir16 + eps)

            if not (np.isnan(green) or np.isnan(swir16)):
                result["ls_MNDWI"] = (green - swir16) / (green + swir16 + eps)

            if not (np.isnan(red) or np.isnan(green)):
                result["ls_NDTI"] = (red - green) / (red + green + eps)

            if not (np.isnan(blue) or np.isnan(green)):
                result["ls_blue_green_ratio"] = blue / (green + eps)

            if not (np.isnan(red) or np.isnan(blue)):
                result["ls_red_blue_ratio"] = red / (blue + eps)
                result["ls_blue_red_ratio"] = blue / (red + eps)

            if not (np.isnan(green) or np.isnan(swir22)):
                result["ls_green_swir22_ratio"] = green / (swir22 + eps)

        return result

    except Exception as e:
        return result


def extract_with_retry(idx: int, lat: float, lon: float, date_str: str) -> tuple:
    """
    Wrapper with retry logic and exponential backoff.
    Returns (idx, result_dict).
    """
    for attempt in range(RETRY_LIMIT):
        try:
            result = extract_one_sample(lat, lon, date_str)
            return (idx, result)
        except Exception as e:
            if attempt == RETRY_LIMIT - 1:
                return (idx, NAN_RESULT.copy())
            time.sleep(2 ** attempt)  # 1s, 2s, 4s backoff

    return (idx, NAN_RESULT.copy())


# =========================================================================
# MAIN EXTRACTION RUNNER (PARALLEL WITH BATCHING)
# =========================================================================

def run_extraction(input_csv: str, output_csv: str, split_name: str):
    """
    Main extraction loop:
    1. Load input CSV (training or validation)
    2. Check for existing checkpoint (resume support)
    3. Extract in parallel with ThreadPoolExecutor
    4. Save checkpoint every BATCH_SIZE rows
    5. Compute and save final output
    """
    print(f"\n{'='*70}")
    print(f"  EXTRACTING: {split_name.upper()}")
    print(f"{'='*70}")

    df = pd.read_csv(input_csv)
    print(f"  Loaded {len(df)} rows from {Path(input_csv).name}")

    # ---- Resume support ----
    completed_indices = set()
    existing_results = []

    if Path(output_csv).exists():
        existing_df = pd.read_csv(output_csv)
        if "_row_idx" in existing_df.columns:
            completed_indices = set(existing_df["_row_idx"].tolist())
            existing_results = existing_df.to_dict("records")
        else:
            # Assume all rows are done if no _row_idx column
            completed_indices = set(range(len(existing_df)))
            existing_results = existing_df.to_dict("records")
        print(f"  Resuming: {len(completed_indices)} rows already extracted")

    # ---- Build work list ----
    work_items = []
    for idx, row in df.iterrows():
        if idx in completed_indices:
            continue
        lat      = float(row["Latitude"])
        lon      = float(row["Longitude"])
        date_str = str(row["Sample Date"])
        work_items.append((idx, lat, lon, date_str))

    total_remaining = len(work_items)
    print(f"  Remaining to extract: {total_remaining}")

    if total_remaining == 0:
        print("  Nothing to extract — all rows already done!")
        return

    # ---- Parallel extraction with batch checkpointing ----
    new_results = []
    errors = 0
    start_time = time.time()

    pbar = tqdm(total=total_remaining, desc=f"  {split_name}", ncols=90, unit="row")

    # Process in batches for checkpointing
    for batch_start in range(0, len(work_items), BATCH_SIZE):
        batch = work_items[batch_start:batch_start + BATCH_SIZE]
        batch_results = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {}
            for (idx, lat, lon, date_str) in batch:
                future = executor.submit(extract_with_retry, idx, lat, lon, date_str)
                futures[future] = (idx, lat, lon, date_str)

            for future in as_completed(futures):
                idx_orig, lat_orig, lon_orig, date_orig = futures[future]
                try:
                    result_idx, band_vals = future.result(timeout=120)

                    # Build output row
                    row_data = {
                        "Latitude": lat_orig,
                        "Longitude": lon_orig,
                        "Sample Date": date_orig,
                        "_row_idx": result_idx,
                    }
                    row_data.update(band_vals)
                    batch_results.append(row_data)

                except Exception as e:
                    errors += 1
                    row_data = {
                        "Latitude": lat_orig,
                        "Longitude": lon_orig,
                        "Sample Date": date_orig,
                        "_row_idx": idx_orig,
                    }
                    row_data.update(NAN_RESULT.copy())
                    batch_results.append(row_data)

                pbar.update(1)

        new_results.extend(batch_results)

        # ---- Save checkpoint ----
        checkpoint_data = existing_results + new_results
        checkpoint_df = pd.DataFrame(checkpoint_data)
        checkpoint_df.to_csv(output_csv, index=False)

        elapsed = time.time() - start_time
        done_so_far = len(new_results)
        rate = done_so_far / elapsed if elapsed > 0 else 0
        remaining_est = (total_remaining - done_so_far) / rate if rate > 0 else 0
        tqdm.write(
            f"    Checkpoint: {done_so_far + len(existing_results)} rows saved | "
            f"Rate: {rate:.1f} rows/sec | "
            f"ETA: {remaining_est/60:.0f} min | "
            f"Errors: {errors}"
        )

    pbar.close()

    # ---- Final save ----
    all_results = existing_results + new_results
    final_df = pd.DataFrame(all_results)

    # Sort by original row index to maintain order
    if "_row_idx" in final_df.columns:
        final_df = final_df.sort_values("_row_idx").reset_index(drop=True)
        final_df.drop(columns=["_row_idx"], inplace=True)

    final_df.to_csv(output_csv, index=False)

    # ---- Summary stats ----
    elapsed_total = (time.time() - start_time) / 60
    band_cols = [c for c in final_df.columns if c.startswith("ls_") and c not in META_COLS]
    nan_pcts = {}
    for col in band_cols:
        if col in final_df.columns:
            nan_pcts[col] = final_df[col].isna().mean() * 100

    print(f"\n  {'─'*50}")
    print(f"  DONE: {split_name}")
    print(f"  {'─'*50}")
    print(f"  Output:       {output_csv}")
    print(f"  Total rows:   {len(final_df)}")
    print(f"  Time:         {elapsed_total:.1f} minutes")
    print(f"  Errors:       {errors}")
    print(f"\n  NaN rates per column:")
    for col, pct in sorted(nan_pcts.items()):
        print(f"    {col:30s}: {pct:5.1f}%")

    # Platform breakdown
    if "ls_platform" in final_df.columns:
        platform_counts = final_df["ls_platform"].value_counts()
        print(f"\n  Platform breakdown:")
        for plat, cnt in platform_counts.items():
            print(f"    {plat:15s}: {cnt:5d} ({cnt/len(final_df)*100:.1f}%)")

    print(f"  {'─'*50}")


# =========================================================================
# ENTRYPOINT
# =========================================================================

if __name__ == "__main__":
    train_input = DATA_DIR / "water_quality_training_dataset.csv"
    val_input   = DATA_DIR / "submission_template.csv"

    if not train_input.exists():
        print(f"ERROR: Training CSV not found: {train_input}")
        sys.exit(1)
    if not val_input.exists():
        print(f"ERROR: Validation CSV not found: {val_input}")
        sys.exit(1)

    print(f"\n  Input files:")
    print(f"    Training:   {train_input} ({pd.read_csv(train_input).shape[0]} rows)")
    print(f"    Validation: {val_input} ({pd.read_csv(val_input).shape[0]} rows)")
    print(f"\n  Output files:")
    print(f"    Training:   {TRAIN_OUT}")
    print(f"    Validation: {VAL_OUT}")

    print(f"\n  Bands being extracted:")
    print(f"    Surface Reflectance: {', '.join(SR_BANDS)}")
    print(f"    Surface Temperature: {ST_BAND}")
    print(f"    Coastal Aerosol:     {COASTAL_BAND} (Landsat 8+ only)")
    print(f"\n  Derived indices:")
    print(f"    {', '.join(INDEX_COLS)}")

    print(f"\n  Parallel extraction: {MAX_WORKERS} workers")
    print(f"  Checkpoint every {BATCH_SIZE} rows — safe to Ctrl+C and resume!")

    # ---- Extract training data (9319 rows) ----
    t_start = time.time()
    run_extraction(str(train_input), str(TRAIN_OUT), split_name="training")

    # ---- Extract validation data (200 rows) ----
    run_extraction(str(val_input), str(VAL_OUT), split_name="validation")

    # ---- Final summary ----
    total_elapsed = (time.time() - t_start) / 3600
    print(f"\n{'='*70}")
    print(f"  ALL DONE in {total_elapsed:.1f} hours")
    print(f"  Training  -> {TRAIN_OUT}")
    print(f"  Validation -> {VAL_OUT}")
    print(f"{'='*70}")
    print(f"\n  Next step: Run the model script that uses these features!")
    print(f"  The new columns will be prefixed with 'ls_' to distinguish")
    print(f"  from the original EY-provided Landsat data.")

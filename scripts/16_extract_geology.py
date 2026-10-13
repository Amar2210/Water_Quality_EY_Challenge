"""
============================================================
EY Water Quality Challenge - Geology / Lithology Extraction
============================================================
Extracts geological/lithological information for each river location.

APPROACH: Multi-source extraction with fallbacks
  1. Macrostrat API (macrostrat.org) — global geological map, REST API
  2. If Macrostrat fails, derive geology proxies from existing soil data

WHY GEOLOGY MATTERS:
  - Bedrock type is THE primary natural driver of water mineral content
  - Limestone/dolomite → high alkalinity, moderate-high EC
  - Granite/gneiss → low alkalinity, low EC
  - Sandstone → moderate alkalinity, variable EC
  - Shale/mudstone → moderate-high alkalinity, higher EC
  - South Africa has extremely diverse geology (Karoo Basin, Bushveld Complex,
    Cape Fold Belt, Witwatersrand Basin)
  - Without geology, the model has no way to know WHY one river has
    EC=800 (limestone area) and another has EC=100 (granite area)

MACROSTRAT API:
  GET https://macrostrat.org/api/v2/geologic_units/map?lat=X&lng=Y&response=long
  Returns: lithology, rock type, age, formation name

OUTPUT: data/extracted/geology_data.csv
  Columns: Latitude, Longitude, lith_type, lith_class, rock_type,
           geo_age_top, geo_age_bottom, is_sedimentary, is_igneous, etc.

ESTIMATED TIME: 5-15 minutes (186 API calls, with rate limiting)
============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'ey_challenge_notebooks')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'data', 'extracted')
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'geology_data.csv')
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, 'geology_checkpoint.csv')

# Macrostrat API endpoint
MACROSTRAT_URL = "https://macrostrat.org/api/v2/geologic_units/map"

# Rate limiting: be nice to the API
REQUEST_DELAY = 0.5  # seconds between requests
MAX_RETRIES = 3
MAX_WORKERS = 4  # keep low to avoid overwhelming API

# Lithology categories relevant to water chemistry
# Maps raw lithology strings to simplified categories
LITH_CATEGORIES = {
    # Carbonate rocks → HIGH alkalinity, HIGH EC
    'limestone': 'carbonate', 'dolomite': 'carbonate', 'dolostone': 'carbonate',
    'marble': 'carbonate', 'chalk': 'carbonate', 'travertine': 'carbonate',
    'carbonate': 'carbonate',
    
    # Siliciclastic sedimentary → MODERATE alkalinity
    'sandstone': 'siliciclastic', 'siltstone': 'siliciclastic',
    'conglomerate': 'siliciclastic', 'breccia': 'siliciclastic',
    'quartzite': 'siliciclastic', 'arkose': 'siliciclastic',
    
    # Fine-grained sedimentary → MODERATE-HIGH alkalinity, HIGHER EC
    'shale': 'fine_sedimentary', 'mudstone': 'fine_sedimentary',
    'claystone': 'fine_sedimentary', 'slate': 'fine_sedimentary',
    'phyllite': 'fine_sedimentary', 'marl': 'fine_sedimentary',
    
    # Igneous felsic → LOW alkalinity, LOW EC
    'granite': 'felsic_igneous', 'rhyolite': 'felsic_igneous',
    'granodiorite': 'felsic_igneous', 'syenite': 'felsic_igneous',
    'pegmatite': 'felsic_igneous', 'aplite': 'felsic_igneous',
    'gneiss': 'felsic_igneous',  # metamorphic but usually felsic origin
    
    # Igneous mafic → MODERATE alkalinity (more minerals)
    'basalt': 'mafic_igneous', 'gabbro': 'mafic_igneous',
    'dolerite': 'mafic_igneous', 'diabase': 'mafic_igneous',
    'andesite': 'mafic_igneous', 'diorite': 'mafic_igneous',
    'norite': 'mafic_igneous', 'pyroxenite': 'mafic_igneous',
    
    # Ultramafic → MODERATE EC, high magnesium
    'peridotite': 'ultramafic', 'serpentinite': 'ultramafic',
    'dunite': 'ultramafic', 'chromitite': 'ultramafic',
    
    # Evaporites → VERY HIGH EC (dissolved salts)
    'evaporite': 'evaporite', 'halite': 'evaporite', 'gypsum': 'evaporite',
    'anhydrite': 'evaporite', 'salt': 'evaporite',
    
    # Volcanic → VARIABLE
    'tuff': 'volcanic', 'volcanic': 'volcanic', 'lava': 'volcanic',
    'ignimbrite': 'volcanic', 'pyroclastic': 'volcanic',
    
    # Metamorphic general
    'schist': 'metamorphic', 'amphibolite': 'metamorphic',
    'hornfels': 'metamorphic', 'migmatite': 'metamorphic',
    'granulite': 'metamorphic',
    
    # Unconsolidated (alluvium, soil, sand)
    'alluvium': 'unconsolidated', 'sand': 'unconsolidated',
    'gravel': 'unconsolidated', 'clay': 'unconsolidated',
    'till': 'unconsolidated', 'loess': 'unconsolidated',
    'colluvium': 'unconsolidated', 'laterite': 'unconsolidated',
}

# EC impact ranking for lithology categories (1=low EC, 5=very high EC)
EC_IMPACT = {
    'felsic_igneous': 1,
    'siliciclastic': 2,
    'metamorphic': 2,
    'volcanic': 2,
    'unconsolidated': 3,
    'mafic_igneous': 3,
    'ultramafic': 3,
    'fine_sedimentary': 3,
    'carbonate': 4,
    'evaporite': 5,
}

# Alkalinity impact ranking
ALK_IMPACT = {
    'felsic_igneous': 1,
    'volcanic': 2,
    'siliciclastic': 2,
    'unconsolidated': 2,
    'metamorphic': 2,
    'mafic_igneous': 3,
    'ultramafic': 3,
    'fine_sedimentary': 3,
    'carbonate': 5,
    'evaporite': 4,
}


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


def classify_lithology(lith_string):
    """Map a raw lithology string to a simplified category."""
    if not lith_string or pd.isna(lith_string):
        return 'unknown'
    
    lith_lower = lith_string.lower().strip()
    
    # Direct match
    if lith_lower in LITH_CATEGORIES:
        return LITH_CATEGORIES[lith_lower]
    
    # Partial match (check if any key is contained in the string)
    for key, category in LITH_CATEGORIES.items():
        if key in lith_lower:
            return category
    
    return 'unknown'


def query_macrostrat(lat, lon):
    """
    Query Macrostrat API for geological information at a lat/lon point.
    Returns dict with geological features.
    """
    result = {
        'Latitude': lat, 'Longitude': lon,
        'geo_source': 'macrostrat',
        'geo_lith_raw': None,
        'geo_lith_category': 'unknown',
        'geo_rock_type': None,
        'geo_age_top_ma': None,
        'geo_age_bottom_ma': None,
        'geo_formation': None,
        'geo_ec_impact': 3,  # default moderate
        'geo_alk_impact': 2,  # default moderate
    }
    
    try:
        params = {
            'lat': lat,
            'lng': lon,
            'response': 'long',
        }
        
        resp = requests.get(MACROSTRAT_URL, params=params, timeout=15)
        
        if resp.status_code != 200:
            result['geo_source'] = f'macrostrat_error_{resp.status_code}'
            return result
        
        data = resp.json()
        
        if not data.get('success') or not data.get('success', {}).get('data', []):
            # Try the different response structure
            features = data.get('success', {}).get('data', {}).get('mapData', [])
            if not features:
                features = data.get('success', {}).get('data', [])
            if not features:
                result['geo_source'] = 'macrostrat_no_data'
                return result
        else:
            features = data['success']['data']
        
        if not features:
            result['geo_source'] = 'macrostrat_empty'
            return result
        
        # Take the first (top/most recent) geological unit
        unit = features[0] if isinstance(features, list) else features
        
        # Extract lithology
        liths = unit.get('liths', [])
        if liths:
            if isinstance(liths[0], dict):
                primary_lith = liths[0].get('name', liths[0].get('lith', ''))
            else:
                primary_lith = str(liths[0])
        else:
            primary_lith = unit.get('lith', unit.get('lithology', ''))
        
        result['geo_lith_raw'] = primary_lith
        result['geo_lith_category'] = classify_lithology(primary_lith)
        
        # Rock type (sedimentary, igneous, metamorphic)
        result['geo_rock_type'] = unit.get('lith_type', unit.get('type', None))
        
        # Geological age (in millions of years ago)
        result['geo_age_top_ma'] = unit.get('t_age', unit.get('age_top', None))
        result['geo_age_bottom_ma'] = unit.get('b_age', unit.get('age_bottom', None))
        
        # Formation name
        result['geo_formation'] = unit.get('strat_name', unit.get('name', None))
        
        # EC and alkalinity impact scores
        cat = result['geo_lith_category']
        result['geo_ec_impact'] = EC_IMPACT.get(cat, 3)
        result['geo_alk_impact'] = ALK_IMPACT.get(cat, 2)
        
    except requests.exceptions.Timeout:
        result['geo_source'] = 'macrostrat_timeout'
    except requests.exceptions.ConnectionError:
        result['geo_source'] = 'macrostrat_connection_error'
    except Exception as e:
        result['geo_source'] = f'macrostrat_error: {str(e)[:100]}'
    
    return result


def query_with_retry(lat, lon, max_retries=MAX_RETRIES):
    """Query Macrostrat with retry and backoff."""
    for attempt in range(max_retries):
        result = query_macrostrat(lat, lon)
        
        if 'error' not in result.get('geo_source', '').lower() and \
           'timeout' not in result.get('geo_source', '').lower():
            return result
        
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)
    
    return result  # Return last attempt even if it failed


def derive_soil_geology_proxies(locations_df, soil_data_path):
    """
    Derive geology proxy features from existing soil data.
    Fallback when Macrostrat has no coverage.
    
    Rationale:
    - High pH + high CEC + high clay → likely carbonate/calcareous bedrock
    - Low pH + low CEC + high sand → likely felsic igneous/siliceous bedrock  
    - High SOC + moderate pH → organic-rich sedimentary deposits
    """
    if not os.path.exists(soil_data_path):
        print(f"  Soil data not found at {soil_data_path}")
        return locations_df
    
    soil = pd.read_csv(soil_data_path)
    
    # Merge soil data with locations
    merged = locations_df.merge(soil, on=['Latitude', 'Longitude'], how='left')
    
    if 'soil_phh2o' in merged.columns:
        # Geology proxy from soil pH
        # High pH (>7.5) suggests carbonate influence
        # Low pH (<5.5) suggests felsic/acidic bedrock
        merged['geo_proxy_ph_class'] = pd.cut(
            merged['soil_phh2o'],
            bins=[0, 5.5, 6.5, 7.5, 14],
            labels=['acidic', 'neutral_low', 'neutral_high', 'alkaline']
        ).astype(str)
    
    if 'soil_cec' in merged.columns and 'soil_clay' in merged.columns:
        # CEC per unit clay — indicates clay mineralogy type
        # High CEC/clay = smectite (volcanic/mafic origin)
        # Low CEC/clay = kaolinite (felsic/highly weathered)
        merged['geo_proxy_cec_per_clay'] = merged['soil_cec'] / merged['soil_clay'].clip(lower=1)
    
    if 'soil_sand' in merged.columns and 'soil_phh2o' in merged.columns:
        # Sandy + acidic = likely felsic igneous
        # Clayey + alkaline = likely carbonate/mafic
        merged['geo_proxy_mineral_index'] = (
            merged['soil_phh2o'].fillna(7) * merged['soil_clay'].fillna(30) / 
            merged['soil_sand'].fillna(30).clip(lower=1)
        )
    
    return merged


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print(f"""
{'='*70}
GEOLOGY / LITHOLOGY EXTRACTION
{'='*70}
Source: Macrostrat API + Soil-derived proxies
{'='*70}
""")
    
    # Load locations
    print("[1] Loading locations...")
    locations = load_unique_locations()
    
    # Check for checkpoint
    done_locs = set()
    existing_results = []
    if os.path.exists(CHECKPOINT_FILE):
        checkpoint = pd.read_csv(CHECKPOINT_FILE)
        for _, row in checkpoint.iterrows():
            done_locs.add((round(row['Latitude'], 6), round(row['Longitude'], 6)))
            existing_results.append(row.to_dict())
        print(f"  Resuming: {len(done_locs)} locations already done")
    
    remaining = []
    for _, row in locations.iterrows():
        key = (round(row['Latitude'], 6), round(row['Longitude'], 6))
        if key not in done_locs:
            remaining.append((row['Latitude'], row['Longitude']))
    
    print(f"  Remaining: {len(remaining)} locations")
    
    # Query Macrostrat API
    print(f"\n[2] Querying Macrostrat API ({len(remaining)} locations)...")
    print(f"  Rate limit: {REQUEST_DELAY}s between requests")
    print(f"  Estimated time: {len(remaining) * REQUEST_DELAY / 60:.1f} minutes")
    
    all_results = list(existing_results)
    
    # Sequential with progress bar (API rate limiting makes parallelism risky)
    for i, (lat, lon) in enumerate(tqdm(remaining, desc="  Extracting")):
        result = query_with_retry(lat, lon)
        all_results.append(result)
        
        # Rate limiting
        time.sleep(REQUEST_DELAY)
        
        # Checkpoint every 50
        if (i + 1) % 50 == 0:
            pd.DataFrame(all_results).to_csv(CHECKPOINT_FILE, index=False)
    
    # Convert to DataFrame
    geo_df = pd.DataFrame(all_results)
    
    # Summary of Macrostrat coverage
    print(f"\n[3] Macrostrat API Results:")
    sources = geo_df['geo_source'].value_counts()
    for source, count in sources.items():
        print(f"  {source}: {count} locations")
    
    if 'geo_lith_category' in geo_df.columns:
        print(f"\n  Lithology category distribution:")
        cats = geo_df['geo_lith_category'].value_counts()
        for cat, count in cats.items():
            ec_imp = EC_IMPACT.get(cat, '?')
            alk_imp = ALK_IMPACT.get(cat, '?')
            print(f"    {cat:25s}: {count:3d} locations  (EC impact={ec_imp}, ALK impact={alk_imp})")
    
    # Add soil-derived geology proxies as fallback
    print(f"\n[4] Adding soil-derived geology proxies...")
    soil_path = os.path.join(OUTPUT_DIR, 'soil_data.csv')
    geo_df = derive_soil_geology_proxies(geo_df, soil_path)
    
    # One-hot encode lithology categories for ML
    print(f"\n[5] Encoding geology features for ML...")
    if 'geo_lith_category' in geo_df.columns:
        # One-hot encoding of lithology categories
        for cat in set(LITH_CATEGORIES.values()):
            geo_df[f'geo_is_{cat}'] = (geo_df['geo_lith_category'] == cat).astype(int)
    
    # Save
    print(f"\n[6] Saving results...")
    geo_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print(f"  Saved: {OUTPUT_FILE}")
    print(f"  Columns: {list(geo_df.columns)}")
    
    # Clean up checkpoint
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
    
    print(f"\n{'='*70}")
    print(f"GEOLOGY EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"  Locations: {len(geo_df)}")
    print(f"  Macrostrat coverage: {(geo_df['geo_source'] == 'macrostrat').sum()}/{len(geo_df)}")
    print(f"\n  Next: Check lithology coverage, then run scripts/18_feature_engineering_v8.py")

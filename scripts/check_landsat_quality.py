"""Quick quality check on the full Landsat extraction."""
import pandas as pd
import numpy as np

DATA = r"C:\Users\amarbir\self_learning\ey_challenge\data\extracted"

for split, fname in [("TRAINING", "landsat_full_training.csv"), ("VALIDATION", "landsat_full_validation.csv")]:
    df = pd.read_csv(f"{DATA}/{fname}")
    print(f"\n{'='*60}")
    print(f"  {split}: {fname}")
    print(f"{'='*60}")
    print(f"  Shape: {df.shape}")
    
    # NaN rates
    print(f"\n  NaN rates:")
    band_cols = [c for c in df.columns if c.startswith("ls_") and c not in ["ls_platform", "ls_days_offset", "ls_cloud_cover"]]
    for col in band_cols:
        nan_pct = df[col].isna().mean() * 100
        valid = df[col].dropna()
        if len(valid) > 0:
            print(f"    {col:30s}: {nan_pct:5.1f}% NaN | mean={valid.mean():.4f} | min={valid.min():.4f} | max={valid.max():.4f}")
        else:
            print(f"    {col:30s}: {nan_pct:5.1f}% NaN | ALL EMPTY")
    
    # Platform breakdown
    if "ls_platform" in df.columns:
        print(f"\n  Platform breakdown:")
        for plat, cnt in df["ls_platform"].value_counts().items():
            print(f"    {plat:15s}: {cnt:5d} ({cnt/len(df)*100:.1f}%)")
    
    # Days offset stats
    if "ls_days_offset" in df.columns:
        offsets = df["ls_days_offset"].dropna()
        print(f"\n  Days offset from sample date:")
        print(f"    Mean: {offsets.mean():.1f} | Median: {offsets.median():.1f} | Abs mean: {offsets.abs().mean():.1f}")
    
    # Compare with EY's original coverage
    all_sr_nan = df[["ls_blue", "ls_green", "ls_red", "ls_nir", "ls_swir16", "ls_swir22"]].isna().all(axis=1)
    print(f"\n  Rows with ALL SR bands = NaN: {all_sr_nan.sum()} ({all_sr_nan.mean()*100:.1f}%)")
    print(f"  Rows with at least some SR data: {(~all_sr_nan).sum()} ({(~all_sr_nan).mean()*100:.1f}%)")

# Compare NaN rates with original EY data
print(f"\n\n{'='*60}")
print(f"  COMPARISON: Our extraction vs EY's original")
print(f"{'='*60}")
ey_train = pd.read_csv(r"C:\Users\amarbir\self_learning\ey_challenge\Jupyter Notebook Package\landsat_features_training.csv")
our_train = pd.read_csv(f"{DATA}/landsat_full_training.csv")

ey_nan = ey_train["nir"].isna().mean() * 100
our_nan = our_train["ls_nir"].isna().mean() * 100
print(f"  EY original - training NaN rate (nir):  {ey_nan:.1f}%")
print(f"  Our extract - training NaN rate (ls_nir): {our_nan:.1f}%")
print(f"  EY rows: {len(ey_train)} | Our rows: {len(our_train)}")

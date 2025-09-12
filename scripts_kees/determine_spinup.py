#!/usr/bin/env python3
"""
Compare ground_temperature_distribution across spinup runs and plot RMSE, Bias, Scatter Index.
Save as compare_spinup_runs.py and run with Python.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import logging

# suppress logger INFO/DEBUG messages
# FOUND 3 YEARS
logging.getLogger().setLevel(logging.WARNING)

# Try imports
try:
    from netCDF4 import Dataset
except Exception:
    Dataset = None
    try:
        import xarray as xr
    except Exception:
        xr = None

RUN_ROOT = Path(r'd:\Git\thermo-morphological-model\runs\20250828_determine_spinup\20250825_spinup_batch')

def load_ground_temp(nc_path):
    """Load variable 'ground_temperature_distribution' from results.nc"""
    if Dataset is not None:
        with Dataset(str(nc_path), 'r') as nc:
            for name in ('ground_temperature_distribution', 'ground_temperature'):
                if name in nc.variables:
                    return np.array(nc.variables[name][:])
            raise KeyError(f"'ground_temperature_distribution' not found in {nc_path}")
    elif 'xr' in globals() and xr is not None:
        ds = xr.open_dataset(str(nc_path))
        for name in ('ground_temperature_distribution', 'ground_temperature'):
            if name in ds:
                return ds[name].values
        raise KeyError(f"'ground_temperature_distribution' not found in {nc_path}")
    else:
        raise ImportError("Either netCDF4 or xarray is required to read results.nc")

def trim_to_common_shape(a, b):
    """Trim two arrays to the smallest common shape along each axis."""
    if a.ndim != b.ndim:
        # try to align by adding missing dims at front if one has extra length-1 dims
        min_ndim = min(a.ndim, b.ndim)
        a = a.reshape(a.shape[:min_ndim])
        b = b.reshape(b.shape[:min_ndim])
    common_shape = tuple(min(sa, sb) for sa, sb in zip(a.shape, b.shape))
    slices = tuple(slice(0, n) for n in common_shape)
    return a[slices], b[slices]

def compute_metrics(ref, other):
    """Compute RMSE, Bias, Scatter Index (%) between arrays (same shape)."""
    # ensure same shape: trim if necessary
    if ref.shape != other.shape:
        ref, other = trim_to_common_shape(ref, other)
    diff = other - ref
    rmse = np.sqrt(np.nanmean(diff**2))
    bias = np.nanmean(diff)
    mean_ref = np.nanmean(np.abs(ref))
    si = np.nan if mean_ref == 0 or np.isnan(mean_ref) else (rmse / mean_ref) * 100.0
    return rmse, bias, si

def main():
    runs = [RUN_ROOT / f'run{idx:02d}' for idx in range(1, 11)]
    data = {}
    for r in runs:
        nc_path = r / 'results' / 'results.nc'
        if not nc_path.exists():
            print(f"Warning: {nc_path} missing, skipping")
            data[r.name] = None
            continue
        try:
            arr = load_ground_temp(nc_path)
        except Exception as e:
            print(f"Error loading {nc_path}: {e}")
            data[r.name] = None
            continue
        data[r.name] = arr
        print(f"Loaded {r.name}: shape {arr.shape}")

    ref = data.get('run10')
    if ref is None:
        print("Reference run10 not available. Aborting.")
        return

    labels, rmses, biases, sis = [], [], [], []
    for idx in range(1, 10):
        name = f'run{idx:02d}'
        arr = data.get(name)
        labels.append(name)
        if arr is None:
            rmses.append(np.nan); biases.append(np.nan); sis.append(np.nan)
            continue
        try:
            rmse, bias, si = compute_metrics(ref, arr)
        except Exception as e:
            print(f"Error computing metrics for {name}: {e}")
            rmse, bias, si = np.nan, np.nan, np.nan
        rmses.append(rmse); biases.append(bias); sis.append(si)
        print(f"{name} vs run10 -> RMSE: {rmse:.6f}, Bias: {bias:.6f}, SI: {si:.3f}%")

    # Plot grouped bar chart
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, rmses, width, label='RMSE')
    ax.bar(x, biases, width, label='Bias')
    ax.bar(x + width, sis, width, label='Scatter Index (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Metric value')
    ax.set_title('Comparison vs run10: ground_temperature_distribution')
    ax.legend()
    fig.tight_layout()
    out_path = RUN_ROOT / 'compare_ground_temperature_metrics.png'
    fig.savefig(out_path, dpi=200)
    print(f"Saved figure to {out_path}")

if __name__ == '__main__':
    main()
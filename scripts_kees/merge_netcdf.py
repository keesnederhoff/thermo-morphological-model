from pathlib import Path
import re
import pandas as pd
import xarray as xr

# Define directory
root = Path(r"d:\Git\thermo-morphological-model\runs\20250822_validation_runs\run001_val_gt22\results")
pat = re.compile(r"^(\d+)\.nc$")

# Find files
files = [p for p in root.glob("*.nc") if pat.match(p.name)]
if not files:
    raise SystemExit("No timestep NetCDF files found.")
files = sorted(files, key=lambda p: int(pat.search(p.name).group(1)))

# Read with xarray
ds = xr.open_mfdataset([str(f) for f in files],
    combine="by_coords",
    data_vars="minimal",
    coords="minimal",
    compat="override",
    combine_attrs="drop_conflicts",
    engine="h5netcdf",
    parallel=False,
    chunks={"time": 64},  # tweak if needed
)

# Write out
out = root / "merged.nc"
ds.to_netcdf(out, engine="netcdf4", format="NETCDF4", encoding=encoding)
ds.close()
print(f"Merged file written: {out}")
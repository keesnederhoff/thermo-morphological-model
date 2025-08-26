import os, xarray as xr
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
ds = xr.open_dataset(r'd:\Git\thermo-morphological-model\runs\20250822_validation_runs\run001_val_gt22_v2\results\results.nc', engine='netcdf4', lock=False)
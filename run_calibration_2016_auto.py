# Automatic calibration Artic XBeach
import shutil
import random
import multiprocessing
from pathlib import Path
import sys
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from main import main, Simulation
from netCDF4 import Dataset
import netCDF4
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings(
    "ignore",
    message="'squared' is deprecated in version 1.4 and will be removed in 1.6. To calculate the root mean squared error, use the function'root_mean_squared_error'."
)

# Parameter ranges
param_ranges = {
    "max_depth": [10, 30],                              # 15m kind of in the middle
    "T_melt": [270.4185, 275.8815],                     # a bit large of a range
    "N_thaw_threshold": [2, 40],                        # seems ok, shouldnt have a influence i think
    "L_water_ice": [267200, 400800],                    # might be to high (330000-336000 range chatGPT)
    "rho_water": [800, 1200],                           # might be too high (1000-1030 range chatGPT)
    "rho_ice": [917*0.9, 917*1.1],                      # should be 917 (not 971 as Kevin used)
    "rho_particle": [2120, 3180],                       # seems to high (2400-2900)
    "nb_min": [0.25, 0.78],                             # 0.30-0.55
    "nb_max": [0.25, 0.90],                             # 0.45-0.85
    "c_soil_frozen": [3680000, 5520000],                # 2.0e6 to 3.5e5
    "c_soil_unfrozen": [5600000, 8400000],              # 3.0e6 to 4.5e6
    "k_soil_frozen_min": [0.7, 3.24],                   # 1-2
    "k_soil_frozen_max": [0.7, 3.7],                    # 2-3.5
    "k_soil_unfrozen_min": [0.3, 1.6],                  # 0.3-1
    "k_soil_unfrozen_max": [0.48, 1.5],                 # 1-2
    "geothermal_gradient": [0.02, 0.03]                 # seems good
}

# Settings
base_sim_dir    = Path(r'd:\Git\thermo-morphological-model\runs\20250822_calibration_runs\run003_iterations\base')
run_root_dir    = Path(r'd:\Git\thermo-morphological-model\runs\20250822_calibration_runs\run003_iterations')
n_parallel      = 32
n_epochs        = 3
make_figure     = True

# Sample parameters
def sample_params():
    params = {}
    for k, v in param_ranges.items():
        if v[0] == v[1]:
            params[k] = v[0]
        elif isinstance(v[0], int) and isinstance(v[1], int):
            params[k] = random.randint(v[0], v[1])
        else:
            params[k] = random.uniform(v[0], v[1])
    
    # Overwrite some of them
    params["grid_resolution"] = params["max_depth"] * 10
    
    # Done
    return params

# Update yaml
def update_config_yaml(sim_dir, params):
    config_path = sim_dir / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    for k, v in params.items():
        if 'thermal' in config and k in config['thermal']:
            config['thermal'][k] = v
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

# Compute metrics
def compute_metrics(model_time, model_temp, obs_time, obs_temp):
    model_time          = pd.to_datetime(model_time)
    obs_time            = pd.to_datetime(obs_time)
    df_model_temp       = pd.DataFrame({'time': model_time, 'temp': model_temp}).sort_values('time')
    df_obs_temp         = pd.DataFrame({'time': obs_time, 'temp': obs_temp}).sort_values('time')
    merged              = pd.merge_asof(df_obs_temp, df_model_temp, on='time', direction='nearest', suffixes=('_obs', '_model'))
    valid               = ~pd.isnull(merged['temp_obs']) & ~pd.isnull(merged['temp_model'])
    obs_vals            = merged.loc[valid, 'temp_obs']
    model_vals          = merged.loc[valid, 'temp_model']
    rmse                = mean_squared_error(obs_vals, model_vals, squared=False)
    mae                 = mean_absolute_error(obs_vals, model_vals)
    bias                = (model_vals - obs_vals).mean()
    return rmse, mae, bias

import datetime
def to_native_datetime(dt):
    if hasattr(dt, 'to_datetime'):
        return dt.to_datetime()
    elif isinstance(dt, (list, np.ndarray)):
        return [to_native_datetime(d) for d in dt]
    elif hasattr(dt, 'year'):
        # cftime objects have year/month/day/hour/minute/second attributes
        return datetime.datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
    else:
        return dt

# Run simulation and compute skill
def run_simulation(sim_dir):
    proj_dir = Path(__file__).parent.resolve()
    sim = Simulation(str(sim_dir), proj_dir=proj_dir)
    main(sim)
    try:

        # Load observed data
        df_val = pd.read_csv(
            r'd:\Git\thermo-morphological-model\database\raw_datasets\erikson2\Temp_arrays USGS-UCSC_Oberle\BI_T-1_processed.csv',
            parse_dates=['date_time']
        )

        # Load model output from NetCDF
        nc_path = sim_dir / 'results' / 'results.nc'
        with Dataset(nc_path, 'r') as nc:
            time = nc.variables['time'][:]
            time_units = nc.variables['time'].units
            time_dt = netCDF4.num2date(time, units=time_units)
            time_py = np.array([to_native_datetime(t) for t in time_dt])
            time_pd = pd.to_datetime(time_py)
            ground_temp = nc.variables['ground_temperature_distribution'][:]  # shape: (nt, nz)
            depth_id = nc.variables['depth_id'][:]  # shape: (nz,)
            abs_zgr = nc.variables['abs_zgr'][:]  # shape: (1, 2, nz)
            abs_xgr = nc.variables['abs_xgr'][:]  # shape: (1, 2, nz)
            abs_zgr = abs_zgr[0, 1, :]  # squeeze as in MATLAB
            abs_xgr = abs_xgr[0, 1, :]  # squeeze as in MATLAB

        # Make this depth (insteaed of zb where 15)
        depth_model = (abs_zgr - abs_zgr[0])*-1

        # Depths and column mapping
        depth_map = {
            'dpth0cm': 0.0,
            'dpth25cm': 0.25,
            'dpth50cm': 0.5,
            'dpth100cm': 1.0,
            'dpth125cm': 1.25,
            'dpth150cm': 1.5,
            'dpth175cm': 1.75,
            'dpth200cm': 2.0,
            'dpth225cm': 2.25,
            'dpth250cm': 2.5,
            'dpth275cm': 2.75,
            'dpth300cm': 3.0,
            'dpth325cm': 3.25,
            'dpth350cm': 3.5,
            'dpth375cm': 3.75,
            'dpth400cm': 4.0,
            'dpth425cm': 4.25
        }

        rmse_list = []
        for col, depth_val in depth_map.items():
            
            # Find closest depth index in model output
            idx             = (np.abs(depth_model - depth_val)).argmin()
            model_temp      = ground_temp[:, 1, idx] - 273.15  # Convert K to C and take middle
            obs_temp        = df_val[col]
            rmse, mae, bias = compute_metrics(time_pd, model_temp, df_val['date_time'], obs_temp)
            rmse_list.append(rmse)

        # Combine numbers => simple compute mean
        rmse    = np.mean(rmse_list)

        # Another possiblity (more weight at top)
        weights = np.linspace(1, 0.1, len(rmse_list))  # Example: linearly decreasing weights
        rmse    = np.average(rmse_list, weights=weights)

        # Also make nice figure
        if make_figure is True:

            # Compute skill for the depths
            selected_depths_cm = [50, 100, 200, 300]
            selected_cols = [f'dpth{d}cm' for d in selected_depths_cm]
            depth_labels = [f"{d} cm depth" for d in selected_depths_cm]
            colors = ['C0', 'C1', 'C2', 'C3']

            # Compute metrics for selected depths
            rmse_list = []
            metrics_found = {}
            model_temps = {}

            for col in selected_cols:

                # Find observed val
                depth_val       = depth_map[col]

                # Find closest depth index in model output
                idx             = (np.abs(depth_model - depth_val)).argmin()
                model_temp      = ground_temp[:, 1, idx] - 273.15  # Convert K to C and take middle
                obs_temp        = df_val[col]
                rmse, mae, bias = compute_metrics(time_pd, model_temp, df_val['date_time'], obs_temp)
                rmse_list.append(rmse)
                metrics_found[col] = {'RMSE': rmse, 'MAE': mae, 'Bias': bias}
                model_temps[col] = model_temp

            # Make actual figure
            fig, axs = plt.subplots(len(selected_depths_cm), 1, figsize=(18, 12), sharex=True)
            for ax, depth, col, color, label in zip(
                axs, selected_depths_cm, selected_cols, colors, depth_labels
            ):
                ax.set_title(label)
                ax.plot(time_pd, model_temps[col], color, label='Modelled')
                ax.plot(df_val['date_time'], df_val[col], 'k', label='Measured')

                m = metrics_found[col]
                skill_text = (
                    f"RMSE = {m['RMSE']:.2f} °C\n"
                    f"MAE = {m['MAE']:.2f} °C\n"
                    f"Bias = {m['Bias']:.2f} °C"
                )
                ax.text(
                    0.95, 0.05, skill_text,
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='bottom',
                    horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
                )

                ax.set_xlim((pd.to_datetime("2016-05-15"), pd.to_datetime("2016-09-30")))
                ax.set_ylim((-10, 5))
                ax.legend(loc='upper left')
                ax.set_ylabel('Temperature [C]')
                ax.grid()

            axs[-1].set_xlabel('Time')
            fig.suptitle("Temperature at different soil layers (modelled and measured, for calibration dataset)")
            fig.tight_layout()
            fig.savefig(sim_dir / "temperature_layers.png", dpi=300)
            plt.close()

    except Exception as e:
        rmse = float('inf')
    return rmse

# Define worker
def worker(args):
    idx, params = args
    sim_dir = run_root_dir / f'run_{idx:04d}'
    if sim_dir.exists():
        shutil.rmtree(sim_dir)
    shutil.copytree(base_sim_dir, sim_dir)
    update_config_yaml(sim_dir, params)
    rmse = run_simulation(sim_dir)
    return (params, rmse)

# Main
if __name__ == "__main__":
    best_params = None
    best_rmse = float('inf')
    for epoch in range(n_epochs):
        param_list = [sample_params() for _ in range(n_parallel)]
        with multiprocessing.Pool(n_parallel) as pool:
            results = pool.map(worker, list(enumerate(param_list)))
        for idx, (params, rmse) in enumerate(results):
            #print(f"Params: {params}, RMSE: {rmse}")
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params
                best_sim_idx = idx
            print(f"Epoch {epoch+1}/{n_epochs} best RMSE: {best_rmse} (simulation #{best_sim_idx})")
    print("Best parameters found:", best_params)
    print("Best simulation number:", best_sim_idx)
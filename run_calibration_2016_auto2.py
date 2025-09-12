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
import optuna
import gc  # Import garbage collection module
import ray

warnings.filterwarnings(
    "ignore",
    message="'squared' is deprecated in version 1.4 and will be removed in 1.6. To calculate the root mean squared error, use the function'root_mean_squared_error'."
)
import os

# Ensure child processes use 'spawn' start method early (Windows-safe)
try:
    multiprocessing.set_start_method("spawn", force=True)
except Exception:
    # if
    #  already set or unsupported, ignore
    pass

# Parameter ranges
param_ranges = {
    "max_depth": [10, 30],                              # 15m kind of in the middle
    "T_melt": [270.4185, 275.8815],                     # a bit large of a range
    "L_water_ice": [267200, 400800],                    # might be to high (330000-336000 range chatGPT)
    "rho_water": [1000, 1030],                          # might be too high (1000-1030 range chatGPT)
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
base_sim_dir    = Path(r'd:\Git\thermo-morphological-model\runs\20250822_calibration_runs\run006_iterations_automated\base')
run_root_dir    = Path(r'd:\Git\thermo-morphological-model\runs\20250822_calibration_runs\run006_iterations_automated')
n_parallel      = 1
n_epochs        = 48        # maybe try 300 later?
make_figure     = True

# 24 hours => 48 epochs since 30 minute per epoch
# 48x16 

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
    main(sim, print_to_screen=False)
    nc_path = None
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
    finally:
        # Attempt to remove the NetCDF file after processing (ignore failures)
        try:
            if nc_path is not None and nc_path.exists():
                #nc_path.unlink()
                print('not removing anything anymore')
        except Exception:
            pass
    return rmse

# Optuna objective moved to module level so it is picklable by multiprocessing
def objective(trial):
    # Sample parameters using Optuna according to param_ranges
    params = {}
    for k, v in param_ranges.items():
        if v[0] == v[1]:
            params[k] = v[0]
        elif isinstance(v[0], int) and isinstance(v[1], int):
            params[k] = trial.suggest_int(k, int(v[0]), int(v[1]))
        else:
            params[k] = trial.suggest_float(k, float(v[0]), float(v[1]))
    
    # Derived parameter
    params["grid_resolution"] = params["max_depth"] * 10

    # Create unique sim dir for this trial
    sim_dir = run_root_dir / f"optuna_trial_{trial.number:06d}"
    if sim_dir.exists():
        shutil.rmtree(sim_dir)
    shutil.copytree(base_sim_dir, sim_dir)
    update_config_yaml(sim_dir, params)

    # Run simulation and get RMSE
    rmse = run_simulation(sim_dir)

    # Attach some user-attrs (optional)
    trial.set_user_attr("sim_dir", str(sim_dir))
    trial.set_user_attr("rmse", float(rmse))

    # Log trial results to calibration_prints.txt
    with open(print_log, "a", encoding="utf-8") as f:
        f.write(f"Trial {trial.number} completed.\n")
        f.write(f"  Parameters: {params}\n")
        f.write(f"  RMSE: {rmse}\n")
        f.write(f"  Simulation directory: {sim_dir}\n")

    return float(rmse)

# Define worker
def worker(args):
    idx, params = args
    sim_dir = run_root_dir / f'run_{idx:04d}'
    if sim_dir.exists():
        shutil.rmtree(sim_dir)
    shutil.copytree(base_sim_dir, sim_dir)
    update_config_yaml(sim_dir, params)
    rmse = run_simulation(sim_dir)
    # return sim index so main can record it
    return (int(idx), params, float(rmse))

# Main
if __name__ == "__main__":
    # Prepare result file paths and ensure root dir exists
    run_root_dir.mkdir(parents=True, exist_ok=True)
    results_csv = run_root_dir / "calibration_results.csv"
    print_log   = run_root_dir / "calibration_prints.txt"

    # Ensure sqlite uses WAL for concurrent writers (needed for optuna multiprocessing)
    try:
        import sqlite3
        db_path = (run_root_dir / 'optuna_study.db').as_posix()
        con = sqlite3.connect(db_path)
        con.execute('PRAGMA journal_mode=WAL;')
        con.close()
    except Exception:
        pass

    # Multiprocessing start method already set at module import time above.

    # Initialize/clear print log start line
    with open(print_log, "a", encoding="utf-8") as f:
        f.write(f"Calibration run started: {__import__('datetime').datetime.now()}\n")

    # Create Optuna study with SQLite storage so multi-process optimization is possible
    storage_url = f"sqlite:///{(run_root_dir / 'optuna_study.db').as_posix()}"
    study = optuna.create_study(
        study_name="calibration_optuna",
        direction="minimize",
        storage=storage_url,
        load_if_exists=True
    )

    # Determine number of trials
    n_trials = int(n_epochs) * int(n_parallel)

    # Run optimization (n_jobs controls concurrency)
    try:
        study.optimize(objective, n_trials=n_trials, n_jobs=n_parallel, callbacks=[
            lambda study, trial: gc.collect()  # Force garbage collection after each trial
        ])
    except KeyboardInterrupt:
        print("Optimization interrupted by user.")
        with open(print_log, "a", encoding="utf-8") as f:
            f.write("Optimization interrupted by user.\n")

    # After optimization: export trials to CSV
    try:
        df = study.trials_dataframe()
        # Convert any complex objects to strings if needed
        df.to_csv(results_csv, index=False)
    except Exception as e:
        with open(print_log, "a", encoding="utf-8") as f:
            f.write(f"Failed to write trials dataframe: {e}\n")

    # Final summary and best params
    best_trial = study.best_trial
    best_params = best_trial.params
    best_rmse = best_trial.value

    with open(print_log, "a", encoding="utf-8") as f:
        f.write(f"\nCalibration finished: {__import__('datetime').datetime.now()}\n")
        f.write(f"Best parameters found: {best_params}\n")
        f.write(f"Best trial number: {best_trial.number}\n")
        f.write(f"Best RMSE: {best_rmse}\n")

    # Ensure the best-found values are printed to the console as well
    print("Best parameters found:", best_params)
    print("Best trial number:", best_trial.number)
    print("Best RMSE:", best_rmse)

    # Explicitly run garbage collection at the end
    gc.collect()

    ray.init()

    @ray.remote
    def run_trial(trial_number, storage_url):
        study = optuna.create_study(
            study_name="calibration_optuna",
            direction="minimize",
            storage=storage_url,
            load_if_exists=True
        )
        study.optimize(objective, n_trials=1, n_jobs=1)

    tasks = [run_trial.remote(i, storage_url) for i in range(n_trials)]
    ray.get(tasks)
    ray.shutdown()
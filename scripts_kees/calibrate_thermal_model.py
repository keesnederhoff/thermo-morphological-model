# Arctic-XBeach
# Script to run Arctic-XBeach simulations, calibrate and anlyse
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Settings
outdir = Path(r'd:\Git\thermo-morphological-model\runs\20250822_calibration_runs\run001_cal_gt61\analysis')
outdir.mkdir(exist_ok=True)
mpl.rcdefaults()

def compute_metrics(model_time, model_temp, obs_time, obs_temp):
    # Ensure both time columns are datetime and sorted
    model_time = pd.to_datetime(model_time)
    obs_time = pd.to_datetime(obs_time)
    df_model_temp = pd.DataFrame({'time': model_time, 'temp': model_temp}).sort_values('time')
    df_obs_temp = pd.DataFrame({'time': obs_time, 'temp': obs_temp}).sort_values('time')
    # Use merge_asof for nearest matching
    merged = pd.merge_asof(df_obs_temp, df_model_temp, on='time', direction='nearest', suffixes=('_obs', '_model'))
    valid = ~pd.isnull(merged['temp_obs']) & ~pd.isnull(merged['temp_model'])
    obs_vals = merged.loc[valid, 'temp_obs']
    model_vals = merged.loc[valid, 'temp_model']
    rmse = mean_squared_error(obs_vals, model_vals, squared=False)
    mae = mean_absolute_error(obs_vals, model_vals)
    bias = (model_vals - obs_vals).mean()
    return rmse, mae, bias

def compute_all_metrics(df_model, df_val_data):
    # load in the modelling results
    df_gt = df_model

    # Lets add some columns to convert to Celcius
    erikson_colnames = df_val_data.columns[1:]
    
    colnames = [
        "temp_0.0m[K]","temp_0.25m[K]","temp_0.5m[K]","temp_0.75m[K]",
        "temp_1.0m[K]","temp_1.25m[K]","temp_1.5m[K]","temp_1.75m[K]",
        "temp_2.0m[K]","temp_2.25m[K]","temp_2.5m[K]","temp_2.75m[K]", "temp_2.95m[K]",
        "temp_3.0m[K]","temp_3.25m[K]","temp_3.5m[K]","temp_3.75m[K]",
        "temp_4.0m[K]","temp_4.25m[K]","temp_4.5m[K]","temp_4.75m[K]"
    ]
    
    for colname in colnames:
        
        df_gt[f'{colname[:-3]}[C]'] = df_gt[colname] - 273.15
     
    # remove repeated years from model data
    indice = np.max(df_gt[df_gt.time == df_gt.time.values[0]].index)
    df_gt = df_gt[df_gt.index >= indice]
        
    # make sure both dataframes span the same time frame
    t_start = np.max((df_gt.time.values[0], df_val_data.date_time.values[0]))
    # t_end = np.min((df_gt.time.values[-1], df_val_data.date_time.values[-1]))
    
    t_end = pd.to_datetime("2016-08-18")  # hard cutoff, since data temperature data obtained after this is incomplete

    mask1 = (df_gt['time'] >= t_start) * (df_gt['time'] <= t_end)    
    df1 = df_gt[mask1]

    mask2 = (df_val_data['date_time'] >= t_start) * (df_val_data['date_time'] <= t_end)    
    df2 = df_val_data[mask2]
    
    # print(df1.time)
    # print(df2.date_time)
    
    # compute RMSE per soil layer
    colnames1 = [
        "temp_0.0m[C]","temp_0.25m[C]","temp_0.5m[C]",
        "temp_1.0m[C]","temp_1.25m[C]","temp_1.5m[C]","temp_1.75m[C]",
        "temp_2.0m[C]","temp_2.25m[C]","temp_2.5m[C]","temp_2.75m[C]",
        "temp_3.0m[C]","temp_3.25m[C]","temp_3.5m[C]","temp_3.75m[C]",
        "temp_4.0m[C]","temp_4.25m[C]",
    ]
    colnames2 = [
        'dpth0cm', 'dpth25cm', 'dpth50cm', 
        'dpth100cm', 'dpth125cm', 'dpth150cm', 'dpth175cm', 
        'dpth200cm', 'dpth225cm', 'dpth250cm', 'dpth275cm', 
        'dpth300cm', 'dpth325cm', 'dpth350cm', 'dpth375cm',
        'dpth400cm', 'dpth425cm'
    ]
    metrics = {}
    for c1, c2 in zip(colnames1, colnames2):
        rmse, mae, bias = compute_metrics(
            df_model['time'], df_model[c1],
            df_val_data['date_time'], df_val_data[c2]
        )
        metrics[c2] = {'RMSE': rmse, 'MAE': mae, 'Bias': bias}
    return metrics
    


# Part 1 - load in Erikson data
df_erikson1 = pd.read_csv(r"d:\Git\thermo-morphological-model\database\raw_datasets\erikson2\Temp_arrays USGS-UCSC_Oberle\BI_T-1_processed.csv", parse_dates=['date_time'])

# Part 2 - load in model output data
df_model    = pd.read_csv(r'd:\Git\thermo-morphological-model\runs\20250822_calibration_runs\run001_cal_gt61\results\ground_temperature_timeseries.csv', parse_dates=['time'])

# Part 3 - determine skill
# fig1, ax1 = plt.subplots(figsize=(18, 3))
# fig2, ax2 = plt.subplots(figsize=(18, 3))
# fig3, ax3 = plt.subplots(figsize=(18, 3))
# fig4, ax4 = plt.subplots(figsize=(18, 3))

# Compute skill
metrics_found = compute_all_metrics(df_model, df_erikson1)

# Print metrics for main depths
for depth in ['dpth50cm', 'dpth100cm', 'dpth200cm', 'dpth300cm']:
    m = metrics_found[depth]
    print(f"{depth}: RMSE={m['RMSE']:.2f}, MAE={m['MAE']:.2f}, Bias={m['Bias']:.2f}")

fig, axs = plt.subplots(4, 1, figsize=(18, 12), sharex=True)
ax1, ax2, ax3, ax4 = axs



ax1.set_title('50 cm depth')
ax1.plot(df_model['time'], df_model['temp_0.5m[K]'] - 273.15, 'C0', label=f'Modelled')
ax1.plot(df_erikson1['date_time'], df_erikson1['dpth50cm'], 'k', label='Measured')

ax2.set_title('100 cm depth')
ax2.plot(df_model['time'], df_model['temp_1.0m[K]'] - 273.15, 'C1', label=f'Modelled')
ax2.plot(df_erikson1['date_time'], df_erikson1['dpth100cm'], 'k', label='Measured')

ax3.set_title('200 cm depth')
ax3.plot(df_model['time'], df_model['temp_2.0m[K]'] - 273.15, 'C2', label=f'Modelled')
ax3.plot(df_erikson1['date_time'], df_erikson1['dpth200cm'], 'k', label='Measured')

ax4.set_title('300 cm depth')
ax4.plot(df_model['time'], df_model['temp_3.0m[K]'] - 273.15, 'C3', label=f'Modelled')
ax4.plot(df_erikson1['date_time'], df_erikson1['dpth300cm'], 'k', label='Measured')

# Add skill scores to each subplot
for ax, depth, color in zip([ax1, ax2, ax3, ax4], ['dpth50cm', 'dpth100cm', 'dpth200cm', 'dpth300cm'], ['C0', 'C1', 'C2', 'C3']):
    m = metrics_found[depth]
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

for ax in [ax1, ax2, ax3, ax4]:
    
    ax.set_xlim((pd.to_datetime("2016-05-15"), pd.to_datetime("2016-09-30")))
    ax.set_ylim((-10, 5))
    
    ax.set_xticks([
        pd.to_datetime("2016-05-15"),

        pd.to_datetime("2016-05-31"),
        pd.to_datetime("2016-06-15"),
        pd.to_datetime("2016-06-30"),
        pd.to_datetime("2016-07-15"),
        pd.to_datetime("2016-07-31"),
        pd.to_datetime("2016-08-15"),
        pd.to_datetime("2016-08-31"),
        pd.to_datetime("2016-09-15"),

        pd.to_datetime("2016-09-30"),

    ],
    [   
        None,
        "2016-05-31",
        "2016-06-15",
        "2016-06-30",
        "2016-07-15",
        "2016-07-31",
        "016-08-15",
        "2016-08-31",
        "2016-09-15",
        None
    ]
    )
    
    ax.legend(loc='upper left')
    # ax.set_xlabel('time')
    ax.set_ylabel('Temperature [C]')
    ax.grid()
    
ax4.set_xlabel('Time')

fig.suptitle("Temperature at different soil layers (modelled and measured, for calibration dataset)")
fig.tight_layout()
fig.savefig(outdir / "temperature_layers.png", dpi=300)
plt.close()
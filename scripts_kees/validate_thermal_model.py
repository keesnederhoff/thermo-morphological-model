# Arctic-XBeach
# Script to run Arctic-XBeach simulations, calibrate and anlyse
from pathlib import Path
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Settings
outdir = Path(r'd:\Git\thermo-morphological-model\runs\20250822_validation_runs\run002_kevin_values\analysis')
outdir.mkdir(exist_ok=True)
mpl.rcdefaults()
## Part 1 - Load data
df_erikson = pd.read_csv(r'd:\Git\thermo-morphological-model\database\ts_datasets\ground_temperature_erikson.csv', parse_dates=['time'])

# Part 2A - load in model output data
df_model = pd.read_csv(r'd:\Git\thermo-morphological-model\runs\20250822_validation_runs\run002_kevin_values\results\ground_temperature_timeseries.csv', parse_dates=['time'])
colnames = ['air_temp[K]', 'temp_0.0m[K]', 'temp_0.5m[K]', 'temp_1.0m[K]', 'temp_2.0m[K]', 'temp_2.95m[K]']
for colname in colnames:
    df_model[f'{colname[:-3]}[C]'] = df_model[colname] - 273.15

# Compute skill
# ...existing code...

from sklearn.metrics import mean_squared_error, mean_absolute_error

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

metrics = {}

metrics['0.5m'] = compute_metrics(
    df_model['time'], df_model['temp_0.5m[C]'],
    df_erikson['time'], df_erikson['T50cm']
)
metrics['1.0m'] = compute_metrics(
    df_model['time'], df_model['temp_1.0m[C]'],
    df_erikson['time'], df_erikson['T100cm']
)
metrics['2.0m'] = compute_metrics(
    df_model['time'], df_model['temp_2.0m[C]'],
    df_erikson['time'], df_erikson['T200cm']
)
metrics['2.95m'] = compute_metrics(
    df_model['time'], df_model['temp_2.95m[C]'],
    df_erikson['time'], df_erikson['T295cm']
)

for depth, (rmse, mae, bias) in metrics.items():
    print(f"{depth}: RMSE={rmse:.2f}, MAE={mae:.2f}, Bias={bias:.2f}")

# Define colors for modeled values
soil_color = "#1f77b4"  # blue for modeled soil temperature
air_color = "#ff7f0e"   # orange for air temperature


# Figure size
fig, axs = plt.subplots(5, 1, figsize=(18, 20), constrained_layout=True)
ax0, ax1, ax2, ax3, ax4 = axs

# Top panel: Air and surface
ax0.plot(df_model['time'], df_model['air_temp[C]'], label='Air temperature', color=air_color)
ax0.plot(df_model['time'], df_model['temp_0.0m[C]'], label='Modeled temperature at surface', color=soil_color)

# Soil panels: modeled in blue, observed in black
ax1.plot(df_model['time'], df_model['temp_0.5m[C]'], label='Modeled', color=soil_color)
ax1.plot(df_erikson['time'], df_erikson['T50cm'], label='Observed', color='black')
ax2.plot(df_model['time'], df_model['temp_1.0m[C]'], color=soil_color)
ax2.plot(df_erikson['time'], df_erikson['T100cm'], color='black')
ax3.plot(df_model['time'], df_model['temp_2.0m[C]'], color=soil_color)
ax3.plot(df_erikson['time'], df_erikson['T200cm'], color='black')
ax4.plot(df_model['time'], df_model['temp_2.95m[C]'], color=soil_color)
ax4.plot(df_erikson['time'], df_erikson['T295cm'], color='black')

# ...existing code...
depth_titles = {
    '0.5m': 'Temperature at 0.5m depth',
    '1.0m': 'Temperature at 1.0m depth',
    '2.0m': 'Temperature at 2.0m depth',
    '2.95m': 'Temperature at 3.0m depth'
}

for ax, depth in zip(axs[1:], ['0.5m', '1.0m', '2.0m', '2.95m']):
    rmse, mae, bias = metrics[depth]
    skill_text = (
        f"RMSE = {rmse:.2f} °C\n"
        f"MAE = {mae:.2f} °C\n"
        f"Bias = {bias:.2f} °C"
    )
    ax.text(
        0.95, 0.05, skill_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )
    ax.set_title(depth_titles[depth], fontsize=12)

#

for i, ax in enumerate(axs):
    ax.legend(loc='upper left')
    ax.set_xlabel('time')
    ax.set_ylabel('temperature [C]')
    if i == 0:
        ax.set_ylim((-20, 20))  # Top panel: Air/surface temperature
    else:
        ax.set_ylim((-10, 2))   # Other panels: Soil layers
    ax.set_xlim(pd.Timestamp('2011-05-01'), pd.Timestamp('2011-11-30'))
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
fig.suptitle("Temperature at different soil layers (modeled vs observed)", fontsize=16)
fig.savefig(outdir / "temperature_layers.png", dpi=300)
plt.close()


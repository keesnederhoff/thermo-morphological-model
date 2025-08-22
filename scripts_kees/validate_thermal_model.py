# Arctic-XBeach
# Script to run Arctic-XBeach simulations, calibrate and anlyse
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Part 1 - Load data
df_erikson = pd.read_csv(r'd:\Git\thermo-morphological-model\database\ts_datasets\ground_temperature_erikson.csv')

# Part 2 - load in model output data
df_model = pd.read_csv(r'd:\Git\thermo-morphological-model\validation\data\val_gt24_ground_temperature_timeseries.csv')
colnames = ['air_temp[K]', 'temp_0m[K]', 'temp_0.5m[K]', 'temp_1.0m[K]', 'temp_2.0m[K]', 'temp_2.95m[K]']
for colname in colnames:
    df_model[f'{colname[:-3]}[C]'] = df_model[colname] - 273.15

## Part 3 - plot


# Only compare 1m depth
fig, axs = plt.subplots(1, 1, figsize=(5, 5))
ax1 = axs
ax1.plot(df_model['time'], df_model['temp_1.0m[C]'], label='Modelled temperature at 1m depth')
ax1.plot(df_erikson['time'], df_erikson['T100cm'], label='Measured temperature at 1m depth')
plt.show()

fig, axs = plt.subplots(5, 1, figsize=(15, 15))
ax0, ax1, ax2, ax3, ax4 = axs
ax0.plot(df_model['time'], df_model['temp_0m[C]'], label='Modelled temperature at surface')
ax1.plot(df_model['time'], df_model['temp_0.5m[C]'], label='Modelled temperature at 0.5m depth')
ax1.plot(df_erikson['time'], df_erikson['T50cm'], label='Measured temperature at 0.5m depth')
ax2.plot(df_model['time'], df_model['temp_1.0m[C]'], label='Modelled temperature at 1m depth')
ax2.plot(df_erikson['time'], df_erikson['T100cm'], label='Measured temperature at 1m depth')
ax3.plot(df_model['time'], df_model['temp_2.0m[C]'], label='Modelled temperature at 2m depth')
ax3.plot(df_erikson['time'], df_erikson['T200cm'], label='Measured temperature at 2m depth')
ax4.plot(df_model['time'], df_model['temp_2.95m[C]'], label='Modelled temperature at 2.95m depth')
ax4.plot(df_erikson['time'], df_erikson['T295cm'], label='Measured temperature at 2.95m depth')

for ax in axs:
    ax.legend(loc='upper left')
    ax.set_xlabel('time')
    ax.set_ylabel('temperature [C]')
    ax.set_ylim((-6, 1))

fig.suptitle("Temperature at different soil layers (modelled vs measured)")

plt.show()

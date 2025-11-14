# Arctic-XBeach
# Script to run Arctic-XBeach simulations, calibrate and anlyse
from pathlib import Path
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Settings
outdir = Path(r'd:\Git\thermo-morphological-model\runs\20250822_validation_runs\run003_calibrated_values_v8_33')
outdir.mkdir(exist_ok=True)
mpl.rcdefaults()

## Part 1 - Load data
df_erikson = pd.read_csv(r'd:\Git\thermo-morphological-model\database\ts_datasets\ground_temperature_erikson.csv', parse_dates=['time'])

# Part 2A - load in model output data
df_model = pd.read_csv(r'd:\Git\thermo-morphological-model\runs\20250822_validation_runs\run002_kevin_values\results\ground_temperature_timeseries.csv', parse_dates=['time'])
colnames = ['air_temp[K]', 'temp_0.0m[K]', 'temp_0.5m[K]', 'temp_1.0m[K]', 'temp_2.0m[K]', 'temp_2.95m[K]']
for colname in colnames:
    df_model[f'{colname[:-3]}[C]'] = df_model[colname] - 273.15

# Also load 2D profile of temperature
from netCDF4 import Dataset
import numpy as np

nc_path = r'd:\Git\thermo-morphological-model\runs\20250822_validation_runs\run003_calibrated_values_v8_33\results\results.nc'

with Dataset(nc_path, 'r') as nc:
    zgr = nc.variables['zgr'][:]  # depth grid
    xgr = nc.variables['xgr'][:]  # horizontal grid
    ground_temperature_distribution = nc.variables['ground_temperature_distribution'][:]  # shape: [time, depth, x]
    depth_id = nc.variables['depth_id'][:]  # depth indices or values
    time = nc.variables['time'][:]  # time variable (may need conversion)

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

# Compute combined metrics across all layers
def compute_combined_metrics(metrics_dict, df_model, df_erikson):
    """
    Compute combined RMSE, MAE, and bias across all soil layers.
    
    Parameters:
    - metrics_dict: Dictionary with individual layer metrics
    - df_model: DataFrame with model temperature data
    - df_erikson: DataFrame with observed temperature data
    
    Returns:
    - combined_rmse, combined_mae, combined_bias
    """
    all_obs_vals = []
    all_model_vals = []
    
    # Layer mapping: depth -> (model_col, obs_col)
    layer_mapping = {
        '0.5m': ('temp_0.5m[C]', 'T50cm'),
        '1.0m': ('temp_1.0m[C]', 'T100cm'),
        '2.0m': ('temp_2.0m[C]', 'T200cm'),
        '2.95m': ('temp_2.95m[C]', 'T295cm')
    }
    
    for depth, (model_col, obs_col) in layer_mapping.items():
        # Get time series for this layer
        model_time = pd.to_datetime(df_model['time'])
        obs_time = pd.to_datetime(df_erikson['time'])
        
        df_model_temp = pd.DataFrame({'time': model_time, 'temp': df_model[model_col]}).sort_values('time')
        df_obs_temp = pd.DataFrame({'time': obs_time, 'temp': df_erikson[obs_col]}).sort_values('time')
        
        # Merge and get valid pairs
        merged = pd.merge_asof(df_obs_temp, df_model_temp, on='time', direction='nearest', suffixes=('_obs', '_model'))
        valid = ~pd.isnull(merged['temp_obs']) & ~pd.isnull(merged['temp_model'])
        
        # Collect all valid pairs
        obs_vals = merged.loc[valid, 'temp_obs']
        model_vals = merged.loc[valid, 'temp_model']
        
        all_obs_vals.extend(obs_vals.tolist())
        all_model_vals.extend(model_vals.tolist())
    
    # Convert to numpy arrays for calculation
    all_obs_vals = np.array(all_obs_vals)
    all_model_vals = np.array(all_model_vals)
    
    # Compute combined metrics
    combined_rmse = mean_squared_error(all_obs_vals, all_model_vals, squared=False)
    combined_mae = mean_absolute_error(all_obs_vals, all_model_vals)
    combined_bias = (all_model_vals - all_obs_vals).mean()
    
    return combined_rmse, combined_mae, combined_bias

# Calculate combined metrics
combined_rmse, combined_mae, combined_bias = compute_combined_metrics(metrics, df_model, df_erikson)

print(f"\nCombined metrics across all layers:")
print(f"Combined RMSE = {combined_rmse:.2f} °C")
print(f"Combined MAE = {combined_mae:.2f} °C")  
print(f"Combined Bias = {combined_bias:.2f} °C")
print(f"Total data points = {sum(len(pd.merge_asof(pd.DataFrame({'time': pd.to_datetime(df_erikson['time']), 'temp': df_erikson[obs_col]}), pd.DataFrame({'time': pd.to_datetime(df_model['time']), 'temp': df_model[model_col]}), on='time', direction='nearest').dropna()) for model_col, obs_col in [('temp_0.5m[C]', 'T50cm'), ('temp_1.0m[C]', 'T100cm'), ('temp_2.0m[C]', 'T200cm'), ('temp_2.95m[C]', 'T295cm')])}")

# Add combined metrics to the summary text on plots
combined_skill_text = (
    f"All layers combined:\n"
    f"RMSE = {combined_rmse:.2f} °C\n"
    f"MAE = {combined_mae:.2f} °C\n"
    f"Bias = {combined_bias:.2f} °C"
)

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

# Add combined metrics to the top panel
axs[0].text(
    0.05, 0.95, combined_skill_text,
    transform=axs[0].transAxes,
    fontsize=12,
    verticalalignment='top',
    horizontalalignment='left',
    bbox=dict(facecolor='lightblue', alpha=0.7, edgecolor='none')
)

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



## Make plot where we plot the temperature as 2D matrix (depth and time)
time_origin = pd.Timestamp('2001-01-01')
time_seconds = np.array(time)  # Ensure it's a numpy array, not masked
time_dt = time_origin + pd.to_timedelta(time_seconds, unit='s')

# Make depth
depths = depth_id*-0.1

# ground_temperature_distribution shape: [time, depth, x]
temp_2d = np.squeeze(ground_temperature_distribution[:, 1, :]) - 273.15  # Convert from K to C

# Compute depth of 0 degrees isotherm
def compute_zero_degree_depth(temp_2d, depths, time_dt):
    """
    Compute the depth of 0°C isotherm for each time step.
    
    Parameters:
    - temp_2d: 2D temperature array (time x depth)
    - depths: depth array (negative values)
    - time_dt: datetime array
    
    Returns:
    - DataFrame with time and zero_degree_depth columns
    """
    zero_degree_depths = []
    valid_times = []

    #fig  = plt.figure()
    #plt.plot(temp_2d[:,0])
    #plt.plot(temp_2d[:,1])
    #plt.plot(temp_2d[:,3])
    #plt.show()
    
    for i, t in enumerate(time_dt):
        temp_profile = temp_2d[i, :]  # Temperature at time i for all depths
        
        # Find where temperature crosses 0°C
        # Look for sign changes in temperature
        zero_crossings = np.where(temp_profile > -1)
        [idx,idy] = np.shape(zero_crossings) 
        
        if idy > 1:
            # Take the last crossing (deepest)
            idx = zero_crossings[0][-1]
            
            # Do this simple
            zero_degree_depths.append(depths[idx])
            valid_times.append(t)
        else:
            # If no crossings found, add NaN with the date for better plotting
            zero_degree_depths.append(np.nan)
            valid_times.append(t)
    
    return pd.DataFrame({'time': valid_times, 'zero_degree_depth': zero_degree_depths})

# Compute zero degree depth data
zero_df = compute_zero_degree_depth(temp_2d, depths, time_dt)

# Create combined figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Subplot A: 2D temperature plot with contour
im = ax1.pcolormesh(time_dt, depths, temp_2d.T, shading='auto', cmap='coolwarm', vmin=-10, vmax=2)

# Add zero degree contour line
if len(zero_df) > 0:
    ax1.plot(zero_df['time'], zero_df['zero_degree_depth'], 
             color='black', linewidth=3, label='Active layer')

# Overlay observations as circles
obs_depths = [0.5, 1.0, 2.0, 2.95]
obs_cols = ['T50cm', 'T100cm', 'T200cm', 'T295cm']
for depth, col in zip(obs_depths, obs_cols):
    depth = depth*-1
    df_thin = df_erikson.iloc[::24]
    ax1.scatter(
        df_thin['time'],
        [depth]*len(df_thin),
        c=df_thin[col],
        cmap='coolwarm',
        edgecolor='black',
        s=40,
        label=f'Obs {depth}m',
        vmin=-10,
        vmax=2
    )

ax1.set_xlim(pd.Timestamp('2011-05-01'), pd.Timestamp('2012-01-01'))
ax1.set_ylim(-5, -0.1)
ax1.set_xlabel('Time')
ax1.set_ylabel('Depth [m]')
ax1.set_title('a) Ground Temperature with 0°C Isotherm')
ax1.legend(loc='upper right', fontsize=8)

# Add colorbar for first subplot
cbar = fig.colorbar(im, ax=ax1, label='Temperature [°C]')

# Subplot B: Monthly boxplots of zero degree depth
if len(zero_df) > 0:
    # Add month column to zero_df
    zero_df['month'] = zero_df['time'].dt.month
    zero_df['month_name'] = zero_df['time'].dt.strftime('%b')
    
    # Filter for May to December (months 5-12)
    zero_df_filtered = zero_df[zero_df['month'].isin([5, 6, 7, 8, 9, 10, 11, 12])]
    
    # Create boxplot data
    months_order = ['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_data = []
    month_labels = []
    
    for month_num, month_name in zip([5, 6, 7, 8, 9, 10, 11, 12], months_order):
        month_subset = zero_df_filtered[zero_df_filtered['month'] == month_num]
        if len(month_subset) > 0:
            month_data.append(month_subset['zero_degree_depth'].values)
            month_labels.append(month_name)
    
    # Create boxplot
    if month_data:
        bp = ax2.boxplot(month_data, labels=month_labels, patch_artist=True)
        
        # Color the boxes
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Depth of 0°C Isotherm [m]')
        ax2.set_title('b) Monthly Distribution of Frost Depth')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics text
        overall_mean = zero_df_filtered['zero_degree_depth'].mean()
        overall_std = zero_df_filtered['zero_degree_depth'].std()
        ax2.text(0.02, 0.98, f'Mean: {overall_mean:.2f} m\nStd: {overall_std:.2f} m', 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax2.text(0.5, 0.5, 'No zero-degree crossings found\nin selected time period', 
                transform=ax2.transAxes, ha='center', va='center')
        ax2.set_title('b) Monthly Distribution of Frost Depth')
else:
    ax2.text(0.5, 0.5, 'No zero-degree crossings found', 
            transform=ax2.transAxes, ha='center', va='center')
    ax2.set_title('b) Monthly Distribution of Frost Depth')

plt.tight_layout()
fig.savefig(outdir / "temperature_2D_depth_time_with_frost_analysis.png", dpi=300, bbox_inches='tight')
plt.close(fig)

# Create separate boxplot figure for frost depth analysis
if len(zero_df) > 0:
    fig_box, ax_box = plt.subplots(figsize=(10, 6))
    
    # Remove NaN values for boxplot analysis
    zero_df_clean = zero_df.dropna(subset=['zero_degree_depth'])
    
    if len(zero_df_clean) > 0:
        # Add month column to zero_df_clean
        zero_df_clean = zero_df_clean.copy()  # Avoid SettingWithCopyWarning
        zero_df_clean['month'] = zero_df_clean['time'].dt.month
        zero_df_clean['month_name'] = zero_df_clean['time'].dt.strftime('%b')
        
        # Filter for May to December (months 5-12)
        zero_df_filtered = zero_df_clean[zero_df_clean['month'].isin([5, 6, 7, 8, 9, 10, 11, 12])]
        
        # Create boxplot data
        months_order = ['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_data = []
        month_labels = []
        
        for month_num, month_name in zip([5, 6, 7, 8, 9, 10, 11, 12], months_order):
            month_subset = zero_df_filtered[zero_df_filtered['month'] == month_num]
            if len(month_subset) > 0:
                month_data.append(month_subset['zero_degree_depth'].values)
                month_labels.append(month_name)
        
        # Create boxplot
        if month_data:
            bp = ax_box.boxplot(month_data, labels=month_labels, patch_artist=True)
            
            # Color the boxes with a gradient
            colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(bp['boxes'])))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.8)
            
            ax_box.set_xlabel('Month', fontsize=12)
            ax_box.set_ylabel('Depth of 0°C Isotherm [m]', fontsize=12)
            ax_box.set_title('Monthly Distribution of Frost Depth (0°C Isotherm)', fontsize=14)
            ax_box.grid(True, alpha=0.3)
            
            # Add detailed statistics
            overall_mean = zero_df_filtered['zero_degree_depth'].mean()
            overall_std = zero_df_filtered['zero_degree_depth'].std()
            overall_min = zero_df_filtered['zero_degree_depth'].min()
            overall_max = zero_df_filtered['zero_degree_depth'].max()
            total_points = len(zero_df)
            valid_points = len(zero_df_clean)
            
            stats_text = (
                f'Statistics (May-Dec):\n'
                f'Mean: {overall_mean:.2f} m\n'
                f'Std: {overall_std:.2f} m\n'
                f'Min: {overall_min:.2f} m\n'
                f'Max: {overall_max:.2f} m\n'
                f'Valid points: {valid_points}/{total_points}'
            )
            
            ax_box.text(0.02, 0.98, stats_text, 
                       transform=ax_box.transAxes, verticalalignment='top',
                       fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            plt.tight_layout()
            fig_box.savefig(outdir / "frost_depth_monthly_boxplot.png", dpi=300, bbox_inches='tight')
        else:
            ax_box.text(0.5, 0.5, 'No valid frost depth data found\nin selected time period', 
                       transform=ax_box.transAxes, ha='center', va='center', fontsize=12)
            ax_box.set_title('Monthly Distribution of Frost Depth')
            fig_box.savefig(outdir / "frost_depth_monthly_boxplot.png", dpi=300, bbox_inches='tight')
    else:
        ax_box.text(0.5, 0.5, 'No valid frost depth data found', 
                   transform=ax_box.transAxes, ha='center', va='center', fontsize=12)
        ax_box.set_title('Monthly Distribution of Frost Depth')
        fig_box.savefig(outdir / "frost_depth_monthly_boxplot.png", dpi=300, bbox_inches='tight')
    
    plt.close(fig_box)

# Also save the original 2D plot for backward compatibility
fig_orig, ax_orig = plt.subplots(figsize=(11.69, 8.27/2))
im_orig = ax_orig.pcolormesh(time_dt, depths, temp_2d.T, shading='auto', cmap='coolwarm', vmin=-10, vmax=2)
if len(zero_df) > 0:
    ax_orig.plot(zero_df['time'], zero_df['zero_degree_depth'], 
                 color='black', linewidth=3, label='0°C isotherm')
cbar_orig = fig_orig.colorbar(im_orig, ax=ax_orig, label='Temperature [°C]')
ax_orig.set_xlabel('Time')
ax_orig.set_ylabel('Depth [m]')
ax_orig.set_title('Ground Temperature (Depth vs Time) with 0°C Isotherm')
ax_orig.set_xlim(pd.Timestamp('2011-05-01'), pd.Timestamp('2012-01-01'))
ax_orig.set_ylim(-5, -0.1)
if len(zero_df) > 0:
    ax_orig.legend(loc='upper right')
fig_orig.tight_layout()
fig_orig.savefig(outdir / "temperature_2D_depth_time.png", dpi=300)
plt.close(fig_orig)

# Print zero degree depth statistics
if len(zero_df) > 0:
    # Remove NaN values for statistics
    zero_df_clean = zero_df.dropna(subset=['zero_degree_depth'])
    total_points = len(zero_df)
    valid_points = len(zero_df_clean)
    
    print(f"\nZero-degree isotherm statistics:")
    print(f"Total time points: {total_points}")
    print(f"Valid crossings found: {valid_points} ({valid_points/total_points*100:.1f}%)")
    
    if valid_points > 0:
        print(f"Mean depth: {zero_df_clean['zero_degree_depth'].mean():.2f} m")
        print(f"Standard deviation: {zero_df_clean['zero_degree_depth'].std():.2f} m")
        print(f"Minimum depth: {zero_df_clean['zero_degree_depth'].min():.2f} m")
        print(f"Maximum depth: {zero_df_clean['zero_degree_depth'].max():.2f} m")
    else:
        print("No valid zero-degree crossings found in any time step")
else:
    print("No zero-degree isotherm data generated")

# Active Layer Development Analysis
def analyze_active_layer_development(zero_df):
    """
    Analyze active layer development by month for the last year of data, including frequency and depth statistics.
    
    Parameters:
    - zero_df: DataFrame with time and zero_degree_depth columns
    
    Returns:
    - Dictionary with monthly statistics and last_year_info
    """
    if len(zero_df) == 0:
        return {}
    
    # Filter for last year of data only
    max_date = zero_df['time'].max()
    last_year_start = pd.Timestamp(year=max_date.year, month=1, day=1)
    last_year_end = pd.Timestamp(year=max_date.year, month=12, day=31)
    
    # Filter data to last year only
    last_year_data = zero_df[(zero_df['time'] >= last_year_start) & (zero_df['time'] <= last_year_end)]
    
    if len(last_year_data) == 0:
        print(f"Warning: No data found for last year ({max_date.year})")
        return {}
    
    # Create a copy and add temporal columns
    df_analysis = last_year_data.copy()
    df_analysis['month'] = df_analysis['time'].dt.month
    df_analysis['month_name'] = df_analysis['time'].dt.strftime('%B')
    df_analysis['day_of_year'] = df_analysis['time'].dt.dayofyear
    df_analysis['has_frost'] = ~df_analysis['zero_degree_depth'].isna()
    
    # Store year info for reporting
    analysis_year = max_date.year
    data_start = last_year_data['time'].min()
    data_end = last_year_data['time'].max()
    
    monthly_stats = {}
    
    # Analyze all months from January to December
    months_data = [(1, 'January'), (2, 'February'), (3, 'March'), (4, 'April'), 
                   (5, 'May'), (6, 'June'), (7, 'July'), (8, 'August'),
                   (9, 'September'), (10, 'October'), (11, 'November'), (12, 'December')]
    
    for month_num, month_name in months_data:
        month_data = df_analysis[df_analysis['month'] == month_num]
        
        if len(month_data) > 0:
            total_days = len(month_data)
            frost_days = month_data['has_frost'].sum()
            no_frost_days = total_days - frost_days
            frost_percentage = (frost_days / total_days) * 100 if total_days > 0 else 0
            
            # Statistics for days with frost
            frost_data = month_data.dropna(subset=['zero_degree_depth'])
            
            stats = {
                'month_name': month_name,
                'total_days': total_days,
                'frost_days': frost_days,
                'no_frost_days': no_frost_days,
                'frost_percentage': frost_percentage,
                'mean_depth': frost_data['zero_degree_depth'].mean() if len(frost_data) > 0 else np.nan,
                'max_depth': frost_data['zero_degree_depth'].min() if len(frost_data) > 0 else np.nan,  # min because depths are negative
                'min_depth': frost_data['zero_degree_depth'].max() if len(frost_data) > 0 else np.nan,  # max because depths are negative  
                'std_depth': frost_data['zero_degree_depth'].std() if len(frost_data) > 0 else np.nan
            }
            
            monthly_stats[month_num] = stats
        else:
            monthly_stats[month_num] = {
                'month_name': month_name,
                'total_days': 0,
                'frost_days': 0,
                'no_frost_days': 0,
                'frost_percentage': 0,
                'mean_depth': np.nan,
                'max_depth': np.nan,
                'min_depth': np.nan,
                'std_depth': np.nan
            }
    
    # Add metadata about the analysis
    monthly_stats['_metadata'] = {
        'analysis_year': analysis_year,
        'data_start': data_start,
        'data_end': data_end,
        'total_data_points': len(df_analysis)
    }
    
    return monthly_stats

# Perform active layer analysis
print(f"\n{'='*80}")
print("ACTIVE LAYER DEVELOPMENT ANALYSIS")
print(f"{'='*80}")

if len(zero_df) > 0:
    monthly_stats = analyze_active_layer_development(zero_df)
    
    # Get analysis metadata
    if '_metadata' in monthly_stats:
        metadata = monthly_stats['_metadata']
        analysis_year = metadata['analysis_year']
        data_start = metadata['data_start']
        data_end = metadata['data_end']
        total_points = metadata['total_data_points']
        
        print(f"\nAnalysis Period: {analysis_year} (Last Year of Available Data)")
        print(f"Data Range: {data_start.strftime('%Y-%m-%d')} to {data_end.strftime('%Y-%m-%d')}")
        print(f"Total Data Points: {total_points}")
    
    # Print detailed monthly statistics
    print(f"\nMonthly Active Layer Statistics for {analysis_year if '_metadata' in monthly_stats else 'Last Year'}:")
    print(f"{'Month':<12} {'Days':<6} {'Frost':<6} {'No Frost':<9} {'Frost %':<8} {'Mean Depth':<12} {'Max Depth':<12} {'Min Depth':<12}")
    print(f"{'-'*12} {'-'*6} {'-'*6} {'-'*9} {'-'*8} {'-'*12} {'-'*12} {'-'*12}")
    
    for month_num in range(1, 13):
        stats = monthly_stats.get(month_num, {})
        month_name = stats.get('month_name', 'Unknown')
        total_days = stats.get('total_days', 0)
        frost_days = stats.get('frost_days', 0)
        no_frost_days = stats.get('no_frost_days', 0)
        frost_percentage = stats.get('frost_percentage', 0)
        mean_depth = stats.get('mean_depth', np.nan)
        max_depth = stats.get('max_depth', np.nan)
        min_depth = stats.get('min_depth', np.nan)
        
        # Format depth values
        mean_str = f"{mean_depth:.2f} m" if not np.isnan(mean_depth) else "N/A"
        max_str = f"{max_depth:.2f} m" if not np.isnan(max_depth) else "N/A"
        min_str = f"{min_depth:.2f} m" if not np.isnan(min_depth) else "N/A"
        
        print(f"{month_name:<12} {total_days:<6} {frost_days:<6} {no_frost_days:<9} {frost_percentage:<7.1f}% {mean_str:<12} {max_str:<12} {min_str:<12}")
    
    # Seasonal analysis - filter for last year only
    print(f"\n{'Seasonal Analysis for Last Year:'}")
    
    # Get last year data
    max_date = zero_df['time'].max()
    last_year_start = pd.Timestamp(year=max_date.year, month=1, day=1)
    last_year_end = pd.Timestamp(year=max_date.year, month=12, day=31)
    last_year_zero_df = zero_df[(zero_df['time'] >= last_year_start) & (zero_df['time'] <= last_year_end)]
    
    seasons = {
        'Winter': [12, 1, 2],
        'Spring': [3, 4, 5], 
        'Summer': [6, 7, 8],
        'Autumn': [9, 10, 11]
    }
    
    for season_name, months in seasons.items():
        season_data = last_year_zero_df[last_year_zero_df['time'].dt.month.isin(months)]
        if len(season_data) > 0:
            season_clean = season_data.dropna(subset=['zero_degree_depth'])
            total_season_days = len(season_data)
            frost_season_days = len(season_clean)
            frost_season_percentage = (frost_season_days / total_season_days) * 100 if total_season_days > 0 else 0
            
            if len(season_clean) > 0:
                mean_depth = season_clean['zero_degree_depth'].mean()
                max_depth = season_clean['zero_degree_depth'].min()  # min because negative
                print(f"{season_name:<8}: {frost_season_days:>3}/{total_season_days:>3} days with frost ({frost_season_percentage:>5.1f}%), "
                      f"Mean: {mean_depth:>6.2f}m, Max: {max_depth:>6.2f}m")
            else:
                print(f"{season_name:<8}: {frost_season_days:>3}/{total_season_days:>3} days with frost ({frost_season_percentage:>5.1f}%), No frost data")
    
    # Active layer thaw analysis (identify thaw periods) - last year only
    print(f"\nActive Layer Thaw Analysis for Last Year:")
    zero_df_analysis = last_year_zero_df.copy()
    zero_df_analysis['month'] = zero_df_analysis['time'].dt.month
    
    # Find thaw season (typically May-September when frost depth decreases or disappears)
    thaw_months = [5, 6, 7, 8, 9]  # May to September
    thaw_data = zero_df_analysis[zero_df_analysis['month'].isin(thaw_months)]
    thaw_data_clean = thaw_data.dropna(subset=['zero_degree_depth'])
    
    if len(thaw_data) > 0:
        total_thaw_days = len(thaw_data)
        no_frost_thaw_days = len(thaw_data) - len(thaw_data_clean)
        complete_thaw_percentage = (no_frost_thaw_days / total_thaw_days) * 100
        
        print(f"Thaw season (May-Sep): {no_frost_thaw_days}/{total_thaw_days} days completely thawed ({complete_thaw_percentage:.1f}%)")
        
        if len(thaw_data_clean) > 0:
            shallowest_frost = thaw_data_clean['zero_degree_depth'].max()  # max because negative values
            deepest_frost = thaw_data_clean['zero_degree_depth'].min()   # min because negative values
            print(f"During partial thaw days: shallowest frost at {shallowest_frost:.2f}m, deepest at {deepest_frost:.2f}m")
    
    # Freeze-up analysis (typically October-April when frost depth increases) - last year only
    freeze_months = [10, 11, 12, 1, 2, 3, 4]
    freeze_data = zero_df_analysis[zero_df_analysis['month'].isin(freeze_months)]
    freeze_data_clean = freeze_data.dropna(subset=['zero_degree_depth'])
    
    if len(freeze_data) > 0:
        total_freeze_days = len(freeze_data)
        frost_freeze_days = len(freeze_data_clean)
        frost_freeze_percentage = (frost_freeze_days / total_freeze_days) * 100
        
        print(f"Freeze season (Oct-Apr): {frost_freeze_days}/{total_freeze_days} days with frost ({frost_freeze_percentage:.1f}%)")
        
        if len(freeze_data_clean) > 0:
            deepest_winter_frost = freeze_data_clean['zero_degree_depth'].min()  # min because negative
            mean_winter_frost = freeze_data_clean['zero_degree_depth'].mean()
            print(f"Maximum winter frost depth: {deepest_winter_frost:.2f}m")
            print(f"Mean winter frost depth: {mean_winter_frost:.2f}m")

else:
    print("No frost depth data available for active layer analysis")
# Simple example of running Arctic-XBeach
# Load modules
from pathlib import Path
import sys
import yaml
import shutil
from main import main, Simulation
import multiprocessing

# Find path
proj_dir = Path(__file__).parent.resolve()
if str(proj_dir) not in sys.path:
    sys.path.insert(0, str(proj_dir))

# Borrowed from run_calibration_2016_auto.py
def update_config_yaml(sim_dir, start_date, end_date):
    config_path = sim_dir / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    # Update start and end date in config
    config['model']['time_start']   = start_date
    config['model']['time_end']     = end_date
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

# Base directory for runs
base_sim_dir = Path(r'd:\Git\thermo-morphological-model\runs\20250828_determine_spinup\base')
run_root_dir = Path(r'd:\Git\thermo-morphological-model\runs\20250828_determine_spinup\20250825_spinup_batch')
run_root_dir.mkdir(exist_ok=True)

# Generate start/end dates for 10 runs
from datetime import datetime, timedelta
run_dates = []
for i in range(10):
    start_year = 2016 - i
    start_date = f"01-01-{start_year:02d}"
    end_date = "01-01-2017"
    run_dates.append((start_date, end_date))


# Multiprocessing for parallel runs
def worker(args):
    idx, start_date, end_date = args
    sim_dir = run_root_dir / f"run{idx:02d}"
    if sim_dir.exists():
        shutil.rmtree(sim_dir)
    shutil.copytree(base_sim_dir, sim_dir)
    update_config_yaml(sim_dir, start_date, end_date)
    print(f"Running simulation {idx:02d}: {start_date} to {end_date}")
    sim = Simulation(str(sim_dir), proj_dir=proj_dir)
    main(sim)
    return idx

if __name__ == "__main__":
    args_list = [(idx, start_date, end_date) for idx, (start_date, end_date) in enumerate(run_dates, 1)]
    with multiprocessing.Pool(processes=10) as pool:
        results = pool.map(worker, args_list)
    print("All simulations completed.")


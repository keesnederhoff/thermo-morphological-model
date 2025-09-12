# Simple example of running Arctic-XBeach
# Load modules
from pathlib import Path
import sys
import yaml
import shutil
from main import main, Simulation
import multiprocessing
import numpy as np

# Find path
proj_dir = Path(__file__).parent.resolve()
if str(proj_dir) not in sys.path:
    sys.path.insert(0, str(proj_dir))

# Base directory for runs
base_sim_dir = Path(r'd:\Git\thermo-morphological-model\runs\20250825_determin_stability\base')
run_root_dir = Path(r'd:\Git\thermo-morphological-model\runs\20250825_determin_stability\20250829_batch_ghostnode2')
run_root_dir.mkdir(exist_ok=True)

# list divisors and count multiples <= 3600
n = 3600
divs = [d for d in range(1, n+1) if n % d == 0]
print("divisor count:", len(divs))
timesteps_wanted = []
for k in range(1, 901):
    if n % k == 0:
        count = n // k
        timesteps_wanted.append(k)
        print(f"k={k}, multiples <=3600: {count} (round)")

# Remove some numbers
for i in sorted([10, 12, 14, 16, 18, 20], reverse=True):
    timesteps_wanted.pop(i)
nsteps_wanted = len(timesteps_wanted)
print(nsteps_wanted)

# Multiprocessing for parallel runs
def update_config_yaml(sim_dir, thermal_timestep):
    config_path = sim_dir / 'config.yaml'
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading YAML {config_path}: {e}")
            raise
    # ensure native Python type (float) to avoid PyYAML representer/constructor issues
    config.setdefault('thermal', {})
    config['thermal']['dt'] = float(thermal_timestep)
    with open(config_path, 'w') as f:
        try:
            yaml.dump(config, f)
        except Exception as e:
            print(f"Error writing YAML {config_path}: {e}")
            raise

# Borrowed from run_calibration_2016_auto.py
def worker(args):
    idx, timestep = args
    sim_dir = run_root_dir / f"run{idx:02d}"
    if sim_dir.exists():
        shutil.rmtree(sim_dir)
    shutil.copytree(base_sim_dir, sim_dir)
    # ensure native Python float when updating config
    update_config_yaml(sim_dir, float(timestep))
    print(f"Running simulation {idx:02d}: thermal_dt = {timestep}s")
    sim = Simulation(str(sim_dir), proj_dir=proj_dir)
    main(sim)
    return idx

if __name__ == "__main__":
    args_list = [(idx, t) for idx, t in enumerate(timesteps_wanted, 1)]
    with multiprocessing.Pool(processes=nsteps_wanted) as pool:
        results = pool.map(worker, args_list)
    print("All simulations completed.")


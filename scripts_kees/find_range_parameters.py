import os
import yaml

base_dir = r'd:\Git\thermo-morphological-model\runs'

# 1. Find all directories in base_dir
dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
configs = {}
# 2. Read all config.yaml files
for d in dirs:
    config_path = os.path.join(d, 'config.yaml')
    if os.path.isfile(config_path):
        with open(config_path, 'r') as f:
            try:
                configs[d] = yaml.safe_load(f)
            except Exception as e:
                print(f"Error reading {config_path}: {e}")

# 3. Compare ranges of values used
# Assuming configs are dicts with numeric values or lists
ranges = {}
for dir_name, config in configs.items():
    print(f"Processing config from {dir_name}: {config}")  # Show config values being read
    thermal = config.get('thermal')
    if not thermal:
        print(f"No 'thermal' section found in {dir_name}")
        continue
    for key, value in thermal.items():
        if isinstance(value, (int, float)):
            ranges.setdefault(key, []).append(value)
        elif isinstance(value, list) and all(isinstance(v, (int, float)) for v in value):
            ranges.setdefault(key, []).extend(value)

print("Ranges of values used across configs:")
for key, values in ranges.items():
    print(f"{key}: min={min(values)}, max={max(values)}, values={set(values)}")
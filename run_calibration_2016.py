# Simple example of running Arctic-XBeach
# Load modules
from pathlib import Path
import sys
from main import main, Simulation

# Find path
proj_dir = Path(__file__).parent.resolve()
if str(proj_dir) not in sys.path:
    sys.path.insert(0, str(proj_dir))

# Run simulation
sim = Simulation("runs/20250822_calibration_runs/run000_cal_gt61_testing/", proj_dir=proj_dir)
main(sim)


from pathlib import Path
import sys

proj_dir = Path(__file__).parent.resolve()
if str(proj_dir) not in sys.path:
    sys.path.insert(0, str(proj_dir))

from main import main, Simulation

sim = Simulation("runs/cal_gt1")
# If you need to set proj_dir, do it like this:
# sim.proj_dir = Path("d:/Git/thermo-morphological-model/")
main(sim)

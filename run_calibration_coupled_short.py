# Simple example of running Arctic-XBeach
# Load modules
from pathlib import Path
import sys
import cProfile
import pstats
from main import main, Simulation

# Find path
proj_dir = Path(__file__).parent.resolve()
if str(proj_dir) not in sys.path:
    sys.path.insert(0, str(proj_dir))

# Profiling the simulation
def run_with_profiling():
    # Run simulation
    sim = Simulation(r'd:\Git\thermo-morphological-model\runs\20250922_coupled_runs\run001_val_per2_2_short', proj_dir=proj_dir)
    main(sim)

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    run_with_profiling()
    profiler.disable()

    # Save and print profiling results
    with open("profile_results.txt", "w") as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.strip_dirs()
        stats.sort_stats("cumulative")  # Sort by cumulative time
        stats.print_stats()

    print("Profiling complete. Results saved to 'profile_results.txt'.")
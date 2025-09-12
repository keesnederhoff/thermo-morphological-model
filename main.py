import os
from pathlib import Path
import sys
import time

from IPython import get_ipython
import numpy as np
import pandas as pd

from utils.model import Simulation
from utils.bathymetry import generate_schematized_bathymetry
from utils.miscellaneous import textbox, datetime_from_timestamp

import argparse, logging, logging.handlers, sys

# --- Simple logger setup (new) ---
logger = logging.getLogger("thermo_model")
logger.setLevel(logging.INFO)

def setup_logger(sim, print_to_screen=True):
    if logger.handlers:
        return
    log_file = os.path.join(sim.cwd, "run.log")
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    if print_to_screen:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
# ---------------------------------

def _setup_logging(level: str, run_id: str, log_file: Path | None):
    root = logging.getLogger()
    root.setLevel(getattr(logging, level))

def main(sim, print_to_screen=True):
    """Run this function to perform a simulation

    Args:
        sim (Simulation): instance of the Simulation class
    """

    setup_logger(sim, print_to_screen)
    # Start time
    t_start = time.time()

    logger.info('Initializing Arctic-XBeach')

    config = sim.config
    logger.debug("Succesfully read configuration")

    # read temporal parameters
    sim.set_temporal_params(
        config.model.time_start,
        config.model.time_end,
        config.model.timestep
        )
    logger.debug("Succesfully set temporal parameters")
        
    # load in forcing data
    sim.load_forcing(
        os.path.join(sim.proj_dir, sim.config.data.forcing_data_path)
    )
    logger.debug("Succesfully loaded forcing")
    
    # load hydrodynamic forcing
    sim.initialize_hydro_forcing(
        os.path.join(sim.proj_dir, sim.config.data.storm_data_path),
        )
    logger.debug("Succesfully loaded hydrodynamic forcing")
    
    # this variable is used to determine if xbeach should be ran for each timestep (not looking at 2% runup yet)
    xb_times = sim.timesteps_with_xbeach_active()
    logger.debug("Succesfully generated xbeach times")
    
    # generate schematized bathymetry
    if sim.config.bathymetry.with_schematized_bathymetry:
        xgr, zgr = generate_schematized_bathymetry(
            bluff_flat_length=sim.config.bathymetry.bluff_flat_length,
        
            bluff_height=sim.config.bathymetry.bluff_height, 
            bluff_slope=sim.config.bathymetry.bluff_slope,
            
            beach_width=sim.config.bathymetry.beach_width, 
            beach_slope=sim.config.bathymetry.beach_slope,
            
            nearshore_max_depth=sim.config.bathymetry.nearshore_max_depth, 
            nearshore_slope=sim.config.bathymetry.nearshore_slope,
            
            offshore_max_depth=sim.config.bathymetry.offshore_max_depth, 
            offshore_slope=sim.config.bathymetry.offshore_slope,
            
            contintental_flat_width=sim.config.bathymetry.continental_flat_width,
            
            with_artificial=sim.config.bathymetry.with_artificial,
            artificial_max_depth=sim.config.bathymetry.artificial_max_depth,
            artificial_slope=sim.config.bathymetry.artificial_slope,
            
            N=sim.config.bathymetry.N,
            artificial_flat=sim.config.bathymetry.artificial_flat
        )
        
        np.savetxt("x.grd", xgr)
        np.savetxt("bed.dep", zgr)
        
        logger.info("Succesfully generated schematized bathymetry")
    
    
    # generate initial grid files and save them
    xgr, zgr, ne_layer = sim.generate_initial_grid(
        nx=sim.config.bathymetry.nx if 'nx' in sim.config.bathymetry.keys() else None,
        bathy_path=sim.config.bathymetry.depfile,
        bathy_grid_path=sim.config.bathymetry.xfile
        )
    logger.debug("Succesfully generated grid")
    
    # initialize xbeach module
    sim.initialize_xbeach_module()
    logger.debug("Succesfully initialized xbeach module")
    
    # initialize first xbeach timestep
    if sim.config.xbeach.with_xbeach:
        sim.xbeach_times[0] = sim.check_xbeach(0)
    else:
        sim.xbeach_times[0] = 0
    
    # initialize thermal model
    sim.initialize_thermal_module()
    logger.debug("Succesfully initialized thermal module")
    
    # initialize solar flux calculator
    if sim.config.thermal.with_solar_flux_calculator:
        sim.initialize_solar_flux_calculator(
            sim.config.model.time_zone_diff,
            angle_min=sim.config.thermal.angle_min,
            angle_max=sim.config.thermal.angle_max,
            delta_angle=sim.config.thermal.delta_angle,
            t_start=sim.config.thermal.t_start,
            t_end=sim.config.thermal.t_end,
            )
    logger.debug("Succesfully initialized solar flux calculator")
    
    # show CFL values (they have already been checked to be below 0.5)
    logger.debug(f"Current maximum CFL {np.max(sim.cfl_matrix):.4f}")

    # loop through (xbeach) timesteps
    logger.info("Starting Arctic-XBeach")
    
    ################################################
    ##                                            ##
    ##            # MAIN LOOP                     ##
    ##                                            ##
    ################################################
    
    last_progress_info = -1  # Track last percentage logged at 5% intervals
    for timestep_id in np.arange(len(sim.T)):
        
        # Count timesteps
        logger.debug(f"Timestep {timestep_id+1}/{len(sim.T)}")

        # Track progress
        if timestep_id > 0:
            elapsed = time.time() - t_start
            avg_step_time = elapsed / timestep_id
            remaining_steps = len(sim.T) - (timestep_id + 1)
            eta_seconds = avg_step_time * remaining_steps
            progress_pct = int(((timestep_id + 1) / len(sim.T)) * 100)
            if progress_pct % 5 == 0 and progress_pct != last_progress_info:
                eta_hours = eta_seconds / 3600
                if eta_hours < 1:
                    logger.info(f"Progress {progress_pct}% | avg_step={avg_step_time:.1f}s | {sim.timestamps[timestep_id]} | ETA ~ {eta_hours * 60:.2f}min")
                else:
                    logger.info(f"Progress {progress_pct}% | avg_step={avg_step_time:.1f}s | {sim.timestamps[timestep_id]} | ETA ~ {eta_hours:.2f}h")
                last_progress_info = progress_pct

        # write output variables to output file every output interval
        if timestep_id in sim.temp_output_ids:
            sim.write_output(timestep_id, t_start)
            logger.debug("Succesfully generated output")

        # used for validation of the temperature model
        if 'save_ground_temp_layers' in sim.config.output.keys():
            
            sim.save_ground_temp_layers_in_memory(
                timestep_id, 
                layers=sim.config.output.save_ground_temp_layers,
                heat_fluxes=sim.config.output.heat_fluxes,
                write=(timestep_id == np.arange(len(sim.T))[-1]),
                )
        
        # check whether to run XBeach or not for this timestep
        if sim.config.xbeach.with_xbeach and not all(np.abs(sim.thaw_depth) < 0.001):
            sim.xbeach_times[timestep_id] = sim.check_xbeach(timestep_id)
        else:
            sim.xbeach_times[timestep_id] = 0
            
        # check if xbeach is enabled for current timestep
        if sim.xbeach_times[timestep_id] and sim.config.xbeach.with_xbeach:
            
            # export current thaw depth to a file
            sim.write_ne_layer()
                        
             # generate params.txt file 
            sim.xbeach_setup(timestep_id)
            
            logger.info(f"Starting xbeach for timestep {sim.timestamps[timestep_id]}")
            
            # call xbeach (could include batch file?)
            run_succesful = sim.start_xbeach(
                os.path.join(sim.proj_dir, Path("xbeach/XBeach_1.24.6057_Halloween_win64_netcdf/xbeach.exe")),
                sim.cwd
            )
            
            try:
                if run_succesful:
                    logger.info(f"succesfully ran xbeach for timestep {sim.timestamps[timestep_id]} to {sim.timestamps[timestep_id+1]}")
                else:
                    logger.error(f"failed to run xbeach for timestep {sim.timestamps[timestep_id]} to {sim.timestamps[timestep_id+1]}")
            except IndexError:
                logger.info(f"xbeach ran succesfully for final timestep timestep ({sim.timestamps[timestep_id]})")
                        
            # if this was one of the first storms, the output is of higher temporal resolution and it is saved in the results folder
            if sim.copy_this_xb_output:
                sim.copy_xb_output_to_result_dir(fp_xbeach_output="xboutput.nc")
                logger.info("succesfully generated high resolution storm output")
                
            # check if xbeach should be ran for the next timestep (if so, the x-grid doesn't update since the same grid is necessary for the hotstart feature)
            if sim.config.xbeach.with_xbeach and timestep_id + 1 < len(sim.T):
                sim.xbeach_times[timestep_id + 1] = sim.check_xbeach(timestep_id + 1)
            else:
                sim.xbeach_times[timestep_id + 1] = 0
                        
            # copy updated morphology to thermal module, and update the thermal grid with the new morphology
            sim.update_grid(timestep_id, fp_xbeach_output="xboutput.nc")  # this thing right here is pretty slow (TO BE CHANGED)
            
            # copy all xb output to the results folder 
            # # (this is necessary as the xb output in the run folder will be overwritten after the next xb iteration)
            sim.dump_xb_output(timestep_id)
        
        # loop through thermal subgrid timestep
        for subgrid_timestep_id in np.arange(0, config.model.timestep * 3600, config.thermal.dt):
            sim.thermal_update(timestep_id, subgrid_timestep_id)
            
        # calculate the current thaw depth
        sim.find_thaw_depth()
            
    # write xbeach timesteps
    sim.write_xb_timesteps()
    logger.info('Arctic-XBeach Finished!')
    logger.info(f"Simulation started at: {datetime_from_timestamp(t_start)}")
    logger.info(f"Simulation finished at: {datetime_from_timestamp(time.time())}")
    logger.info(f"Total simulation time: {(time.time() - t_start) / 1:.1f} seconds")
    logger.info(f"Total simulation time: {(time.time() - t_start) / 60:.1f} minutes")
    logger.info(f"Total simulation time: {(time.time() - t_start) / 3600:.1f} hours")
    # Add this line to ensure NetCDF is closed and flushed
    if sim.nc_writer is not None:
        sim.nc_writer.close()

    return sim.xgr, sim.zgr

if __name__ == '__main__':
    
    ##| To run script from Terminal:
    ##| cd C:\Users\bruij_kn\OneDrive - Stichting Deltares\Documents\GitHub\thermo-morphological-model
    ##| python main.py run_id
    
    # Parser
    parser = argparse.ArgumentParser(description="Run the Arctic-XBeach simulation.")
    parser.add_argument("runid", help="The run ID for the simulation.")
    parser.add_argument("--no-screen-log", action="store_true", help="Disable logging to the screen.")
    args = parser.parse_args()

    # reduce ipython cache size to free up memory
    ipython = get_ipython()
    if ipython:
        ipython.Completer.cache_size = 5

    # set the 'runid' to the model run that you would like to perform
    runid = sys.argv[1]

    # new: ensure project root (folder containing this main.py) on sys.path
    proj_dir = Path(__file__).parent.resolve()
    if str(proj_dir) not in sys.path:
        sys.path.insert(0, str(proj_dir))

    # initialize simulation with explicit proj_dir
    sim = Simulation(runid, proj_dir=proj_dir)

    # Pass the argument to main
    main(sim, print_to_screen=not args.no_screen_log)

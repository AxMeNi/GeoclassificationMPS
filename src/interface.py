# -*- coding:utf-8 -*-
__projet__ = "GeoclassificationMPS"
__nom_fichier__ = "interface"
__author__ = "MENGELLE Axel"
__date__ = "juillet 2024"

from launcher import *
from data_treatment import *

import numpy as np
import pandas as pd


# ╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ INTERFACE FOR PROGRAMMING A COMBINED DEESSE AND LOOPUI SIMULATION                                                  ║
# ╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

def get_simulation_info():


    
    ##################### LOCATIONS OF THE CSV DATA FILE #####################
    
    csv_file_path = r"C:\Users\Axel (Travail)\Documents\ENSG\CET\GeoclassificationMPS\data\data_csv.csv"
    
    # Expected CSV File Format (Columns are separataed by ";"):
    #
    # The CSV file should contain information about variables to be loaded as numpy arrays.
    # Each line in the CSV file should be formatted as follows:
    #
    # Column 1: var_name      - The name of the variable.
    # Column 2: categ_conti   - The type of variable, either "categorical" or "continuous".
    # Column 3: sim_aux       - Indicates whether the variable is an auxiliary ("aux") or simulated ("sim") variable.
    # Column 4: path          - The full file path to the .npy file containing the numpy array data. (format .npy and the array must be in 2 dimensions)
    #
    # Example (first line contains headers):
    #
    # var_name;categ_conti;sim_aux;path
    # grid_geo;categorical;aux;C:\path\to\grid_geo.npy
    # grid_grv;continuous;sim;C:\path\to\grid_grv.npy
    # grid_lmp;continuous;sim;C:\path\to\grid_lmp.npy
    # grid_mag;continuous;sim;C:\path\to\grid_mag.npy
         
    ##################### RANDOM PARAMETERS #####################

    seed = 12345

    ##################### NOVALUE #####################

    novalue = -9999999

    ##################### TRAINING IMAGE PARAMETERS #####################

    ti_pct_area = 33  
    ti_ndisks = 1  # 
    ti_realid = 1  # 
    xycv = False  # Flag for cross-validation
    

    

    ##################### DEESSE SIMULATION PARAMETERS #####################
    

    nn = 12  # Number of neighboring nodes
    dt = 0.1  # Distance threshold
    ms = 0.25  # Maximum scan fraction
    numberofmpsrealizations = 1  # Number of Deesse realizations
    nthreads = 1  # Number of threads for parallel processing

    ##################### LAUNCHING PARAMETERS #####################

    configs = [
        (33, 1, 1, 10, 4, True, True), # Percentage area of the TI, Number of disks in the TI, Realization ID for the TI
        (33, 1, 1, 10, 4, True, False),
        (33, 1, 1, 10, 4, False, False),
        (33, 1, 1, 10, 4, False, True)
    ]

    ##################### COLORMAP PARAMETERS #####################

    cm = plt.get_cmap('tab20')
    myclrs = np.asarray(cm.colors)[[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11], :]
    n_bin = 11
    cmap_name = 'my_tab20'
    mycmap = LinearSegmentedColormap.from_list(cmap_name, myclrs, N=n_bin)
    ticmap = LinearSegmentedColormap.from_list('ticmap', np.vstack(([0, 0, 0], myclrs)), N=n_bin + 1)

    ##################### SHORTEN THE SIMULATION #####################

    shorten = False
    
    ##################### PICKING SIM AND AUX VAR #####################
    
    simulated_var, auxiliary_var, names_var, types_var = create_auxiliary_and_simulated_var(csv_file_path)
    simulated_var, auxiliary_var = check_variables(simulated_var, auxiliary_var, names_var, types_var, novalue)
    
    ##################### GRID DIMENSIONS #####################
    
    SGDimIsDataDim = True #True if the simulation grid is the size of the data files
    
    ##################### PICKING SIM AND AUX VAR #####################
    
    nr, nc = get_sim_grid_dimensions(simulated_var)

    return simulated_var, auxiliary_var, types_var, names_var, nn, dt, ms, numberofmpsrealizations, nthreads, configs
            

# ╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ LAUNCH THE SIMULATIONS                                                                                             ║
# ╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


def execute_shorter_program(ti_pct_area, ti_ndisks, ti_realid, mps_nreal, nthreads, geolcd, timesleep=0, verb=True):
    """
    Function to execute the main program with given parameters.
    """

    # Generate the TI mask
    grid_msk = gen_ti_mask(nx, ny, ti_pct_area, ti_ndisks, myseed + ti_realid)

    # Build the TI with given parameters
    geocodes, ngeocodes, tiMissingGeol, cond_data = build_ti(
        grid_msk, ti_ndisks, ti_pct_area, ti_realid, geolcd)

    # Run DEESSE simulation
    deesse_output = run_deesse(tiMissingGeol, mps_nreal, nn, dt, ms, seed, nthreads, geolcd, cond_data)

    # Retrieve the simulation results
    sim = deesse_output['sim']

    # Perform statistics on the realizations
    # Gather all realizations into one image
    all_sim = gn.img.gatherImages(sim)  # all_sim is one image with nreal variables
    # Compute the pixel-wise proportion for the given categories
    all_sim_stats = gn.img.imageCategProp(all_sim, geocodes)

    # Initialize the realizations array with NaN values
    realizations = np.ones((ny, nx, mps_nreal)) * np.nan
    for i in range(mps_nreal):
        ix = i * tiMissingGeol.nv
        realizations[:, :, i] = all_sim.val[ix, 0, :, :]  # Assign the simulation values to the realizations array

    # Create a title for the plot
    addtitle = f'geolcd: {geolcd} - xycv: {xycv}'
    # Plot the realizations and reference grid
    plot_real_and_ref(realizations, reference=grid_geo, mask=1 - grid_msk, nrealmax=mps_nreal, addtitle=addtitle)

    # Print message indicating completion
    print("Simulation and plotting complete.")

    # Additional analysis and output can be added here



def launch_simulation():
    
    for config in configs:
        ti_pct_area, ti_ndisks, ti_realid, mps_nreal, nthreads, geolcd, xycv = config
        print(f"Running configuration: geolcd={geolcd}, xycv={xycv}")
        
        if shorten :
            execute_program(ti_pct_area, ti_ndisks, ti_realid, mps_nreal, nthreads, geolcd)
        else :
            launcher(simulated_var = simulated_var_modified, 
                auxiliary_var = auxiliary_var_modified, 
                var_names = var_names, 
                var_types = var_types, 
                ti_pct_area = ti_pct_area, 
                ti_ndisks = ti_ndisks, 
                ti_realid = ti_realid, 
                mps_nreal = mps_nreal, 
                nthreads = nthreads, 
                geolcd = True, 
                timesleep = 0, 
                verb = False,
                addtitle = 'addtitle',
                seed = seed)
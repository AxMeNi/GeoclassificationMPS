# -*- coding:utf-8 -*-
__projet__ = "GeoclassificationMPS"
__nom_fichier__ = "interface"
__author__ = "MENGELLE Axel"
__date__ = "juillet 2024"

from launcher import *
from data_treatment import *
from matplotlib.colors import LinearSegmentedColormap

import numpy as np
import pandas as pd

#################################################################
#     ##  ### ##      ##    ##    ###     ##    ####   ##    ####
### #### # ## #### ##### ##### ### ## ###### ### ## ##### #######
### #### ## # #### #####   ###  # ###   ####     ## #####   #####
### #### ###  #### ##### ##### # #### ###### ### ## ##### #######
#     ## #### #### #####    ## ## ### ###### ### ###   ##    ####
#################################################################

# ╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ INTERFACE FOR PROGRAMMING A COMBINED DEESSE AND LOOPUI SIMULATION                                                  ║
# ╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

def get_simulation_info():
    
    ##################### LOCATIONS OF THE CSV DATA FILE #####################
    
    csv_file_path = r"C:\Users\Axel (Travail)\Documents\ENSG\CET\GeoclassificationMPS\test\data_csv.csv"
    
    # Expected CSV File Format (Columns are separataed by ";"):
    #
    # The CSV file should contain information about variables to be loaded as numpy arrays.
    # Each line in the CSV file should be formatted as follows:
    #
    # Column 1: var_name      - The name of the variable.
    # Column 2: categ_conti   - The type of variable, either "categorical" or "continuous".
    # Column 3: nature        - Indicates whether the variable is asimulated ("sim") variable, auxiliary descriptive ("auxTI") variable, auxiliary conditioning ("auxSG"), or conditioning ("cond") variable.
    # Column 4: path          - The full file path to the .npy file containing the numpy array data. (format .npy and the array must be in 2 dimensions)
    #
    # Example (first line contains headers):
    #
    # var_name;categ_conti;nature;path
    # grid_geo;categorical;sim;C:\path\to\grid_geo_sim.npy
    # grid_grv;continuous;sim;C:\path\to\grid_grv.npy
    # grid_lmp;continuous;sim;C:\path\to\grid_lmp.npy
    # grid_mag;continuous;sim;C:\path\to\grid_mag.npy
    # grid_geo;categorical;condIm;C:\path\to\grid_geo_cond.npy
    # auxiliary;continuous;auxTI;C:\path\to\auxTI.npy
    # auxiliary;continuous;auxSG;C:\path\to\auxSG.npy
         
    ##################### RANDOM PARAMETERS #####################

    seed = 852

    ##################### NOVALUE #####################

    novalue = -9999999

    ##################### TRAINING IMAGE PARAMETERS #####################
    
    ti_methods = ["DependentCircles", "DependentSquares", "IndependentSquares", "ReducedTiCd"] #List of method
    ti_pct_area = None 
    ti_nshapes = 1   
    
    pct_ti_sg_overlap=50  
    pct_sg=10
    pct_ti=30
    cc_sg=None
    rr_sg=None
    cc_ti=None
    rr_ti=None
    
    ##################### DEESSE SIMULATION PARAMETERS #####################

    nn = 12  # Number of neighboring nodes
    dt = 0.1  # Distance threshold
    ms = 0.25  # Maximum scan fraction
    numberofmpsrealizations = 1  # Number of Deesse realizations
    nthreads = 1  # Number of threads for parallel processing

    ##################### COLORMAP PARAMETERS #####################

    cm = plt.get_cmap('tab20')
    myclrs = np.asarray(cm.colors)[[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11], :]
    n_bin = 11
    cmap_name = 'my_tab20'
    mycmap = LinearSegmentedColormap.from_list(cmap_name, myclrs, N=n_bin)
    ticmap = LinearSegmentedColormap.from_list('ticmap', np.vstack(([0, 0, 0], myclrs)), N=n_bin + 1)
    print(type(mycmap),type(ticmap), type(cm))
    ##################### SHORTEN THE SIMULATION #####################

    shorten = False
    
    ##################### PICKING SIM AND AUX VAR #####################
    
    check_ti_methods(ti_methods)
    
    sim_var, auxTI_var, auxSG_var, condIm_var, names_var, types_var = create_variables(csv_file_path)
    sim_var, auxTI_var, auxSG_var, condIm_var = check_variables(sim_var, auxTI_var, auxSG_var, condIm_var, names_var, types_var, novalue)
    nvar = count_variables(names_var)
    
    nr, nc = get_sim_grid_dimensions(sim_var)
   
    print(f"Data dimension : \n \t >> Number of rows : {nr} \n \t >> Number of columns : {nc}")
       
    return seed, \
            ti_methods, \
            ti_pct_area, ti_nshapes, \
            pct_ti_sg_overlap, pct_sg, pct_ti, cc_sg, rr_sg, cc_ti, rr_ti, \
            nn, dt, ms, numberofmpsrealizations, nthreads, \
            cm, myclrs, n_bin, cmap_name, mycmap, ticmap, \
            shorten, \
            sim_var, auxTI_var, auxSG_var, condIm_var, names_var, types_var, \
            nr, nc 
            

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



def launch_simulation(seed, 
                    ti_methods, 
                    ti_pct_area, ti_nshapes,
                    pct_ti_sg_overlap, pct_sg, pct_ti, cc_sg, rr_sg, cc_ti, rr_ti,
                    nn, dt, ms, numberofmpsrealizations, nthreads,
                    cm, myclrs, n_bin, cmap_name, mycmap, ticmap,
                    shorten,
                    sim_var, auxTI_var, auxSG_var, condIm_var, names_var, types_var,
                    nr, nc):
    
    
    if shorten :
        execute_program(ti_pct_area, ti_ndisks, ti_realid, mps_nreal, nthreads)
    else :
        launcher(seed, 
                ti_methods, 
                ti_pct_area, ti_shapes,
                pct_ti_sg_overlap, pct_sg, pct_ti, cc_sg, rr_sg, cc_ti, rr_ti,
                nn, dt, ms, numberofmpsrealizations, nthreads,
                cm, myclrs, n_bin, cmap_name, mycmap, ticmap,
                sim_var, auxTI_var, auxSG_var, condIm_var, names_var, types_var,
                nr, nc)
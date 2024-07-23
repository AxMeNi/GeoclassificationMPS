# -*- coding:utf-8 -*-
__projet__ = "GeoclassificationMPS"
__nom_fichier__ = "interface"
__author__ = "MENGELLE Axel"
__date__ = "juillet 2024"

from launcher import *
from data_treatment import *

import numpy as np


# ╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ INTERFACE FOR PROGRAMMING A COMBINED DEESSE AND LOOPUI SIMULATION                                                  ║
# ╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

def get_simulation_info():

    ##################### DATA DEFINITION FOR THE SIMULATION #####################
    varAval11, varBval11 = 2, 3.142
    varAval12, varBval12 = 2, 1.987
    varAval13, varBval13 = 3, 8.654

    varAval21, varBval21 = 3, 2.345
    varAval22, varBval22 = 5, 4.321
    varAval23, varBval23 = 5, 9.876
    
    # Auxiliary variable values (example values)
    var1val11, var2val11 = 0.123, 1.456
    var1val12, var2val12 = 2.345, 3.678
    var1val13, var2val13 = 4.567, 5.890

    var1val21, var2val21 = 6.789, 7.012
    var1val22, var2val22 = 8.234, 9.345
    var1val23, var2val23 = 0.456, 1.234
    
    names_var = [
        ["varA", "varB"],
        ["aux1", "aux2"]
    ]

    # Type must be set as "continuous" or "categorical".
    types_var = [
        ["categorical", "continuous"],
        ["continuous", "continuous"]
    ]
    
    #Simulated variables can be partially informed
    sim_var = list((np.array([[varAval11, varAval12, varAval13],
                              [varAval21, varAval22, varAval23]],dtype=int),
                    
                    np.array([[varBval11, varBval12, varBval13],
                              [varBval21, varBval22, varBval23]],dtype=float)
                  ))
    #Auxiliary variables must be fully informed
    aux_var = list((np.array([[var1val11, var1val12, var1val13], 
                              [var1val21, var1val22, var1val23]],dtype=float),
                              
                    np.array([[var2val11, var2val12, var2val13], 
                              [var2val21, var2val22, var2val23]],dtype=float)
                  ))


    ##################### LOCATIONS OF THE LOADING AND SAVING FILES #####################

    # Name of the pre-processed data file
    suffix = "-simple"  # "", "-simple", "-very-simple"
    data_filename = "mt-isa-data" + suffix + ".pickle"

    # Paths to the saving FOLDERS
    path2ti = "./ti/"  # Training images path
    path2cd = "./cd/"  # Conditioning data path
    path2real = "./mpsReal/"  # Realizations path
    path2log = "./log/"  # Logging info path
    path2ind = "./ind/"  # Index data path

    path2data = "C:/Users/Axel (Travail)/Documents/ENSG/CET/GeoclassificationMPS/Missing-Data/data/"
    suffix = "-simple"
    picklefn = "mt-isa-data" + suffix + ".pickle"
    pickledestination = path2data + picklefn
    
    # Load pre-processed data from pickle file
    with open(pickledestination, 'rb') as f:
        [_, grid_geo, grid_lmp, grid_mag,
         grid_grv, grid_ext, vec_x, vec_y
         ] = pickle.load(f)
         
    # create_directories(path2ti,path2cd,path2real,path2log,path2ind)


    ##################### RANDOM PARAMETERS #####################

    seed = 12345

    ##################### NOVALUE #####################

    novalue = -9999999

    ##################### TRAINING IMAGE PARAMETERS #####################

    ti_pct_area = 33  
    ti_ndisks = 1  # 
    ti_realid = 1  # 
    xycv = False  # Flag for cross-validation

    ##################### CONDITIONING DATA PARAMETERS #####################


    
    ##################### PICKING SIM AND AUX VAR #####################
    
    simulated_var, auxiliary_var = create_sim_and_aux(names_var, sim_var, aux_var)
    check_variables(simulated_var, auxiliary_var, names_var, types_var, novalue)

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
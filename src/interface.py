# -*- coding:utf-8 -*-
__projet__ = "GeoclassificationMPS"
__nom_fichier__ = "interface"
__author__ = "MENGELLE Axel"
__date__ = "juillet 2024"

from launcher import *
from data_treatment import *
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime

import numpy as np
import pandas as pd


#################################################################
#     ##  ### ##      ##    ##    ###     ###   ####   ##    ####
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
    
    #\group\ses001\amengelle\
    #C:\Users\00115212\Documents\
    
    csv_file_path = "/group/ses001/amengelle/GeoclassificationMPS/data/data_csv.csv"
    
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
    
    #"DependentCircles", "DependentSquares", "IndependentSquares", "ReducedTiSg"
    ti_methods = ["DependentSquares"] #List of methods
    
    #Parameters for "DependentCircles", "DependentSquares", "IndependentSquares"
    ti_pct_area = 55
    ti_nshapes = 2 
    
    #Parameters for "ReducedTiSg"
    pct_ti_sg_overlap=50  
    pct_sg=30
    pct_ti=70
    cc_sg=None
    rr_sg=None
    cc_ti=None
    rr_ti=None
    
    #Number of random TI and CD sets to generate a simulation with
    nRandomTICDsets = 1
    
    ##################### DEESSE SIMULATION PARAMETERS #####################

    nn = 24  # Number of neighboring nodes
    dt = 0.1  # Distance threshold
    ms = 0.25  # Maximum scan fraction
    numberofmpsrealizations = 5  # Number of Deesse realizations
    nthreads = 4  # Number of threads for parallel processing
    
    ##################### OUTPUT PARAMETERS #####################
    
    #---- To turn On or Off the saving of the output ----#
    saveOutput = True
    
    output_directory = "/group/ses001/amengelle/GeoclassificationMPS/output"
    
    deesse_output_folder = "deesse_output"
    prefix_deesse_output = "simulation"
    
    plot_output_folder = "variability"
    prefix_histogram_disimilarity = "jensen_shannon_divergence"
    prefix_entropy = "entropy"
    prefix_simvar_histograms = "histograms"
    prefix_topological_adjacency = "topological_adjacency"
    prefix_proportions = "proportions"
    path = "/group/ses001/amengelle/GeoclassificationMPS/data/grid_geo.npy"
    reference_var = np.load(r"/group/ses001/amengelle/GeoclassificationMPS/data/grid_geo.npy")
   
   ##################### SHORTEN THE SIMULATION #####################

    shorten = False
    
    ##################### PICKING SIM AND AUX VAR #####################
    
    check_ti_methods(ti_methods)
    
    sim_var, auxTI_var, auxSG_var, condIm_var, names_var, types_var, outputVarFlag = create_variables(csv_file_path)
    sim_var, auxTI_var, auxSG_var, condIm_var = check_variables(sim_var, auxTI_var, auxSG_var, condIm_var, names_var, types_var, novalue)
    nvar = count_variables(names_var)
    
    nr, nc = get_sim_grid_dimensions(sim_var)

    params = {
        'seed': seed,
        'csv_file_path': csv_file_path,
        'novalue': novalue,
        'sim_var': list(sim_var.keys()),
        'auxTI_var': list(auxTI_var.keys()),
        'auxSG_var': list(auxSG_var.keys()),
        'condIm_var':list(condIm_var.keys()),
        'ti_methods': ti_methods,
        'ti_pct_area': ti_pct_area,
        'ti_nshapes': ti_nshapes,
        'pct_ti_sg_overlap': pct_ti_sg_overlap,
        'pct_sg': pct_sg,
        'pct_ti': pct_ti,
        'cc_sg': cc_sg,
        'rr_sg': rr_sg,
        'cc_ti': cc_ti,
        'rr_ti': rr_ti,
        'nRandomTICDsets': nRandomTICDsets,
        'n_neighbouring_nodes': nn,
        'distance_threshold': dt,
        'max_scan_fraction': ms,
        'n_mps_realizations': numberofmpsrealizations,
        'n_threads': nthreads,
        'saveOutput': saveOutput,
        'output_directory': output_directory,
        'deesse_output_folder': deesse_output_folder,
        'prefix_deesse_output': prefix_deesse_output,
        'plot_output_folder': plot_output_folder,
        'prefix_histogram_disimilarity': prefix_histogram_disimilarity,
        'prefix_entropy': prefix_entropy,
        'prefix_simvar_histograms': prefix_simvar_histograms,
        'prefix_topological_adjacency': prefix_topological_adjacency,
        'prefix_proportions': prefix_proportions,
        'reference_var': reference_var,        
        }
    
    return params, \
            shorten, \
            nvar, sim_var, auxTI_var, auxSG_var, condIm_var, names_var, types_var, outputVarFlag, \
            nr, nc 
            

# ╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ LAUNCH THE SIMULATIONS                                                                                             ║
# ╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


def execute_shorter_program(ti_pct_area, ti_ndisks, ti_realid, mps_nreal, nthreads, geolcd, timesleep=0, verb=True):
    return


def launch_simulation(params,
                    shorten,
                    nvar, sim_var, auxTI_var, auxSG_var, condIm_var, names_var, types_var, outputVarFlag,
                    nr, nc,
                    verbose):
    """
    """    
    if shorten :
        execute_shorter_program()
    else :
        launcher(params,
                nvar, sim_var, auxTI_var, auxSG_var, condIm_var, names_var, types_var, outputVarFlag,
                nr, nc, 
                verbose)


def run_simulation(verbose):
    """
    """
    if verbose :
        print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S:%f)') + " <> RETRIEVING SIMULATION INFORMATION")
    params, \
    shorten, \
    nvar, sim_var, auxTI_var, auxSG_var, condIm_var, names_var, types_var, outputVarFlag, \
    nr, nc = get_simulation_info()
    if verbose :
        print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S:%f)') + " <> SIMULATION INFORMATION RETRIEVED")
    
    launch_simulation(params,
                        shorten,
                        nvar, sim_var, auxTI_var, auxSG_var, condIm_var, names_var, types_var, outputVarFlag,
                        nr, nc,
                        verbose)

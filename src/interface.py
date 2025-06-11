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

import argparse



#################################################################
#     ##  ### ##      ##    ##    ###     ###   ####   ##    ####
### #### # ## #### ##### ##### ### ## ###### ### ## ##### #######
### #### ## # #### #####   ###  # ###   ####     ## #####   #####
### #### ###  #### ##### ##### # #### ###### ### ## ##### #######
#     ## #### #### #####    ## #   ## ###### ### ###   ##    ####
#################################################################



# ╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ INTERFACE FOR PROGRAMMING A COMBINED DEESSE AND LOOPUI SIMULATION                                                  ║
# ╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

def get_simulation_info(arg_seed = None, arg_n_ti = None, arg_ti_pct_area = None, arg_num_shape = None, arg_aux_vars = None):
    
    ##################### LOCATIONS OF THE CSV DATA FILE #####################
    
    #\group\ses001\amengelle\
    #C:\Users\00115212\Documents\
    
    csv_file_path = "/group/ses001/amengelle/GeoclassificationMPS/data/data_csv.csv"
    
    # Expected CSV File Format (Columns are separataed by ","):
    #
    # The CSV file should contain information about variables to be loaded as numpy arrays.
    # Each line in the CSV file should be formatted as follows:
    #
    # Column 1: var_name      - The name of the variable.
    # Column 2: categ_conti   - The type of variable, either "categorical" or "continuous".
    # Column 3: nature        - Indicates whether the variable is a simulated ("sim") variable, auxiliary descriptive ("auxTI") variable, auxiliary conditioning ("auxSG"), or conditioning image ("condIm") variable.
    # Column 4: path          - The full file path to the .npy file containing the numpy array data. (format .npy and the array must be in 2 dimensions)
    # Column 5: grid          - The grid where the varaible will be loaded ("TI" or "SG")
    #
    # Example (first line contains headers):
    #
    # var_name,categ_conti,nature,path,grid
    # grid_geo,categorical,sim,C:\path\to\grid_geo_sim.npy,TI
    # grid_grv,continuous,auxTI,C:\path\to\grid_grv.npy,TI
    # grid_grv,continuous,auxSG,C:\path\to\grid_grv.npy,SG
    # grid_lmp,continuous,auxTI,C:\path\to\grid_lmp.npy,TI
    # grid_lmp,continuous,auxSG,C:\path\to\grid_lmp.npy,SG
    # grid_mag,continuous,auxTI,C:\path\to\grid_mag.npy,TI
    # grid_mag,continuous,auxSG,C:\path\to\grid_mag.npy,SG
    # grid_geo,categorical,condIm,C:\path\to\grid_geo_cond.npy,SG
    # auxiliary,continuous,auxTI,C:\path\to\auxTI.npy,TI
    # auxiliary,continuous,auxSG,C:\path\to\auxSG.npy,SG
         
    ##################### RANDOM PARAMETERS #####################

    seed = arg_seed

    ##################### NOVALUE #####################

    novalue = -9999999

    ##################### TRAINING IMAGE PARAMETERS #####################
    
    # The available methods are :
    # "DependentCircles", "DependentSquares", "IndependentSquares", "ReducedTiSg"
    ti_methods = ["DependentSquares"] #List of methods
    
    #Parameters for "DependentCircles", "DependentSquares", "IndependentSquares"
    ti_pct_area = arg_ti_pct_area
    ti_nshapes = arg_num_shape 
    
    #Parameters for "ReducedTiSg"
    pct_ti_sg_overlap=50  
    pct_sg=30
    pct_ti=70
    cc_sg=None #Number of columns of the simulation grid
    rr_sg=None #Number of rows of the simulation grid
    cc_ti=None #Number of columns of the training image
    rr_ti=None #Number of rows of the training image
    
    #Number of random TI and CD sets to generate a simulation with
    nRandomTICDsets = arg_n_ti
    
    ##################### DEESSE SIMULATION PARAMETERS #####################

    nn = 24  # Number of neighboring nodes
    dt = 0.1  # Distance threshold
    ms = 0.25  # Maximum scan fraction
    numberofmpsrealizations = 15  # Number of Deesse realizations
    nthreads = 4  # Number of threads for parallel processing
    
    ##################### OUTPUT PARAMETERS #####################
    
    #---- To turn On or Off the saving of the output ----#
    saveOutput = True #Only for the DeeSse Output
    saveIndicators = True #For the indicators and the standard deviation of the indicators
    
    output_directory = "/group/ses001/amengelle/GeoclassificationMPS/output"
    
    deesse_output_folder = "deesse_output"
    prefix_deesse_output = "simulation"
    
    plot_output_folder = "variability"
    prefix_histogram_dissimilarity = "jensen_shannon_divergence"
    prefix_entropy = "entropy"
    prefix_simvar_histograms = "histograms"
    prefix_topological_adjacency = "topological_adjacency"
    prefix_proportions = "proportions"
    prefix_std_deviation = "std_deviation"
    reference_var = np.load(r"/group/ses001/amengelle/GeoclassificationMPS/data/grid_geo.npy")
    
    ##################### SHORTEN THE SIMULATION #####################

    shorten = False
    
    ##################### PICKING SIM AND AUX VAR #####################
    
    check_ti_methods(ti_methods)
    
    sim_var, auxTI_var_temp, auxSG_var_temp, condIm_var, names_var, types_var, outputVarFlag = create_variables(csv_file_path)
    sim_var, auxTI_var_temp, auxSG_var_temp, condIm_var = check_variables(sim_var, auxTI_var_temp, auxSG_var_temp, condIm_var, names_var, types_var, novalue)
    
    ############################################################################################################
    ### MOVE THAT INTO A FUNCTION ################################################################################
    ############################################################################################################
    auxTI_var = {key: value for key, value in auxTI_var_temp.items() if key in arg_aux_vars}
    auxSG_var = {key: value for key, value in auxSG_var_temp.items() if key in arg_aux_vars}
    outputVarFlag = {key: value for key, value in outputVarFlag.items() if key in arg_aux_vars}
    outputVarFlag["grid_geo"]=True
    names_var = [["grid_geo"],arg_aux_vars,arg_aux_vars,[]]
    types_var[1], types_var[2] = types_var[1][:len(arg_aux_vars)], types_var[2][:len(arg_aux_vars)]
    ############################################################################################################
    
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
        'prefix_histogram_dissimilarity': prefix_histogram_dissimilarity,
        'prefix_entropy': prefix_entropy,
        'prefix_simvar_histograms': prefix_simvar_histograms,
        'prefix_topological_adjacency': prefix_topological_adjacency,
        'prefix_proportions': prefix_proportions,
        'prefix_std_deviation' : prefix_std_deviation,
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


def run_simulation(verbose, arg_seed, arg_n_ti, arg_ti_pct_area, arg_num_shape, arg_aux_vars):
    """
    """
    if verbose :
        print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S:%f)') + " <> RETRIEVING SIMULATION INFORMATION")
    params, \
    shorten, \
    nvar, sim_var, auxTI_var, auxSG_var, condIm_var, names_var, types_var, outputVarFlag, \
    nr, nc = get_simulation_info(arg_seed, arg_n_ti, arg_ti_pct_area, arg_num_shape, arg_aux_vars)
    if verbose :
        print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S:%f)') + " <> SIMULATION INFORMATION RETRIEVED")
    
    launch_simulation(params,
                        shorten,
                        nvar, sim_var, auxTI_var, auxSG_var, condIm_var, names_var, types_var, outputVarFlag,
                        nr, nc,
                        verbose)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Geoclassification MPS")
    parser.add_argument('--seed', type=int, required=True, help="Random seed for the simulation")
    parser.add_argument('--n_ti', type=int, required=True, help="Number of Training Images")
    parser.add_argument('--ti_pct_area', type=int, required=True, help="Percentage of training image area to use")
    parser.add_argument('--num_shape', type=int, required=True, help="Number of shapes for the croping")
    parser.add_argument("--aux_vars", type=str, required=True, help="Comma-separated auxiliary variables")
    args = parser.parse_args()
    aux_vars = args.aux_vars.split(',')
    run_simulation(True, args.seed, args.n_ti, args.ti_pct_area, args.num_shape, aux_vars)

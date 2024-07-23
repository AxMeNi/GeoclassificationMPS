# -*- coding:utf-8 -*-
__projet__ = "GeoclassificationMPS"
__nom_fichier__ = "tests"
__author__ = "MENGELLE Axel"
__date__ = "juillet 2024"


from ti_generation import *
from data_treatment import *
from interface import get_simulation_info

import matplotlib.pyplot as plt
import pickle  # For loading pickled data

##################################### TEST INTERFACE.PY

def test_get_simulation_info():
     simulated_var, auxiliary_var, types_var, names_var, nn, dt, ms, numberofmpsrealizations, nthreads, config = get_simulation_info()

def test_check_variables():
    """
    Test the check_variables function with simulated and real datasets.
    
    This function creates two sets of test variables: one set with hardcoded example values,
    and another set loaded from a pickle file. It then calls the check_variables function
    with these datasets to verify its behavior.
    """
    
    # Simulated variable values (example values)
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

    # Define variable names and types
    names_var = [
        ["varA", "varB"],
        ["aux1", "aux2"]
    ]

    # Type must be set as "continuous" or "categorical".
    types_var = [
        ["categorical", "continuous"],
        ["continuous", "continuous"]
    ]
    
    # Simulated variables can be partially informed
    sim_var = [
        np.array([[varAval11, varAval12, varAval13],
                  [varAval21, varAval22, varAval23]], dtype=int),
        np.array([[varBval11, varBval12, varBval13],
                  [varBval21, varBval22, varBval23]], dtype=float)
    ]
    
    # Auxiliary variables must be fully informed
    aux_var = [
        np.array([[var1val11, var1val12, var1val13], 
                  [var1val21, var1val22, var1val23]], dtype=float),
        np.array([[var2val11, var2val12, var2val13], 
                  [var2val21, var2val22, var2val23]], dtype=float)
    ]
    
    simulated_var = {}
    auxiliary_var = {}
    
    # Populate simulated and auxiliary variables dictionaries
    for i in range(len(names_var[0])):
        simulated_var[names_var[0][i]] = sim_var[i]
    for i in range(len(names_var[1])):
        auxiliary_var[names_var[1][i]] = aux_var[i]
        
    # Call check_variables with the first dataset
    check_variables(simulated_var, auxiliary_var, names_var, types_var)
    
    # Testing with another dataset
    path2data = "C:/Users/Axel (Travail)/Documents/ENSG/CET/GeoclassificationMPS/Missing-Data/data/"
    suffix = "-simple"
    picklefn = "mt-isa-data" + suffix + ".pickle"
    pickledestination = path2data + picklefn
    
    # Load variables from a pickle file
    with open(pickledestination, 'rb') as f:
        [
            _, grid_geo, grid_lmp, grid_mag,
            grid_grv, grid_ext, vec_x, vec_y
        ] = pickle.load(f)
     
    names_var = [
        ["grid_geo", "grid_lmp"],
        []
    ]

    # Type must be set as "continuous" or "categorical".
    types_var = [
        ["categorical", "continuous"],
        []
    ]
    
    sim_var = [grid_geo, grid_lmp]
    aux_var = []
    
    simulated_var = {}
    auxiliary_var = {}
    
    # Populate simulated and auxiliary variables dictionaries
    for i in range(len(names_var[0])):
        simulated_var[names_var[0][i]] = sim_var[i]
    for i in range(len(names_var[1])):
        auxiliary_var[names_var[1][i]] = aux_var[i]
        
    # Call check_variables with the second dataset
    check_variables(simulated_var, auxiliary_var, names_var, types_var)
     
     

##################################### TEST TI_GENERATION.PY

def test_gen_ti_mask_circles():
    """
    """
    nx = 100  # nombre de colonnes
    ny = 100  # nombre de lignes
    ti_pct_area = 50  # pourcentage de l'aire de la grille à couvrir
    ti_ndisks = 10  # nombre de disques
    seed = 15  # graine pour le générateur de nombres aléatoires

    mask = gen_ti_mask_circles(nx, ny, ti_pct_area, ti_ndisks, seed)

    plt.figure(figsize=(8, 8))
    plt.imshow(mask, cmap='gray', origin='lower')
    plt.title(f'Binary Mask Generated with {ti_ndisks} Disks')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def test_gen_ti_mask_squares():
    nx = 100  # nombre de colonnes
    ny = 100  # nombre de lignes
    ti_pct_area = 50  # pourcentage de l'aire de la grille à couvrir
    ti_nsquares = 10  # nombre de carrés
    seed = 15  # graine pour le générateur de nombres aléatoires
    
    mask = gen_ti_mask_squares(nx, ny, ti_pct_area, ti_nsquares, seed)
    
    plt.imshow(mask, cmap='gray')
    plt.show()
    
def test_gen_ti_mask_separated_squares(showCoord=True):
    nx = 1000  # nombre de colonnes
    ny = 1000  # nombre de lignes
    ti_pct_area = 10  # pourcentage de l'aire de la grille à couvrir
    ti_nsquares = 50  # nombre de carrés
    seed = 15  
    plot_size = nx
    
    squares = gen_ti_mask_separatedSquares(nx, ny, ti_pct_area, ti_nsquares, seed)
    
    num_plots = len(squares)
    cols = 5  # Number of columns
    rows = (num_plots + cols - 1) // cols  # Calculate the required number of rows
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 4, rows * 4))
    
    if showCoord:
        for i, square in enumerate(squares):
            print(f"Square {i}:")
            print(square, "\t")

    
    for i in range(rows * cols):
        ax = axs.flat[i] if rows * cols > 1 else axs
        if i < num_plots:
            square_plot = np.zeros((plot_size, plot_size))
            square = squares[i]
            for idx in square:
                if 0 <= idx[0] < plot_size and 0 <= idx[1] < plot_size:
                    square_plot[idx[0], idx[1]] = 1
            ax.imshow(square_plot, cmap='gray', origin='lower')
            ax.set_title(f"Square {i}")
        else:
            ax.axis('off')  # Turn off unused subplots
    
    plt.tight_layout()
    plt.show()

def test_gen_ti_mask_single_square():
    nx, ny = 100, 100
    simgrid_pct = 40
    ti_pct_area = 60
    seed = 15
    
    square1, square2, overlap_percentage = gen_ti_mask_single_square(nx, ny, simgrid_pct, ti_pct_area, seed, nseedGenerations=100, ntryPerSeed=1000, tolerance = 5)
    
    print(f"Effective overlap percentage of the TI : {overlap_percentage}")
    
    plt.figure(figsize=(8, 8))
    plt.imshow(square1, cmap='Blues', origin='lower', alpha=0.5)
    plt.imshow(square2, cmap='Reds', origin='lower', alpha=0.5)
    plt.title("Two Overlapping Squares")
    plt.colorbar(label="Presence")
    plt.axis('off')
    plt.show()
    

    



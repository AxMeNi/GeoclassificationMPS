# -*- coding:utf-8 -*-
__projet__ = "GeoclassificationMPS"
__nom_fichier__ = "tests"
__author__ = "MENGELLE Axel"
__date__ = "juillet 2024"


from ti_generation import *

import matplotlib.pyplot as plt

def test_gen_ti_mask_circles():
    """
    """
    nx = 100  # nombre de colonnes
    ny = 100  # nombre de lignes
    ti_pct_area = 50  # pourcentage de l'aire de la grille à couvrir
    ti_ndisks = 10  # nombre de disques
    seed = 15  # graine pour le générateur de nombres aléatoires

    # Génération du masque
    mask = gen_ti_mask_circles(nx, ny, ti_pct_area, ti_ndisks, seed)

    # Affichage du masque avec un plot
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
    ti_pct_area = 50  # pourcentage de l'aire de la grille à couvrir
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
    ti_pct_area = 50
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
    

    



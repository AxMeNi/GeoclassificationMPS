# -*- coding:utf-8 -*-
__projet__ = "GeoclassificationMPS"
__nom_fichier__ = "ti_generation"
__author__ = "MENGELLE Axel"
__date__ = "juillet 2024"

import numpy as np  # NumPy for numerical operations
from skimage.draw import disk  # For drawing shapes
from skimage.draw import rectangle
from skimage.morphology import binary_dilation  # For morphological operations
from time import *
from math import *

def gen_ti_frame_circles(nc, nr, ti_pct_area, ti_ndisks, seed):
    """
    Generate a binary frame representing multiple disks within a grid.

    Parameters:
    ----------
    nc : int
        Number of columns in the grid.
    nr : int
        Number of rows in the grid.
    ti_pct_area : float
        Percentage of the grid area to cover with disks.
    ti_ndisks : int
        Number of disks to generate.
    seed : int
        Seed for the random number generator.

    Returns:
    -------
    frame : ndarray
        Binary frame with 1s indicating disk positions within the grid.
    """
    rng = np.random.default_rng(seed=seed)
    rndr = rng.integers(low=0, high=nr, size=ti_ndisks)
    rndc = rng.integers(low=0, high=nc, size=ti_ndisks)
    radius = np.floor(np.sqrt((nc * nr * ti_pct_area / 100 / ti_ndisks) / np.pi))
    frame = np.zeros((nr, nc))
    for i in range(ti_ndisks):
        rr, cc = disk((rndr[i], rndc[i]), radius, shape=(nr, nc))
        frame[rr, cc] = 1
    check_pct = np.sum(frame.flatten()) / (nc * nr) * 100
    while check_pct < ti_pct_area:
        frame = binary_dilation(frame)
        check_pct = np.sum(frame.flatten()) / (nc * nr) * 100
    return frame
    
def gen_ti_frame_squares(nc, nr, ti_pct_area, ti_nsquares, seed):
    """
    Generate a binary frame representing multiple squares within a grid.

    Parameters:
    ----------
    nc : int
        Number of columns in the grid.
    nr : int
        Number of rows in the grid.
    ti_pct_area : float
        Percentage of the grid area to cover with squares.
    ti_nsquares : int
        Number of squares to generate.
    seed : int
        Seed for the random number generator.

    Returns:
    -------
    frame : ndarray
        Binary frame with 1s indicating square positions within the grid.
    """
    rng = np.random.default_rng(seed=seed)
    rndr = rng.integers(low=0, high=nr, size=ti_nsquares)
    rndc = rng.integers(low=0, high=nc, size=ti_nsquares)
    side_length = int(np.sqrt((nc * nr * ti_pct_area / 100) / ti_nsquares))
    frame = np.zeros((nr, nc))
    for i in range(ti_nsquares):
        rr_start = max(0, rndr[i] - side_length // 2)
        cc_start = max(0, rndc[i] - side_length // 2)
        rr_end = min(nr - 1, rr_start + side_length - 1)
        cc_end = min(nc - 1, cc_start + side_length - 1)
        rr, cc = rectangle(start=(rr_start, cc_start), end=(rr_end, cc_end))
        frame[rr, cc] = 1
    check_pct = np.sum(frame.flatten()) / (nc * nr) * 100
    while check_pct < ti_pct_area:
        frame = np.pad(frame, 1, mode='constant', constant_values=0)
        frame = binary_dilation(frame)
        frame = frame[1:-1, 1:-1]
        check_pct = np.sum(frame.flatten()) / (nc * nr) * 100
    return frame
    
def gen_ti_frame_separatedSquares(nc, nr, ti_pct_area, ti_nsquares, seed):
    """
    Generate a binary frame representing multiple squares within a grid and return each square as a separate array.

    Parameters:
    ----------
    nc : int
        Number of columns in the grid.
    nr : int
        Number of rows in the grid.
    ti_pct_area : float
        Percentage of the grid area to cover with squares.
    ti_nsquares : int
        Number of squares to generate.
    seed : int
        Seed for the random number generator.

    Returns:
    -------
    squares : list of ndarrays
        List of arrays with 1s indicating square positions within the grid.
    """
    rng = np.random.default_rng(seed=seed)
    rndy = rng.integers(low=0, high=nr, size=ti_nsquares)
    rndx = rng.integers(low=0, high=nc, size=ti_nsquares)
    side_length = int(np.sqrt((nc * nr * ti_pct_area / 100) / ti_nsquares))
    squares = []
    frame = np.zeros((nr, nc))

    for i in range(ti_nsquares):
        rr_start = max(0, rndy[i] - side_length // 2)
        cc_start = max(0, rndx[i] - side_length // 2)
        rr_end = min(nr, rr_start + side_length)
        cc_end = min(nc, cc_start + side_length)

        # Adjust start and end coordinates if the end exceeds the grid bounds
        if rr_end == nr:
            rr_start = max(0, nr - side_length)
        if cc_end == nc:
            cc_start = max(0, nc - side_length)
        
        rr_end = min(nr, rr_start + side_length)
        cc_end = min(nc, cc_start + side_length)
        
        #Rows and columns
        rr, cc = rectangle(start=(rr_start, cc_start), end=(rr_end, cc_end))
        
        # Ensure rr and cc are within bounds
        rr = np.clip(rr, 0, nr-1)
        cc = np.clip(cc, 0, nc-1)

        square_frame = np.zeros((nr, nc))
        square_frame[rr, cc] = 1
        
        square_indexes = np.argwhere(square_frame == 1)
        squares.append(square_indexes)
        

        current_pct = np.sum(frame) / (nc * nl) * 100
    
    return squares
    



def gen_ti_frame_single_rectangle(nc, nr):
    """
    Generate a binary frame representing a training image (TI) and a simulation grid (simgrid) within a larger grid.

    Parameters:
    ----------
    nc : int
        Number of columns in the grid.
    nr : int
        Number of rows in the grid.
    tolerance : int, optional
        Tolerance for positioning adjustments, default is 5.

    Returns:
    -------
    ti_frame : ndarray
        An array indicating the position of the training image within the grid.
    simgrid_mask : ndarray
        An array indicating the position of the simulation grid within the grid.

    Raises:
    ------
    ValueError
        If one of the grid dimensions is too small (<34) to create a smaller simulation grid, suggesting to change SGDimIsDataDim to True.
    """
    if nc  < 34 or nr < 34:
        raise ValueError (f"One of the grid lengths is too small (<34) to create a smaller simulation grid, please consider changing SGDimIsDataDim to True")
    
    # Calculate side length of the ti
    rlength_ti_frame = max(int(sqrt(0.1)*nr),1) # Area of the ti is 10% the area of the grid
    clength_ti_frame = max(int(sqrt(0.1)*nc),1)
    
    # Chose the position of the ti
    rstart_ti_frame = 5
    cstart_ti_frame = 5
    rend_ti_frame = rstart_ti_frame + rlength_ti_frame
    cend_ti_frame = cstart_ti_frame + clength_ti_frame

    # Calculate side length of the simgrid
    rlength_simgrid_mask = max(int(sqrt(0.1)*rlength_ti_frame),1) # Area of the simgrid is 10% the area of the ti
    clength_simgrid_mask = max(int(sqrt(0.1)*clength_ti_frame),1)
    
    # Chose the position of the simgrid
    rstart_simgrid_mask = rend_ti_frame - rlength_simgrid_mask//2
    cstart_simgrid_mask = cend_ti_frame - clength_simgrid_mask//2
    rend_simgrid_mask = rstart_simgrid_mask + rlength_simgrid_mask
    cend_simgrid_mask = cstart_simgrid_mask + clength_simgrid_mask

    
    # Generate frames
    ti_frame = np.zeros((nr, nc))
    simgrid_mask = np.zeros((nr, nc))
    
    r_frame, c_frame = rectangle(start=(rstart_ti_frame, cstart_ti_frame), end=(rend_ti_frame, cend_ti_frame), shape=(nr, nc))
    r_mask, c_mask = rectangle(start=(rstart_simgrid_mask, cstart_simgrid_mask), end=(rend_simgrid_mask, cend_simgrid_mask), shape=(nr, nc))
    
    ti_frame[r_frame, c_frame] = 1
    simgrid_mask[r_mask, c_mask] = 2
    
    return ti_frame, simgrid_mask
    
    


def build_ti():
    return True

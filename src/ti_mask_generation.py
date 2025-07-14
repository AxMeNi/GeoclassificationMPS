# -*- coding:utf-8 -*-
__projet__ = "GeoclassificationMPS"
__nom_fichier__ = "ti_generation"
__author__ = "MENGELLE Axel"
__date__ = "juillet 2024"

from reduced_ti_sg import *

import numpy as np  
from skimage.draw import disk  # For drawing shapes
from skimage.draw import rectangle
from skimage.morphology import binary_dilation  # For morphological operations
from data_treatment import check_custom_mask
from time import *
from math import *



def gen_ti_frame_circles(nr, nc, ti_pct_area = 90, ti_ndisks = 10, seed = None):
    """
    Generate a binary frame representing multiple disks within a grid.

    Parameters:
    ----------
    nr : int
        Number of rows in the grid.
    nc : int
        Number of columns in the grid.
    ti_pct_area : float
        Percentage of the grid area to cover with disks.
    ti_ndisks : int
        Number of disks to generate.
    seed : int
        Seed for the random number generator.

    Returns:
    -------
    frame : list of ndarray
        List with one binary frame with 1s indicating disk positions within the grid.
    need_to_cut : list of boolean
        True if the simulated_var will be needed to be cut within the ti_frame shape to create a smaller TI.
    """
    if seed is None:
        seed = int(rd.randint(1, 2**32 - 1))
        print(f"Seed used to generate the TI : {seed}")
    rng = np.random.default_rng(seed=seed)
    rndr = rng.integers(low=0, high=nr, size=ti_ndisks)
    rndc = rng.integers(low=0, high=nc, size=ti_ndisks)
    radius = np.floor(np.sqrt((nc * nr * ti_pct_area / 100 / ti_ndisks) / np.pi))
    frame = np.zeros((nr, nc))
    for i in range(ti_ndisks):
        rr, cc = disk((rndr[i], rndc[i]), radius, shape=(nr, nc))
        frame[rr, cc] = 1
    check_pct = np.sum(frame.flatten()) / (nc * nr) * 100
    
    while check_pct < 0.95*ti_pct_area or check_pct > 1.05*ti_pct_area:
        frame = np.zeros((nr, nc))
        for i in range(ti_ndisks):
            rr, cc = disk((rndr[i], rndc[i]), radius + 1, shape=(nr, nc))
            frame[rr, cc] = 1
        check_pct = np.sum(frame.flatten()) / (nc * nr) * 100
        radius += 1 

    ti_frame = [frame]
    need_to_cut = [False]

    return ti_frame, need_to_cut
    
    
def gen_ti_frame_squares(nr, nc, ti_pct_area = 90, ti_nsquares = 10, seed = None):
    """
    Generate a binary frame representing multiple squares within a grid.

    Parameters:
    ----------
    nr : int
        Number of rows in the grid.
    nc : int
        Number of columns in the grid.
    ti_pct_area : float
        Percentage of the grid area to cover with squares.
    ti_nsquares : int
        Number of squares to generate.
    seed : int
        Seed for the random number generator.

    Returns:
    -------
    ti_frame : list of ndarray
        List with one binary frame with 1s indicating square positions within the grid.
    need_to_cut : list of boolean
        True if the simulated_var will be needed to be cut within the ti_frame shape to create a smaller TI.
    """
    if seed is None:
        seed = int(rd.randint(1, 2**32 - 1))
        print(f"Seed used to generate the TI : {seed}")
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

    while check_pct < 0.95*ti_pct_area or check_pct > 1.05*ti_pct_area:
        frame = np.zeros((nr, nc))
        for i in range(ti_nsquares):
            rr_start = max(0, rndr[i] - (side_length // 2 + 1))
            cc_start = max(0, rndc[i] - (side_length // 2 + 1))
            rr_end = min(nr - 1, rr_start + side_length)
            cc_end = min(nc - 1, cc_start + side_length)
            rr, cc = rectangle(start=(rr_start, cc_start), end=(rr_end, cc_end))
            frame[rr, cc] = 1
        
        check_pct = np.sum(frame.flatten()) / (nc * nr) * 100
        side_length += 1

    ti_frame = [frame]
    need_to_cut = [False]

    return ti_frame, need_to_cut
  
    
def gen_ti_frame_separatedSquares(nr, nc, ti_pct_area = 90, ti_nsquares = 10, seed = None):
    """
    Generate a binary frame representing multiple squares within a grid and return each square as a separate array.

    Parameters:
    ----------
    nr : int
        Number of rows in the grid.
    nc : int
        Number of columns in the grid.
    ti_pct_area : float
        Percentage of the grid area to cover with squares.
    ti_nsquares : int
        Number of squares to generate.
    seed : int
        Seed for the random number generator.
    
    Returns:
    -------
    ti_frames_list : list of ndarrays
        List of arrays with 1s indicating square positions within the grid.
    need_to_cut : list of boolean
        True if the simulated_var will be needed to be cut within the ti_frame shape to create a smaller TI.
    """
    if seed is None:
        seed = int(rd.randint(1,2**32-1))
        print(f"Seed used to generate the TI : {seed}")
    rng = np.random.default_rng(seed=seed)
    rndr = rng.integers(low=0, high=nr, size=ti_nsquares)
    rndc = rng.integers(low=0, high=nc, size=ti_nsquares)
    side_length = int(np.sqrt((nc * nr * ti_pct_area / 100) / ti_nsquares))
    ti_frames_list = []
    square_frame = np.zeros((nr, nc))

    for i in range(ti_nsquares):
        rr_start = max(0, rndr[i] - side_length // 2)
        cc_start = max(0, rndc[i] - side_length // 2)
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
        rr, cc = rectangle(start=(rr_start, cc_start), end=(rr_end, cc_end), shape=(nr, nc))
        
        # Ensure rr and cc are within bounds
        rr = np.clip(rr, 0, nr-1)
        cc = np.clip(cc, 0, nc-1)
  
        square_frame = np.zeros((nr, nc))
        square_frame[rr, cc] = 1
        ti_frames_list.append(square_frame)
        
    need_to_cut = [True for _ in range(len(ti_frames_list))]
    
    return ti_frames_list, need_to_cut


def gen_ti_frame_custom(nr, nc, custom_mask_path):
    """
    """
    ti_frame = np.load(custom_mask_path)
    check_custom_mask(ti_frame,nr,nc)
    need_to_cut = [False]
    ti_frame_C = [ti_frame]
    return ti_frame_C, need_to_cut


def gen_ti_frame_sg_mask(nr, nc, pct_ti_sg_overlap=10, pct_sg=None, pct_ti=None, cc_sg=None, rr_sg=None, cc_ti=None, rr_ti=None, seed=None):
    """
    Generate a binary frame representing a single rectangle within a grid and a simulation grid mask.

    Parameters:
    ----------
    nr : int
        Number of rows in the grid.
    nc : int
        Number of columns in the grid.
    pct_ti_sg_overlap : float, optional
        Percentage overlap between the training image and the simulation grid (default is 10).
    pct_sg : float, optional
        Percentage of the grid area to cover with the simulation grid, if dimensions are not provided (default is 10).
    pct_ti : float, optional
        Percentage of the grid area to cover with the training image, if dimensions are not provided (default is 30).
    cc_sg : int, optional
        Width of the simulation grid. If not provided, calculated based on `pct_sg`.
    rr_sg : int, optional
        Height of the simulation grid. If not provided, calculated based on `pct_sg`.
    cc_ti : int, optional
        Width of the training image. If not provided, calculated based on `pct_ti`.
    rr_ti : int, optional
        Height of the training image. If not provided, calculated based on `pct_ti`.
    seed : int, optional
        Seed for the random number generator.

    Returns:
    -------
    ti_frame : list of ndarray
        List with one array with 1s indicating the position of the training image within the grid.
    simgrid_mask : ndarray
        An array with 2s indicating the position of the simulation grid within the grid.
    cc_sg : int
        The column size of the simulation grid
    rr_sg : int
        The row size of the simulation grid
    need_to_cut : list of boolean
        True if the simulated_var will be needed to be cut within the ti_frame shape to create a smaller TI.
    """
    output = get_ti_sg(nc, nr, 
                      cc_sg, rr_sg, pct_sg, 
                      cc_ti, rr_ti, pct_ti, 
                      pct_ti_sg_overlap, seed)

    c0_sg, cc_sg, r0_sg ,rr_sg, c0_overlap, cc_overlap, r0_overlap, rr_overlap, c0_ti, cc_ti, r0_ti, rr_ti = output

    frame = np.zeros((nr, nc))
    simgrid_mask = np.zeros((nr, nc))

    nr_frame, nc_frame = rectangle(start=(r0_ti, c0_ti), end=(r0_ti+rr_ti, c0_ti + cc_ti), shape=(nr, nc))
    nr_mask, nc_mask = rectangle(start=(r0_sg, c0_sg), end=(r0_sg+rr_sg, c0_sg+cc_sg), shape=(nr, nc))
    # nr_frame, nc_frame = rectangle(start=(r0_overlap, c0_overlap), end=(r0_overlap+rr_overlap, c0_overlap+cc_overlap), shape=(nr, nc))
    
    frame[nr_frame, nc_frame] = 1
    simgrid_mask[nr_mask, nc_mask] = 1
    
    ti_frame = []
    ti_frame.append(frame)
    
    need_to_cut = [True]
    
    return ti_frame, need_to_cut, simgrid_mask, cc_sg, rr_sg
    

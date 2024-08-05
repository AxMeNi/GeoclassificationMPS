# -*- coding:utf-8 -*-
__projet__ = "GeoclassificationMPS"
__nom_fichier__ = "ti_generation"
__author__ = "MENGELLE Axel"
__date__ = "juillet 2024"

from reduced_ti_sg import *

import numpy as np  
import geone as gn
from skimage.draw import disk  # For drawing shapes
from skimage.draw import rectangle
from skimage.morphology import binary_dilation  # For morphological operations
from time import *
from math import *

def gen_ti_frame_circles(nr, nc, ti_pct_area, ti_ndisks, seed):
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
    ti_frame = []
    ti_frame.append(frame)
    
    need_to_cut = [False]
    
    return ti_frame
    
def gen_ti_frame_squares(nr, nc, ti_pct_area, ti_nsquares, seed):
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
    ti_frame = []
    ti_frame.append(frame)
    
    need_to_cut = [False]
    
    return ti_frame, need_to_cut
    
def gen_ti_frame_separatedSquares(nr, nc, ti_pct_area, ti_nsquares, seed):
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
    rng = np.random.default_rng(seed=seed)
    rndr = rng.integers(low=0, high=nr, size=ti_nsquares)
    rndc = rng.integers(low=0, high=nc, size=ti_nsquares)
    side_length = int(np.sqrt((nc * nr * ti_pct_area / 100) / ti_nsquares))
    ti_frames_list = []
    frame = np.zeros((nr, nc))

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
        rr, cc = rectangle(start=(rr_start, cc_start), end=(rr_end, cc_end))
        
        # Ensure rr and cc are within bounds
        rr = np.clip(rr, 0, nr-1)
        cc = np.clip(cc, 0, nc-1)

        square_frame = np.zeros((nr, nc))
        square_frame[rr, cc] = 1
        
        square_indexes = np.argwhere(square_frame == 1)
        ti_frames_list.append(square_indexes)
        

        current_pct = np.sum(frame) / (nc * nr) * 100
        
    need_to_cut = [True for _ in range(len(ti_frames_list))]
    
    return ti_frames_list, need_to_cut
    



def gen_ti_frame_single_rectangle(nr, nc, ti_sg_overlap_percentage=10, pct_sg=10, pct_ti=30, cc_sg=None, rr_sg=None, cc_ti=None, rr_ti=None, seed=None):
    """
    Generate a binary frame representing a single rectangle within a grid and a simulation grid mask.

    Parameters:
    ----------
    nr : int
        Number of rows in the grid.
    nc : int
        Number of columns in the grid.
    ti_sg_overlap_percentage : float, optional
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
                      ti_sg_overlap_percentage, seed)

    c0_sg, cc_sg, r0_sg ,rr_sg, c0_overlap, cc_overlap, r0_overlap, rr_overlap, c0_ti, cc_ti, r0_ti, rr_ti = output

    frame = np.zeros((nr, nc))
    simgrid_mask = np.zeros((nr, nc))

    nr_frame, nc_frame = rectangle(start=(r0_ti, c0_ti), end=(r0_ti+rr_ti, c0_ti + cc_ti), shape=(nr, nc))
    nr_mask, nc_mask = rectangle(start=(r0_sg, c0_sg), end=(r0_sg+rr_sg, c0_sg+cc_sg), shape=(nr, nc))
    
    frame[nr_frame, nc_frame] = 1
    simgrid_mask[nr_mask, nc_mask] = 1
    
    ti_frame = []
    ti_frame.append(frame)
    
    need_to_cut = [True]
    
    return ti_frame, need_to_cut, simgrid_mask, cc_sg, rr_sg
    
    
    



def build_ti(ti_frames_list, 
            need_to_cut, 
            simulated_var, 
            nc_simgrid, nr_simgrid, 
            auxiliary_var, 
            types_var, names_var, 
            novalue, 
            simgrid_mask = None):
    """
    """
              
    # Building TI(s)
    ti_list = []
    
    for i in range(len(ti_frames_list)):
        ti_frame = ti_frames_list[i]
        ntc = need_to_cut[i]
        
        #Case for which a cut is needed
        if ntc:
        
            name = "TI{}_{}".format(i,time())
            ti = gn.img.Img(nv=0)
            
            #Reshape of the simulated var and integration to the TI
            for var_name, var_value in simulated_var.items():
            
                var_value_masked = np.where(ti_frame == 1, var_value, novalue)

                rows = np.any(ti_frame, axis=1)
                cols = np.any(ti_frame, axis=0)
                
                row_start, row_end = np.where(rows)[0][[0, -1]]
                col_start, col_end = np.where(cols)[0][[0, -1]]
                
                ti.set_grid(nx=col_end-col_start+1, ny=row_end-row_start+1, nz=1, sx=1, sy=1, sz=1, ox=0, oy=0, oz=0)
                
                var_value_cut = var_value_masked[row_start:row_end+1, col_start:col_end+1]
                ti.append_var(val=var_value_cut, varname=var_name)
                
            #Reshape of the auxiliary var and integration to the TI
            for var_name, var_value in auxiliary_var.items():
                var_value_masked = np.where(ti_frame == 1, var_value, novalue)

                rows = np.any(ti_frame, axis=1)
                cols = np.any(ti_frame, axis=0)

                row_start, row_end = np.where(rows)[0][[0, -1]]
                col_start, col_end = np.where(cols)[0][[0, -1]]
                
                ti.set_grid(nx=col_end-col_start+1, ny=row_end-row_start+1, nz=1, sx=1, sy=1, sz=1, ox=0, oy=0, oz=0)

                var_value_cut = var_value_masked[row_start:row_end+1, col_start:col_end+1]
                
                ti.append_var(val=var_value_cut, varname=var_name)
                
            
            ti_list.append(ti)
            
        #Case for which no cut is needed
        else:
        
            name = "TI{}_{}".format(i,time())
            ti = gn.img.Img(nv=0)
            
            n_row, n_col = auiliary_var[names_var[1][0]].shape()
            
            ti.set_grid(nx=n_col, ny=n_row, nz=1, sx=1, sy=1, sz=1, ox=0, oy=0, oz=0)
            
            #Integration of the simulated_var in the TI
            for var_name, var_value in simulated_var.items() :
                var_value_masked = np.where(ti_frame == 1, var_value, novalue)

                simulated_var_updated[var_name] = var_value_masked
                
                ti.append_var(val=var_value_cut, varname=var_name)
            
            #No application of the mask to the auxiliary var which have to be fully informed and integration to the TI
            for var_name, var_value in simulated_var.items() :

                ti.append_var(val=var_value_cut, varname=var_name)
            
            ti_list.append(ti)
          
        
    
    # Building conditioning AUXILIARY data
    cd_list = []
    
    if simgrid_mask is not None:
    
        name = "CondData{}_{}".format(i,time())
        cd = gn.img.Img(nv=0)
        
        # Integration of the auxiliary_var in the simulation grid to control the non stationarity
        # The values of the aux var here is to control the non stationarity of the data
        if simgrid_mask is not None:
            for var_name, var_value in auxiliary_var.items():
                print(var_name)
                var_value_masked = np.where(simgrid_mask == 1, var_value, novalue)

                rows = np.any(simgrid_mask, axis=1)
                cols = np.any(simgrid_mask, axis=0)

                row_start, row_end = np.where(rows)[0][[0, -1]]
                col_start, col_end = np.where(cols)[0][[0, -1]]
                
                cd.set_grid(nx=col_end-col_start+1, ny=row_end-row_start+1, nz=1, sx=1, sy=1, sz=1, ox=0, oy=0, oz=0)

                var_value_cut = var_value_masked[row_start:row_end+1, col_start:col_end+1]
                cd.append_var(val=var_value_cut, varname=var_name)

            cd_list.append(cd)
            
        else:
            cd.set_grid(nx=nc_simgrid, ny=nr_simgrid, nz=1, sx=1, sy=1, sz=1, ox=0, oy=0, oz=0)
            
            for var_name, var_value in auxiliary_var.items():
                cd.append_var(val=var_value_cut, varname=var_name)
            
            cd_list.append(cd)
         
            
    #Building conditioning SIMULATED data 
    
    return ti_list, cd_list
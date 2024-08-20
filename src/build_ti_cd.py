# -*- coding:utf-8 -*-
__projet__ = "GeoclassificationMPS"
__nom_fichier__ = "interface"
__author__ = "MENGELLE Axel"
__date__ = "juillet 2024"

import numpy as np
import geone as gn
from time import time



def build_ti_cd(ti_frames_list, 
                need_to_cut, 
                sim_var, 
                nc_simgrid, nr_simgrid, 
                auxTI_var, auxSG_var,
                names_var,
                simgrid_mask = None,
                condIm_var = None):
    """
    Build training images (TI) and conditioning data (CD) for geostatistical simulations.

    Parameters
    ----------
    ti_frames_list : list of arrays
        List of binary arrays representing frames or masks that indicate the areas to include in the TIs.
    need_to_cut : list of bool
        List of boolean flags indicating whether each corresponding TI frame needs to be cut to remove excess areas.
    sim_var : dict
        Dictionary containing simulation variables, where keys are variable names, and values are NumPy arrays representing 
        the simulated data.
    nc_simgrid : int
        Number of columns in the simulation grid.
    nr_simgrid : int
        Number of rows in the simulation grid.
    auxTI_var : dict
        Dictionary of auxiliary variables for the training images, where keys are variable names and values are NumPy arrays.
    auxSG_var : dict
        Dictionary of auxiliary variables for the simulation grid, where keys are variable names and values are NumPy arrays.
    names_var : list of str
        List of variable names used in both the simulation and auxiliary variables.
    simgrid_mask : array, optional
        Binary mask array for the simulation grid, used to indicate regions of interest in the simulation. Default is None.
    condIm_var : dict, optional
        Dictionary containing conditioning simulated data, where keys are variable names and values are NumPy arrays. 
        Default is None.

    Returns
    -------
    ti_list : list
        List of generated training images, where each TI is an instance of the gn.img.Img class.
    cd_list : list
        List of conditioning data images, where each is an instance of the gn.img.Img class.

    Notes
    -----
    - If `need_to_cut` is True, the function crops the grid to remove areas not indicated in `ti_frames_list`.
    - Simulated variables (`sim_var`) and auxiliary variables (`auxTI_var`) are reshaped and integrated into each TI.
    - For conditioning data, auxiliary variables (`auxSG_var`) are applied to control non-stationarity within the simulation grid.
    - Optional conditioning simulated data (`condIm_var`) is also included if provided.
    """
    # Building TI(s)
    ti_list = []
    
    for i in range(len(ti_frames_list)):
        ti_frame = ti_frames_list[i]
        ntc = need_to_cut[i]
        
        # Case for which a cut is needed
        if ntc:
        
            name = "TI{}_{}".format(i,time())
            ti = gn.img.Img(nv=0,name=name)
            
            # Reshape of the simulated var and integration to the TI
            for var_name, var_value in sim_var.items():
            
                var_value_masked = np.where(ti_frame == 1, var_value, np.nan)

                rows = np.any(ti_frame, axis=1)
                cols = np.any(ti_frame, axis=0)
                
                row_start, row_end = np.where(rows)[0][[0, -1]]
                col_start, col_end = np.where(cols)[0][[0, -1]]
                
                ti.set_grid(nx=col_end-col_start+1, ny=row_end-row_start+1, nz=1, sx=1, sy=1, sz=1, ox=0, oy=0, oz=0)
                
                var_value_cut = var_value_masked[row_start:row_end+1, col_start:col_end+1]
                
                ti.append_var(val=var_value_cut, varname=var_name)
                
            # Reshape of the auxiliary desciptive var and integration to the TI
            for var_name, var_value in auxTI_var.items():
                var_value_masked = np.where(ti_frame == 1, var_value, np.nan)

                rows = np.any(ti_frame, axis=1)
                cols = np.any(ti_frame, axis=0)

                row_start, row_end = np.where(rows)[0][[0, -1]]
                col_start, col_end = np.where(cols)[0][[0, -1]]
                
                ti.set_grid(nx=col_end-col_start+1, ny=row_end-row_start+1, nz=1, sx=1, sy=1, sz=1, ox=0, oy=0, oz=0)

                var_value_cut = var_value_masked[row_start:row_end+1, col_start:col_end+1]
                
                ti.append_var(val=var_value_cut, varname=var_name)
                           
            gn.img.writeImageTxt(f"TI{i}.txt", ti)      
            ti = gn.img.readImageTxt(f"TI{i}.txt")    
            ti_list.append(ti)
            
        # Case for which no cut is needed
        else:
        
            name = "TI{}_{}".format(i,time())
            ti = gn.img.Img(nv=0,name=name)
                       
            ti.set_grid(nx=nc_simgrid, ny=nr_simgrid, nz=1, sx=1, sy=1, sz=1, ox=0, oy=0, oz=0)
            
            # Integration of sim_var in the TI
            for var_name, var_value in sim_var.items() :
                var_value_masked = np.where(ti_frame == 1, var_value, np.nan)
                ti.append_var(val=var_value_masked, varname=var_name)
            
            # No application of the mask to the auxiliary var which have to be fully informed and integration to the TI
            for var_name, var_value in auxTI_var.items() :

                ti.append_var(val=var_value, varname=var_name)
            
            ti_list.append(ti)
          
        
    
    # Building conditioning AUXILIARY data
    cd_list = []

    name = "CondData{}_{}".format(i,time())
    cd = gn.img.Img(nv=0, name=name)
    
    # Integration of the auxiliary_var in the simulation grid to control the non stationarity
    # The values of the aux var here is to control the non stationarity of the data
    for var_name, var_value in auxSG_var.items():
        
        # Case for which a cut is needed because the SG is reduced
        if var_value.shape != (nr_simgrid,nc_simgrid) :    
            var_value_masked = np.where(simgrid_mask == 1, var_value, np.nan)

            rows = np.any(simgrid_mask, axis=1)
            cols = np.any(simgrid_mask, axis=0)

            row_start, row_end = np.where(rows)[0][[0, -1]]
            col_start, col_end = np.where(cols)[0][[0, -1]]
            
            cd.set_grid(nx=col_end-col_start, ny=row_end-row_start, nz=1, sx=1, sy=1, sz=1, ox=0, oy=0, oz=0)

            var_value_cut = var_value_masked[row_start:row_end, col_start:col_end]   
            cd.append_var(val=var_value_cut, varname=var_name)
            
        else :
            var_value_masked = np.where(simgrid_mask == 1, var_value, np.nan)
            cd.set_grid(nx=nc_simgrid, ny=nr_simgrid, nz=1, sx=1, sy=1, sz=1, ox=0, oy=0, oz=0)
            cd.append_var(val=var_value_masked, varname=var_name)
        
    gn.img.writeImageTxt(f"{name}.txt", cd)      
    cd = gn.img.readImageTxt(f"{name}.txt")     
    
    cd_list.append(cd)    

    
            
    # Building conditioning SIMULATED data 
    if len(condIm_var) != 0:
        name = "CondData{}_{}".format(i,time())
        cd = gn.img.Img(nv=0, name=name)
        
        for var_name, var_value in condIm_var.items():
        
            # Case for which a cut is needed because the SG is reduced
            if var_value.shape != (nr_simgrid,nc_simgrid) :
                var_value_masked = np.where(simgrid_mask == 1, var_value, np.nan)

                rows = np.any(simgrid_mask, axis=1)
                cols = np.any(simgrid_mask, axis=0)

                row_start, row_end = np.where(rows)[0][[0, -1]]
                col_start, col_end = np.where(cols)[0][[0, -1]]
                
                cd.set_grid(nx=col_end-col_start, ny=row_end-row_start, nz=1, sx=1, sy=1, sz=1, ox=0, oy=0, oz=0)

                var_value_cut = var_value_masked[row_start:row_end, col_start:col_end]   
                cd.append_var(val=var_value_cut, varname=var_name)
            else:
                var_value_masked = np.where(simgrid_mask == 1, var_value, np.nan)
                cd.set_grid(nx=nc_simgrid, ny=nr_simgrid, nz=1, sx=1, sy=1, sz=1, ox=0, oy=0, oz=0)
                cd.append_var(val=var_value_masked, varname=var_name)
        
        gn.img.writeImageTxt(f"{name}.txt", cd)      
        cd = gn.img.readImageTxt(f"{name}.txt")     
        
        cd_list.append(cd)  
    
    return ti_list, cd_list


def gen_twenty_random_ti_cd(seed, method = "DependentCircles"):
    if method not in ["DependentCircles", "DependentSquares", "IndependentSquares", ""]
    ti_lists = []
    cd_lists = []
    for i in range(19):
        while cd.unique not in ti.unique:
            if method = "DependentCircles"
            gen_ti_frame_circles(nr, nc, ti_pct_area, ti_ndisks, seed)
            ti_list, cd_list = build_ti_cd(ti_frame2, need_to_cut2, sim_var, nc, nr, auxTI_var, auxSG_var, names_var, simgrid_mask1, condIm_var)
            seed += 1
        cd_lists.append()
        ti_lists.append() 
        
    return 

def chose_random_masks():
    return
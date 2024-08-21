# -*- coding:utf-8 -*-
__projet__ = "GeoclassificationMPS"
__nom_fichier__ = "launcher"
__author__ = "MENGELLE Axel"
__date__ = "juillet 2024"

"""
Script for processing and visualization of geospatial data.

This script imports necessary modules and initializes parameters for data processing
and visualization. It also loads pre-processed data from a pickle file.

Author: Guillaume Pirot
Date: Fri Jul 28 11:12:36 2023
"""
from simulation_functions_ter import *
from ti_mask_generation import *
import matplotlib.pyplot as plt


#### COLORS PARAMETERS
cm = plt.get_cmap('tab20')
defaultclrs = np.asarray(cm.colors)[[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11], :]
n_bin = 11
cmap_name = 'my_tab20'
defaultcmap = LinearSegmentedColormap.from_list(cmap_name, defaultclrs, N=n_bin)
defaultticmap = LinearSegmentedColormap.from_list('ticmap', np.vstack(([0, 0, 0], defaultclrs)), N=n_bin + 1)




def launcher(seed, 
            ti_methods, 
            ti_pct_area, ti_nshapes,
            pct_ti_sg_overlap, pct_sg, pct_ti, cc_sg, rr_sg, cc_ti, rr_ti,
            nn, dt, ms, numberofmpsrealizations, nthreads,
            cm, myclrs, n_bin, cmap_name, mycmap, ticmap,
            sim_var, auxTI_var, auxSG_var, condIm_var, names_var, types_var,
            nr, nc
            ):
    """

    """
    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)') + " - INIT")
    time.sleep(timesleep)
    
    #Initialization of variables
    ti_list = []
    cd_list = []
    
    #Create a simulation grid mask based on no values of the auxiliary variables
    simgrid_mask_aux = create_sg_mask(auxTI_var, auxSG_var, nr, nc)
    
    #Creation of the TI and of the SG
    if "DependentCircles" in ti_methods :
        ti_frame_DC, ntc_DC = gen_ti_frame_circles(nr, nc, ti_pct_area, ti_nshapes, seed)
        ti_list_DC, cd_list_DC = build_ti_cd(ti_frame_DC, need_to_cut_DC, sim_var, nc, nr, auxTI_var, auxSG_var, names_var, simgrid_mask_aux, condIm_var)
        ti_list.append(ti_list_DC)
        cd_list.append(cd_list_DC)
        
    if "DependentSquares" in ti_methods :
        ti_frame_DS, ntc_DS = gen_ti_frame_squares(nr, nc, ti_pct_area, ti_nshapes, seed)
        ti_list_DS, cd_list_DS = build_ti_cd(ti_frame_DS, need_to_cut_DS, sim_var, nc, nr, auxTI_var, auxSG_var, names_var, simgrid_mask_aux, condIm_var)
        ti_list.append(ti_list_DS)
        cd_list.append(cd_list_DS)
        
    if "IndependentSquares" in ti_methods :
        ti_frame_IS, ntc_IS = gen_ti_frame_separatedSquares(nr, nc, ti_pct_area, ti_nshapes, seed)
        ti_list_IS, cd_list_IS = build_ti_cd(ti_frame_IS, need_to_cut_IS, sim_var, nc, nr, auxTI_var, auxSG_var, names_var, simgrid_mask_aux, condIm_var)
        ti_list.append(ti_list_IS)
        cd_list.append(cd_list_IS)
        
    if "ReducedTiCd" in ti_methods :
        ti_frame_RTC, need_to_cut_RTC, simgrid_mask_RTC, cc_sg, rr_sg = gen_ti_frame_cd_mask(nr, nc, pct_ti_sg_overlap, pct_sg, pct_ti, cc_sg, rr_sg, cc_ti, rr_ti, seed)
        simgrid_mask_merged = merge_masks(simgrid_mask_aux, simgrid_mask_RTC)
        ti_list_RTC, cd_list_RTC = build_ti_cd(ti_frame_RTC, need_to_cut_RTC, sim_var, cc_sg, rr_sg, auxTI_var, auxSG_var, names_var, simgrid_mask_merged, condIm_var)
        ti_list.append(ti_list_RTC)
        cd_list.append(cd_list_RTC)
    
    nTI = len(ti_list)
    
    
    return

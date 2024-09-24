# -*- coding:utf-8 -*-
__projet__ = "GeoclassificationMPS"
__nom_fichier__ = "proportions"
__author__ = "MENGELLE Axel"
__date__ = "aout 2024"

from utils import cartesian_product

import numpy as np
import pandas as pd



# def get_geocodes_proportions(ngeocodes, grid_geo, grid_mask, geocodes )
    # prop_ti = np.zeros(ngeocodes)
    # prop_ref = np.zeros(ngeocodes)
    # for i in range(ngeocodes):
        # prop_ref[i] = np.sum(grid_geo == geocodes[i]) / np.prod(grid_msk.shape) * 100 #computing the proportions (percentages) of cells corresponding to specific geocodes within a reference grid (grid_geo) 
        # prop_ti[i] = np.sum((grid_geo * grid_msk) == geocodes[i]) / np.sum(grid_msk) * 100 #computing the proportions (percentages) of cells corresponding to specific geocodes within a reference grid (grid_geo but masked)
    
    # stats_check = pd.DataFrame(columns=['geocodes', 'prop_ref', 'prop_ti'])
    # stats_check['geocodes'] = geocodes
    # stats_check['prop_ref'] = prop_ref
    # stats_check['prop_ti'] = prop_ti
    
    # return stats_check


def get_bins(nbins, auxTI_var, auxSG_var, sim_var, simgrid_mask, eps, bintype='reg'):
    
    bins = {}
    for var_name, var_value in auxTI_var.items():
        if bintype == 'reg':
            bins[var_name] = np.linspace(np.nanmin(var_value[simgrid_mask == 1]), 
                                             np.nanmax(var_value[simgrid_mask == 1]), nbins + 1)
        elif bintype == 'pct':
            bins_pctile = np.linspace(0, 100, nbins + 1)
            bins[var_name] = np.nanpercentile(var_value.flatten(), bins_pctile)
    
    for var_name, var_value in sim_var.items():
        if bintype == 'reg':
            bins[var_name] = np.linspace(np.nanmin(var_value[simgrid_mask == 1]), 
                                             np.nanmax(var_value[simgrid_mask == 1]), nbins + 1)
        elif bintype == 'pct':
            bins_pctile = np.linspace(0, 100, nbins + 1)
            bins[var_name] = np.nanpercentile(var_value.flatten(), bins_pctile)
    
    
    for var_name, var_value in bins.items():
        bins[var_name] = var_value - eps

    return bins
    

def get_joint_dist(auxTI_var, sim_var, bins, nbins):
    n_conti_var = len(auxTI_var)
    
    n_values_categ = len(np.unique(sim_var[next(iter(sim_var))])) 

    #Initialize array of shape (nbins, nbins, ..., nbins, n_values_categ)
    joint_dist = np.zeros(tuple([nbins for _ in range(n_conti_var)] + [n_values_categ])) 
    
    #Create a list of values from 1 to nbins
    values = [i for i in range(1, nbins + 1)) 
    
    #For each continous variables, we will iterate on its bins
    #We create all the possible combinations of bins with each continuous variables by using a cartesian product
    #For example, if there are 3 bins and 4 variables, the combinations list will look like this:
    #[(1,1,1,1),(1,1,1,2),(1,1,1,3),(1,1,2,1),(1,1,2,2),...,(3,3,2,3),(3,3,3,3)]
    combinations = cartesian_product(*([values]*n_conti_var))
    
    for categi in range(n_values_categ):
        conditions = []
        for combination in combinations:
            for i, (cont_var_name, cont_var_val) in enumerate(auxTI_var.items()):
                lower_bin = bins[cont_var_name][combination[i]]
                upper_bin = bins[cont_var_name][combination[i]+1]
                conditions.append((cont_var_value > lower_bin) & (cont_var_value <= lower_bin))
            
            for categ_var_name, categ_var_val in sim_var.items()
                combined_condition = np.logical_and.reduce(conditions+[(sim_var == categi])])
                
    # return class_hist_count_joint_dist


def count_joint_marginals(joint_dist):
    """
    Calculate marginal distributions from joint distributions.

    Parameters:
    ----------
    joint_dist : ndarray
        Joint distribution array.

    Returns:
    -------
    marg_mag : ndarray
        Marginal distribution for magnetism.
    marg_grv : ndarray
        Marginal distribution for gravity.
    marg_lmp : ndarray
        Marginal distribution for lmp.
    joint_mag_grv : ndarray
        Partial joint distribution between magnetism and gravity.
    """
    marg_mag = np.sum(joint_dist, axis=(1, 2))
    marg_grv = np.sum(joint_dist, axis=(0, 2))
    marg_lmp = np.sum(joint_dist, axis=(0, 1))
    joint_mag_grv = np.sum(joint_dist, axis=2)

    return marg_mag, marg_grv, marg_lmp, joint_mag_grv


def get_prop(hist_count_mx):
    """
    Calculate proportions from counts.

    Parameters:
    ----------
    hist_count_mx : ndarray
        Array of counts.

    Returns:
    -------
    total_mx : ndarray
        Total counts.
    prop_mx : ndarray
        Proportions.
    """
    nclass = hist_count_mx.shape[-1]
    total_mx = np.sum(hist_count_mx, axis=-1)
    tmp_shape = np.asarray(hist_count_mx.shape)
    tmp_shape[-1] = 1
    tmp_shape = tuple(tmp_shape)
    tmp = np.repeat(np.reshape(total_mx, tmp_shape), nclass, axis=-1)
    prop_mx = np.zeros(hist_count_mx.shape)
    prop_mx[total_mx > eps] = hist_count_mx[total_mx > eps] / tmp[total_mx > eps]

    return total_mx, prop_mx
# -*- coding:utf-8 -*-
__projet__ = "GeoclassificationMPS"
__nom_fichier__ = "proportions"
__author__ = "MENGELLE Axel"
__date__ = "aout 2024"

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


def get_vec_bins(nbins, auxTI_var, auxSG_var, condIm_var, simgrid_mask, eps, bintype='reg'):
    
    bins_aux = {}
    for var_name, var_value in auxTI_var.items():
        if bintype == 'reg':
            bins_aux[var_name] = np.linspace(np.nanmin(var_value[simgrid_mask == 1].flatten()),
                                  np.nanmax(var_value[simgrid_mask == 1].flatten()), nbins + 1)
        elif bintype == 'pct':
            bins_pctile = np.linspace(0, 100, nbins + 1)
            bins_aux[var_name] = np.nanpercentile(var_value.flatten(), bins_pctile)
    
    for var_name, var_value in condIm_var.items():
                if bintype == 'reg':
            bins_aux[var_name] = np.linspace(np.nanmin(var_value[simgrid_mask == 1].flatten()),
                                  np.nanmax(var_value[simgrid_mask == 1].flatten()), nbins + 1)
        elif bintype == 'pct':
            bins_pctile = np.linspace(0, 100, nbins + 1)
            bins_aux[var_name] = np.nanpercentile(var_value.flatten(), bins_pctile)
    
    for i in len(bins_aux):
        bins_aux[i] = bins_aux[i] - eps
        
    return bins_aux
    
    
def count_joint_dist(ti_mag, ti_grv, ti_lmp, ti_geo, vec_mag, vec_grv, vec_lmp, geocodes):
    """
    Count joint distributions of variables and geological codes.

    Parameters:
    ----------
    ti_mag : ndarray
        Array of magnetism values.
    ti_grv : ndarray
        Array of gravity values.
    ti_lmp : ndarray
        Array of lmp values.
    ti_geo : ndarray
        Array of geological codes.
    vec_mag : ndarray
        Vector of bins for magnetism variable.
    vec_grv : ndarray
        Vector of bins for gravity variable.
    vec_lmp : ndarray
        Vector of bins for lmp variable.
    geocodes : ndarray
        Unique geological codes.

    Returns:
    -------
    class_hist_count_joint_dist : ndarray
        4D array containing counts of joint distributions.
    """
    ngeocodes = len(geocodes)
    class_hist_count_joint_dist = np.zeros((nbins, nbins, nbins, ngeocodes))

    for c in prange(ngeocodes):
        for i in prange(nbins):
            mag_lb = vec_mag[i]
            mag_ub = vec_mag[i + 1]
            for j in prange(nbins):
                grv_lb = vec_grv[j]
                grv_ub = vec_grv[j + 1]
                for k in prange(nbins):
                    lmp_lb = vec_lmp[k]
                    lmp_ub = vec_lmp[k + 1]
                    tmp_cnt = np.sum(1 * ((ti_mag > mag_lb) & (ti_mag <= mag_ub) &
                                          (ti_grv > grv_lb) & (ti_grv <= grv_ub) &
                                          (ti_lmp > lmp_lb) & (ti_lmp <= lmp_ub) &
                                          (ti_geo == geocodes[c])
                                          ))
                    class_hist_count_joint_dist[i, j, k, c] = tmp_cnt

    return class_hist_count_joint_dist


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
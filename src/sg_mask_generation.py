# -*- coding:utf-8 -*-
__projet__ = "GeoclassificationMPS"
__nom_fichier__ = "sg_mask_generation"
__author__ = "MENGELLE Axel"
__date__ = "août 2024"

import numpy as np

def create_sg_mask(sim_var, auxdesc_var, auxcond_var, cond_var, nr, nc):
    """
    Generate a simulation grid mask based on the presence of missing values across different variable types.

    This function creates a mask for a simulation grid where cells are marked based on the presence 
    of missing values (`np.nan`) in the provided variables. The mask is initialized to zeros and updated 
    to ones wherever missing values are detected in any of the variables.

    Parameters:
    ----------
    sim_var : dict
        Dictionary containing simulated variables. Each key is a variable name and each value is a NumPy array.
    auxdesc_var : dict
        Dictionary containing auxiliary descriptive variables. Each key is a variable name and each value is a NumPy array.
    auxcond_var : dict
        Dictionary containing auxiliary conditioning variables. Each key is a variable name and each value is a NumPy array.
    cond_var : dict
        Dictionary containing conditioning variables. Each key is a variable name and each value is a NumPy array.
    nr : int
        Number of rows in the simulation grid.
    nc : int
        Number of columns in the simulation grid.

    Returns:
    -------
    sim_mask : np.ndarray
        A 2D NumPy array (mask) with the same dimensions as the simulation grid. Cells with missing values in any of the variables 
        are marked with 1, while other cells remain 0.

    Notes:
    -----
    - The mask is initialized to all zeros and is updated to ones where missing values are found.
    - This function assumes that all input arrays (variables) have consistent dimensions matching the simulation grid size.
    """
    sim_mask = np.zeros((nr, nc))
    for var_name, var_value in auxdesc_var.items():
        sim_mask = np.where(np.isnan(var_value), 1, sim_mask)
    for var_name, var_value in auxcond_var.items():
        sim_mask = np.where(np.isnan(var_value), 1, sim_mask)
    return sim_mask
        
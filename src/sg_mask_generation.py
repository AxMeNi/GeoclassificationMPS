# -*- coding:utf-8 -*-
__projet__ = "GeoclassificationMPS"
__nom_fichier__ = "sg_mask_generation"
__author__ = "MENGELLE Axel"
__date__ = "août 2024"

import numpy as np
import matplotlib.pyplot as plt



def create_sg_mask(auxTI_var, auxSG_var, nr, nc):
    """
    Generate a simulation grid mask based on the presence of missing values across different variable types.

    This function creates a mask for a simulation grid where cells are marked based on the presence 
    of missing values (`np.nan`) in the provided variables. The mask is initialized to ones and updated 
    to zeros wherever missing values are detected in any of the variables.

    Parameters:
    ----------
    auxTI_var : dict
        Dictionary containing auxiliary descriptive variables. Each key is a variable name and each value is a NumPy array.
    auxSG_var : dict
        Dictionary containing auxiliary conditioning variables. Each key is a variable name and each value is a NumPy array.
    nr : int
        Number of rows in the simulation grid.
    nc : int
        Number of columns in the simulation grid.

    Returns:
    -------
    sim_mask : np.ndarray
        A 2D NumPy array (mask) with the same dimensions as the simulation grid. Cells with missing values in any of the variables 
        are marked with 0, while other cells remain 1.

    Notes:
    -----
    - The mask is initialized to all zeros and is updated to ones where missing values are found.
    - This function assumes that all input arrays (variables) have consistent dimensions matching the simulation grid size.
    """
    sim_mask = np.ones((nr, nc))
    for var_name, var_value in auxTI_var.items():
        sim_mask = np.where(np.isnan(var_value), 0, sim_mask)
    for var_name, var_value in auxSG_var.items():
        sim_mask = np.where(np.isnan(var_value), 0, sim_mask)
    return sim_mask


def merge_masks(mask1, mask2):
    """
    Merge two binary masks into a single mask.

    This function creates a new mask where:
    - The values from `mask2` are used if they are equal to 0.
    - Otherwise, the values from `mask1` are used.

    The resulting mask will have a value of 1 wherever `mask2` has a value of 1, and will retain 
    the values from `mask1` elsewhere.

    Parameters:
    ----------
    mask1 : np.ndarray
        A 2D NumPy array representing the first binary mask. It contains 0 and 1.
    mask2 : np.ndarray
        A 2D NumPy array representing the second binary mask. This mask is expected to have values of 0 or 1.

    Returns:
    -------
    np.ndarray
        A 2D NumPy array that is the result of merging `mask1` and `mask2`. The resulting mask will have 
        values of 1 where `mask2` is 1, and values from `mask1` where `mask2` is 0.

    Notes:
    -----
    - The function assumes that `mask1` and `mask2` have the same shape.
    - The merging is done in such a way that `mask2` takes precedence over `mask1` where `mask2` is 1.
    """
    if mask1.shape != mask2.shape:
        raise ValueError(f"mask {mask1} and mask {mask2} shapes are different. Cannot merge masks of different shapes.")
    mask = np.where(mask2 == 0, 0, mask1)
    return mask
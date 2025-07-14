# -*- coding:utf-8 -*-
__projet__ = "GeoclassificationMPS"
__nom_fichier__ = "utils"
__author__ = "MENGELLE Axel"
__date__ = "sept 2024"


from sklearn import manifold

import numpy as np

import pickle



def find_farthest_points_from_centroid(mds_positions, n_points=3):
    """
    Finds the n_points farthest from the centroid in a given 2D MDS representation.

    Parameters:
    -----------
    mds_positions : ndarray of shape (n_samples, 2)
        2D coordinates from the MDS representation.
    n_points : int, optional
        Number of farthest points to find. Default is 4.

    Returns:
    --------
    farthest_indices : list of int
        Indices of the n_points farthest from the centroid.
    """

    centroid = np.mean(mds_positions, axis=0)
    distances = np.linalg.norm(mds_positions - centroid, axis=1)
    farthest_indices = np.argsort(distances)[-n_points:]

    return farthest_indices.tolist()


def load_pickle_file(file_path):
    """
    Load data from a pickle file.

    Parameters:
    ----------
    file_path : str
        The path to the pickle file to be loaded.

    Returns:
    -------
    data : object
        The data loaded from the pickle file.
    """
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


def calc_min_expMax(cd, ti_list, aux_var_idx, expMax):
    """
    Determine the minimal expMax parameter for geone deesse simulation,
    expMax means "expansion maximum" and is the maximum range extension allowed for the
    values difference for a same variable in the TI and in the SG (for auxiliary variable or conditioning data).

    Parameters:
    ----------
    cd : Img geone
        conditionning image 
    ...
    
    Returns:
    -------
    expMax: float
        expMax
    """
    minti = np.nanmin(ti_list[0].val[aux_var_idx, 0, :, :])
    maxti = np.nanmax(ti_list[0].val[aux_var_idx, 0, :, :])
    mincd = np.nanmin(cd.val[-1, 0, :, :])
    maxcd = np.nanmax(cd.val[-1, 0, :, :])
    ###----## BASED ON GEONE TEST ##----###
    new_min_ti = min ( mincd, minti )
    new_max_ti = max ( maxcd, maxti )
    expMax = max((new_max_ti-new_min_ti)/(maxti-minti)-1,expMax)
    return expMax
# -*- coding:utf-8 -*-
__projet__ = "GeoclassificationMPS"
__nom_fichier__ = "simulation_functions_ter"
__author__ = "MENGELLE Axel"
__date__ = "juillet 2024"

"""
Script for processing and visualization of geospatial data.

This script imports necessary modules and initializes parameters for data processing
and visualization. It also loads pre-processed data from a pickle file.

Author: Guillaume Pirot
Date: Fri Jul 28 11:12:36 2023
"""

# import modules
import numpy as np  # NumPy for numerical operations
import pandas as pd  # Pandas for data manipulation
import matplotlib.pyplot as plt  # Matplotlib for plotting
from matplotlib.colors import LinearSegmentedColormap  # For custom colormaps
from mpl_toolkits.axes_grid1 import make_axes_locatable  # For colorbar adjustments
import pickle  # For loading pickled data
from datetime import datetime  # For handling date and time
import geone as gn  # Geostatistical library
from scipy import interpolate  # For interpolation
import sys  # For system-related functions
from numba import jit, prange  # For JIT compilation
from skimage.draw import disk  # For drawing shapes
from skimage.morphology import binary_dilation  # For morphological operations
import os  # For operating system functions
import time  # For time-related functions
from loopui import kldiv, topological_adjacency, entropy  # Custom functions
from sklearn import manifold  # For manifold learning
from scipy.cluster.hierarchy import dendrogram, linkage  # For hierarchical clustering

# Set a seed for reproducibility
myseed = 12345

# Show version of Python and geone
print(sys.version_info)
print('geone version: ' + gn.__version__)

# Define paths to directories
path2data = "./data/"
path2ti = "./ti/"
path2cd = "./cd/"
path2real = "./mpsReal/"
path2log = "./log/"
path2ind = "./ind/"

# Define suffix for filenames
suffix = "-simple"  # "", "-simple", "-very-simple"

# Define filename for pickled data
picklefn = "mt-isa-data" + suffix + ".pickle"

# Parameters for data visualization
nbins = 20
nthresholds_tpfn = 11

# DEESSE Parameters
nneighboringNode = 12
distanceThreshold = 0.1
maxScanFraction = 0.25



# Define small epsilon value
eps = np.finfo(float).eps




# %% DEESSE FUNCTIONS


def run_deesse(tiMissingGeol, mps_nreal, nneighboringNode=12, distanceThreshold=0.1, maxScanFraction=0.25, seed=444,
               nthreads=4, geolcd=True, cond_data=None):
    """
    Run the DEESSE algorithm to generate multiple point simulations (MPS).

    Parameters:
    ----------
    tiMissingGeol : geone.img.Img
        Geostatistical image object representing the training images.
    mps_nreal : int
        Number of realizations to generate.
    nneighboringNode : int, optional
        Number of neighboring nodes.
    distanceThreshold : float, optional
        Distance threshold for the algorithm.
    maxScanFraction : float, optional
        Maximum scan fraction.
    seed : int, optional
        Seed for the random number generator.
    nthreads : int, optional
        Number of threads to use.
    geolcd : bool, optional
        Flag indicating whether to include geological codes.
    cond_data : geone.img.Img or None, optional
        Conditional data object if `geolcd` is False, otherwise None.

    Returns:
    -------
    deesse_output : geone.deesseinterface.DeesseOutput
        Output object containing the results of the DEESSE algorithm.
    """
    distanceType = ['continuous'] * tiMissingGeol.nv
    distanceType[0] = 'categorical'

    if geolcd == True:
        dataImage = tiMissingGeol
    else:
        if cond_data is None:
            print('Error: missing conditioning data image.')
            return -1
        else:
            dataImage = cond_data

    deesse_input = gn.deesseinterface.DeesseInput(
        nx=tiMissingGeol.nx, ny=tiMissingGeol.ny, nz=tiMissingGeol.nz,
        sx=tiMissingGeol.sx, sy=tiMissingGeol.sy, sz=tiMissingGeol.sz,
        ox=tiMissingGeol.ox, oy=tiMissingGeol.oy, oz=tiMissingGeol.oz,
        nv=tiMissingGeol.nv, varname=tiMissingGeol.varname,
        nTI=1, TI=tiMissingGeol,
        dataImage=dataImage,
        distanceType=distanceType,
        nneighboringNode=[nneighboringNode] * tiMissingGeol.nv,
        distanceThreshold=[distanceThreshold] * tiMissingGeol.nv,
        maxScanFraction=maxScanFraction,
        npostProcessingPathMax=1,
        seed=seed,
        nrealization=mps_nreal
    )

    deesse_output = gn.deesseinterface.deesseRun(deesse_input, nthreads=nthreads)
    return deesse_output


# %% TI INDICATORS FUNCTIONS

def get_vec_bins(grid_msk, bintype='reg'):
    """
    Calculate bin vectors for variables based on grid mask.

    Parameters:
    ----------
    grid_msk : ndarray
        Mask defining areas of interest in the grid.
    bintype : {'reg', 'pct'}, optional
        Type of binning method:
        - 'reg': Regular bins based on minimum and maximum values.
        - 'pct': Bins based on percentiles.

    Returns:
    -------
    vec_mag : ndarray
        Vector of bins for magnetism variable.
    vec_grv : ndarray
        Vector of bins for gravity variable.
    vec_lmp : ndarray
        Vector of bins for lamp variable.
    """
    if bintype == 'reg':
        vec_mag = np.linspace(np.nanmin(grid_mag[grid_msk == 1].flatten()),
                              np.nanmax(grid_mag[grid_msk == 1].flatten()), nbins + 1)
        vec_grv = np.linspace(np.nanmin(grid_grv[grid_msk == 1].flatten()),
                              np.nanmax(grid_grv[grid_msk == 1].flatten()), nbins + 1)
        vec_lmp = np.linspace(np.nanmin(grid_lmp[grid_msk == 1].flatten()),
                              np.nanmax(grid_lmp[grid_msk == 1].flatten()), nbins + 1)
    elif bintype == 'pct':
        pctile_vec = np.linspace(0, 100, nbins + 1)
        vec_mag = np.nanpercentile(grid_mag.flatten(), pctile_vec)
        vec_grv = np.nanpercentile(grid_grv.flatten(), pctile_vec)
        vec_lmp = np.nanpercentile(grid_lmp.flatten(), pctile_vec)

    vec_mag[0] = vec_mag[0] - eps
    vec_grv[0] = vec_grv[0] - eps
    vec_lmp[0] = vec_lmp[0] - eps

    return vec_mag, vec_grv, vec_lmp


@jit(nopython=True)
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
        Array of lamp values.
    ti_geo : ndarray
        Array of geological codes.
    vec_mag : ndarray
        Vector of bins for magnetism variable.
    vec_grv : ndarray
        Vector of bins for gravity variable.
    vec_lmp : ndarray
        Vector of bins for lamp variable.
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
        Marginal distribution for lamp.
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


import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn import manifold
import pandas as pd


def plot_marginals(marg_mag, marg_grv, marg_lmp, suptitle):
    """
    Plot marginal histograms for magnetism, gravity, and lamp variables.

    Parameters:
    ----------
    marg_mag : ndarray
        Marginal histogram for magnetism.
    marg_grv : ndarray
        Marginal histogram for gravity.
    marg_lmp : ndarray
        Marginal histogram for lamp.
    suptitle : str
        Title for the plot.
    """
    nbins = marg_mag.shape[0]
    plt.subplots(1, 3, figsize=(12, 5), dpi=300)

    plt.subplot(1, 3, 1), plt.title('Marginal hist count mag'), plt.imshow(marg_mag, origin='lower',
                                                                           cmap='Blues'), plt.xlabel(
        'lithocode'), plt.ylabel('bin number'), plt.ylim((0.5, nbins - 0.5))
    plt.subplot(1, 3, 2), plt.title('Marginal hist count grv'), plt.imshow(marg_grv, origin='lower',
                                                                           cmap='Blues'), plt.xlabel(
        'lithocode'), plt.ylabel('bin number'), plt.ylim((0.5, nbins - 0.5))
    plt.subplot(1, 3, 3), plt.title('Marginal hist count 1vd'), plt.imshow(marg_lmp, origin='lower',
                                                                           cmap='Blues'), plt.xlabel(
        'lithocode'), plt.ylabel('bin number'), plt.ylim((0.5, nbins - 0.5))

    plt.suptitle(suptitle)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.show()


def shannon_entropy(prop_mx):
    """
    Calculate Shannon's entropy from proportions.

    Parameters:
    ----------
    prop_mx : ndarray
        Proportions matrix.

    Returns:
    -------
    H : ndarray
        Shannon's entropy values.
    """
    nclass = prop_mx.shape[-1]
    tmp_pi = prop_mx
    tmp_logpi = np.zeros(tmp_pi.shape)
    tmp_logpi[tmp_pi > 0] = np.log(prop_mx[tmp_pi > 0]) / np.log(nclass)
    H = -np.sum(tmp_pi * tmp_logpi, axis=-1)
    H[H <= 0] = 0
    return H


def plot_shannon_entropy_marginals(shannon_entropy_marg, shannon_entropy_labl, shannon_entropy_joint):
    """
    Plot Shannon's entropy for marginals and joint distributions.

    Parameters:
    ----------
    shannon_entropy_marg : ndarray
        Shannon's entropy for marginals.
    shannon_entropy_labl : list
        Labels for Shannon's entropy marginals.
    shannon_entropy_joint : ndarray
        Shannon's entropy for joint distributions.
    """
    nbins = shannon_entropy_marg.shape[1]
    bins = np.linspace(1, nbins, nbins)

    plt.subplots(1, 2, figsize=(8, 3.5), dpi=300)

    plt.subplot(1, 2, 1), plt.title("Marginal Shannon's entropy")
    plt.plot(bins, shannon_entropy_marg.T)
    plt.legend(shannon_entropy_labl, loc='best')
    plt.xlabel('marginal distribution bins'), plt.ylabel("Shannon's entropy"), plt.ylim((0, 1))

    ax = plt.subplot(1, 2, 2)
    plt.title("Joint Shannon's entropy")
    im = plt.imshow(shannon_entropy_joint, origin='lower', extent=[0.5, nbins - 0.5, 0.5, nbins - 0.5], cmap='Blues',
                    vmin=0, vmax=1)
    plt.xlabel('grv bins'), plt.ylabel('mag bins')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    plt.subplots_adjust(bottom=0.1, right=1.0, top=0.9)
    plt.show()


def get_jsdist_all(prop_joint_dist, prop_marg_mag, prop_marg_grv, prop_marg_lmp, prop_joint_mag_grv):
    """
    Calculate Jensen-Shannon divergences for all distributions.

    Parameters:
    ----------
    prop_joint_dist : ndarray
        Proportions of joint distributions.
    prop_marg_mag : ndarray
        Proportions of marginal distribution for magnetism.
    prop_marg_grv : ndarray
        Proportions of marginal distribution for gravity.
    prop_marg_lmp : ndarray
        Proportions of marginal distribution for lamp.
    prop_joint_mag_grv : ndarray
        Proportions of joint distribution between magnetism and gravity.

    Returns:
    -------
    jsdist_joint_dist : ndarray
        Jensen-Shannon divergences for joint distributions.
    jsdist_marg_mag : ndarray
        Jensen-Shannon divergences for marginal distribution of magnetism.
    jsdist_marg_grv : ndarray
        Jensen-Shannon divergences for marginal distribution of gravity.
    jsdist_marg_lmp : ndarray
        Jensen-Shannon divergences for marginal distribution of lamp.
    jsdist_joint_mag_grv : ndarray
        Jensen-Shannon divergences for joint distribution between magnetism and gravity.
    """
    ngeocodes = prop_joint_dist.shape[-1]
    jsdist_joint_dist = np.zeros((ngeocodes, ngeocodes))
    jsdist_marg_mag = np.zeros((ngeocodes, ngeocodes))
    jsdist_marg_grv = np.zeros((ngeocodes, ngeocodes))
    jsdist_marg_lmp = np.zeros((ngeocodes, ngeocodes))
    jsdist_joint_mag_grv = np.zeros((ngeocodes, ngeocodes))

    base = ngeocodes
    for i in range(ngeocodes):
        for j in range(i):
            jsdist_joint_dist[i, j] = jsdist_joint_dist[j, i] = (
                kldiv(prop_joint_dist[..., i].flatten(), prop_joint_dist[..., j].flatten(), base, 'js')
            )
            jsdist_marg_mag[i, j] = jsdist_marg_mag[j, i] = (
                kldiv(prop_marg_mag[..., i].flatten(), prop_marg_mag[..., j].flatten(), base, 'js')
            )
            jsdist_marg_grv[i, j] = jsdist_marg_grv[j, i] = (
                kldiv(prop_marg_grv[..., i].flatten(), prop_marg_grv[..., j].flatten(), base, 'js')
            )
            jsdist_marg_lmp[i, j] = jsdist_marg_lmp[j, i] = (
                kldiv(prop_marg_lmp[..., i].flatten(), prop_marg_lmp[..., j].flatten(), base, 'js')
            )
            jsdist_joint_mag_grv[i, j] = jsdist_joint_mag_grv[j, i] = (
                kldiv(prop_joint_mag_grv[..., i].flatten(), prop_joint_mag_grv[..., j].flatten(), base, 'js')
            )

    return jsdist_joint_dist, jsdist_marg_mag, jsdist_marg_grv, jsdist_marg_lmp, jsdist_joint_mag_grv


mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=myseed,
                   dissimilarity="precomputed", n_jobs=1)


def plot_jsdivmx_mds_hist(geocodes, jsdist_mx, prefix, class_hist_count=None):
    """
    Plot Jensen-Shannon divergence matrix using Multi-Dimensional Scaling (MDS).

    Parameters:
    ----------
    geocodes : ndarray
        Geological codes.
    jsdist_mx : ndarray
        Jensen-Shannon divergence matrix.
    prefix : str
        Prefix for plot title.
    class_hist_count : ndarray, optional
        Histogram counts for classification.

    Returns:
    -------
    None
    """
    ngeocodes = len(geocodes)
    mdspos_jsdist_mx = mds.fit(jsdist_mx).embedding_

    bx = (mdspos_jsdist_mx[:, 0].max() - mdspos_jsdist_mx[:, 0].min()) / 100
    by = (mdspos_jsdist_mx[:, 1].max() - mdspos_jsdist_mx[:, 1].min()) / 100

    plt.subplots(1, 3, figsize=(13, 3.5), dpi=300)

    plt.subplot(1, 3, 1), plt.title(prefix + ' - Jensen Shannon Divergence'), plt.xlabel('lithocodes'), plt.ylabel(
        'lithocodes')
    plt.imshow(jsdist_mx, origin='upper', cmap='Reds', extent=[0.5, ngeocodes + 0.5, ngeocodes + 0.5, 0.5])

    ax = plt.subplot(1, 3, 2)
    plt.title(prefix + ' - Multi-Dimensional-Scaling'), plt.xlabel('MDS component 1'), plt.ylabel('MDS component 2')
    im = plt.scatter(mdspos_jsdist_mx[:, 0], mdspos_jsdist_mx[:, 1], c=geocodes, cmap=mycmap,
                     s=100, label='lithocode hist', marker='+', vmin=0.5, vmax=11.5)

    for i in range(ngeocodes):
        plt.text(mdspos_jsdist_mx[i, 0] + bx, mdspos_jsdist_mx[i, 1] + by, str(geocodes[i]), size=12,
                 color=myclrs[i, :])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    plt.subplot(1, 3, 3)
    if class_hist_count is None:
        plt.axis('off')
    elif len(class_hist_count.shape) == 2:
        plt.title(prefix + ' - Histograms'), plt.xlabel('lithocodes'), plt.ylabel('bins')
        plt.imshow(class_hist_count, origin='lower', cmap='Blues', extent=[0.5, ngeocodes + 0.5, 0.5, nbins + 0.5])
    else:
        plt.axis('off')

    plt.show()


def get_confusion_matrix(realizations, reference, mask):
    """
    Calculate confusion matrix from realizations and reference data.

    Parameters:
    ----------
    realizations : ndarray
        Realizations data.
    reference : ndarray
        Reference data.
    mask : ndarray
        Mask for training and testing.

    Returns:
    -------
    confusion_matrix : ndarray
        Confusion matrix.
    classes : list
        Classes list.
    training_size : ndarray
        Training sizes.
    testing_size : ndarray
        Testing sizes.
    """
    classes = np.unique(reference)
    nclasses = len(classes)
    nreal = realizations.shape[-1]
    tmp_msk = np.tile(np.reshape(mask, (ny, nx, 1)), (1, 1, nreal))
    confusion_matrix = np.ones((nclasses, nclasses)) * np.nan
    training_size = np.zeros(nclasses)
    testing_size = np.zeros(nclasses)

    for i in range(nclasses):
        tmp = reference == classes[i]
        training_size[i] = np.sum(tmp * (1 - mask))
        testing_size[i] = np.sum(tmp * mask * nreal)
        tmp_ref = np.tile(np.reshape(tmp, (ny, nx, 1)), (1, 1, nreal))

        for j in range(nclasses):
            tmp_sim = realizations == classes[j]
            confusion_matrix[i, j] = np.sum((tmp_sim * tmp_ref * tmp_msk).flatten())

    return confusion_matrix.astype(int), classes, training_size, testing_size


def get_true_false_pos_neg(confusion_matrix, classes, training_size, testing_size):
    """
    Calculate True/False Positive/Negative rates from confusion matrix.

    Parameters:
    ----------
    confusion_matrix : ndarray
        Confusion matrix.
    classes : list
        Classes list.
    training_size : ndarray
        Training sizes.
    testing_size : ndarray
        Testing sizes.

    Returns:
    -------
    tfpn : DataFrame
        True/False Positive/Negative rates per lithocode.
    """
    tfpn = pd.DataFrame(columns=['code', 'TP', 'FP', 'TN', 'FN', 'training', 'testing'])
    tfpn['code'] = classes
    tfpn['TP'] = np.diag(confusion_matrix)
    tfpn['FP'] = np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix)
    tfpn['FN'] = np.sum(confusion_matrix, axis=1) - np.diag(confusion_matrix)
    tfpn['TN'] = np.sum(confusion_matrix.flatten()) + np.diag(confusion_matrix) - np.sum(confusion_matrix,
                                                                                         axis=0) - np.sum(
        confusion_matrix, axis=1)
    tfpn['training'] = training_size
    tfpn['testing'] = testing_size
    return tfpn


def classification_performance(true_false_pos_neg_df):
    """
    Calcule les métriques de performance de classification à partir d'un DataFrame contenant les valeurs de TP, TN, FP et FN.

    Paramètres:
    true_false_pos_neg_df (pd.DataFrame): DataFrame contenant les colonnes 'TP', 'TN', 'FP', 'FN'.

    Retour:
    None: Met à jour le DataFrame en ajoutant des colonnes pour les métriques de performance.
    """
    TP = true_false_pos_neg_df['TP'].values.astype(float)
    TN = true_false_pos_neg_df['TN'].values.astype(float)
    FP = true_false_pos_neg_df['FP'].values.astype(float)
    FN = true_false_pos_neg_df['FN'].values.astype(float)

    # Calcul des métriques de performance
    true_false_pos_neg_df['accuracy'] = (TP + TN) / (TP + TN + FP + FN)
    true_false_pos_neg_df['precision'] = TP / (TP + FP)
    true_false_pos_neg_df['recall'] = TP / (TP + FN)
    true_false_pos_neg_df['F1'] = 2 * true_false_pos_neg_df['precision'] * true_false_pos_neg_df['recall'] / (
            true_false_pos_neg_df['precision'] + true_false_pos_neg_df['recall'])
    true_false_pos_neg_df['MCC'] = ((TP * TN - FP * FN) /
                                    np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
    return


def get_confusion_matrix_th(realizations, reference, mask, nthresholds):
    """
    Calcule les matrices de confusion pour plusieurs seuils.

    Paramètres:
    realizations (np.array): Réalisations simulées.
    reference (np.array): Référence pour la classification.
    mask (np.array): Masque définissant les zones d'intérêt.
    nthresholds (int): Nombre de seuils à considérer.

    Retour:
    tuple: (confusion_matrix, classes, thresholds)
        confusion_matrix (np.array): Matrices de confusion pour chaque seuil.
        classes (np.array): Classes uniques présentes dans la référence.
        thresholds (np.array): Seuils utilisés pour le calcul.
    """
    classes = np.unique(reference)
    nclasses = len(classes)
    nreal = realizations.shape[-1]
    thresholds = np.linspace(0, 1, nthresholds)
    tmp_msk = mask
    confusion_matrix = np.ones((2, 2, nclasses, nthresholds)) * np.nan

    for i in range(nclasses):
        tmp_ref = reference == classes[i]
        for k in range(nthresholds):
            probmap = np.mean(realizations == classes[i], axis=2)
            tmp_sim = probmap >= thresholds[k]
            confusion_matrix[0, 0, i, k] = np.sum(mask * tmp_ref * tmp_sim)  # TP
            confusion_matrix[0, 1, i, k] = np.sum(mask * (1 - tmp_ref) * tmp_sim)  # FP
            confusion_matrix[1, 0, i, k] = np.sum(mask * tmp_ref * (1 - tmp_sim))  # FN
            confusion_matrix[1, 1, i, k] = np.sum(mask * (1 - tmp_ref) * (1 - tmp_sim))  # TN

    return confusion_matrix.astype(int), classes, thresholds


def get_tpr_fpr(confusion_matrix_th):
    """
    Calcule les taux de vrais positifs (TPR) et les taux de faux positifs (FPR) à partir des matrices de confusion.

    Paramètres:
    confusion_matrix_th (np.array): Matrices de confusion pour chaque seuil.

    Retour:
    tuple: (TPR, FPR)
        TPR (np.array): Taux de vrais positifs pour chaque classe et chaque seuil.
        FPR (np.array): Taux de faux positifs pour chaque classe et chaque seuil.
    """
    TP = confusion_matrix_th[0, 0, :, :]
    FP = confusion_matrix_th[0, 1, :, :]
    FN = confusion_matrix_th[1, 0, :, :]
    TN = confusion_matrix_th[1, 1, :, :]
    TPR = TP / (TP + FN)
    FPR = FP / (TN + FP)
    return TPR, FPR


def plot_real_and_ref(realizations, reference, mask, nrealmax=3, addtitle=''):
    """
    Trace les réalisations simulées et la référence.

    Paramètres:
    realizations (np.array): Réalisations simulées.
    reference (np.array): Référence pour la classification.
    mask (np.array): Masque définissant les zones d'intérêt.
    nrealmax (int, optionnel): Nombre maximum de réalisations à tracer. Par défaut, 3.
    addtitle (str, optionnel): Titre additionnel à ajouter aux sous-titres des graphes.

    Retour:
    None: Affiche les graphes.
    """
    shrinkfactor = 0.55
    plot_msk2 = 1 - mask
    nr2plot = np.min([nrealmax, 3])
    ncol = nr2plot + 1
    plt.figure(figsize=(5 * ncol, 10), dpi=300)

    for i in range(nr2plot):
        plt.subplot(1, ncol, i + 1)
        plt.title('Real #' + str(i) + ' ' + addtitle)
        tmp = realizations[:, :, i]
        im = plt.imshow(tmp, origin='lower', cmap=mycmap, interpolation='none', vmin=0.5, vmax=11.5)
        plt.imshow(plot_msk2, origin='lower', cmap='gray', alpha=0.3)
        plt.axis('off')
        plt.colorbar(im, shrink=shrinkfactor)

    plt.subplot(1, ncol, nr2plot + 1)
    plt.title('Reference')
    im = plt.imshow(reference, origin='lower', cmap=mycmap, interpolation='none', vmin=0.5, vmax=11.5)
    plt.imshow(plot_msk2, origin='lower', cmap='gray', alpha=0.3)
    plt.axis('off')
    plt.colorbar(im, shrink=shrinkfactor)
    plt.show()


def plot_entropy_and_confusionmx(geocode_entropy, confusion_matrix, mps_nreal):
    """
    Trace l'entropie des codes géologiques et la matrice de confusion.

    Paramètres:
    geocode_entropy (np.array): Entropie des codes géologiques.
    confusion_matrix (np.array): Matrice de confusion des lithocodes.
    mps_nreal (int): Nombre de réalisations MPS.

    Retour:
    None: Affiche les graphes.
    """
    plt.subplots(1, 2, figsize=(12, 8), dpi=300)
    shrinkfactor = 0.8

    plt.subplot(1, 2, 1)
    im = plt.imshow(geocode_entropy, origin='lower', extent=grid_ext, cmap='Reds')
    plt.xlabel('Easting')
    plt.ylabel('Northing')
    plt.title('Geological code Entropy over ' + str(mps_nreal) + ' realizations')
    plt.colorbar(im, shrink=shrinkfactor)

    plt.subplot(1, 2, 2)
    im = plt.imshow(np.log10(confusion_matrix), interpolation='none', cmap='Greens')
    plt.title('Lithocode log$_{10}$ confusion matrix')
    plt.xlabel('predicted lithocode')
    plt.ylabel('reference lithocode')
    cb = plt.colorbar(im, shrink=shrinkfactor)
    cb.set_label('log$_{10}$ count')
    plt.show()
    return


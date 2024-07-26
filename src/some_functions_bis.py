# -*- coding:utf-8 -*-
__projet__ = "missing-data"
__nom_fichier__ = "test_functions_bis"
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

# Create directories if they do not exist
if not os.path.exists(path2ti):
    os.makedirs(path2ti)
if not os.path.exists(path2cd):
    os.makedirs(path2cd)
if not os.path.exists(path2real):
    os.makedirs(path2real)
if not os.path.exists(path2log):
    os.makedirs(path2log)
if not os.path.exists(path2ind):
    os.makedirs(path2ind)

# Load pre-processed data from pickle file
pickledestination = path2data + picklefn
with open(pickledestination, 'rb') as f:
    [_, grid_geo, grid_lmp, grid_mag,
     grid_grv, grid_ext, vec_x, vec_y
     ] = pickle.load(f)

print("#######", " \nGrid_geo", grid_geo, " \nGrid_lmp",grid_lmp, " \nGrid_mag",grid_mag, " \nGrid_grv",grid_grv, " \nGrid_ext",grid_ext, " \nvec_x",vec_x, " \nvec_y",vec_y)


# Extract dimensions of grid_geo
ny, nx = grid_geo.shape

# Define custom colormap
cm = plt.get_cmap('tab20')
myclrs = np.asarray(cm.colors)[[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11], :]
n_bin = 11
cmap_name = 'my_tab20'
mycmap = LinearSegmentedColormap.from_list(cmap_name, myclrs, N=n_bin)
ticmap = LinearSegmentedColormap.from_list('ticmap', np.vstack(([0, 0, 0], myclrs)), N=n_bin + 1)

# Define small epsilon value
eps = np.finfo(float).eps


# %% DEESSE FUNCTIONS

def gen_ti_mask(nx, ny, ti_pct_area, ti_ndisks, seed):
    """
    Generate a binary mask representing multiple disks within a grid.

    Parameters:
    ----------
    nx : int
        Number of columns in the grid.
    ny : int
        Number of rows in the grid.
    ti_pct_area : float
        Percentage of the grid area to cover with disks.
    ti_ndisks : int
        Number of disks to generate.
    seed : int
        Seed for the random number generator.

    Returns:
    -------
    mask : ndarray
        Binary mask with 1s indicating disk positions within the grid.
    """
    rng = np.random.default_rng(seed=seed)
    rndy = rng.integers(low=0, high=ny, size=ti_ndisks)
    rndx = rng.integers(low=0, high=nx, size=ti_ndisks)
    radius = np.floor(np.sqrt((nx * ny * ti_pct_area / 100 / ti_ndisks) / np.pi))
    mask = np.zeros((ny, nx))
    for i in range(ti_ndisks):
        rr, cc = disk((rndy[i], rndx[i]), radius, shape=(ny, nx))
        mask[rr, cc] = 1
    check_pct = np.sum(mask.flatten()) / (nx * ny) * 100
    while check_pct < ti_pct_area:
        mask = binary_dilation(mask)
        check_pct = np.sum(mask.flatten()) / (nx * ny) * 100
    return mask


def build_ti(grid_msk, ti_ndisks, ti_pct_area, ti_realid, geolcd=True, xycv=False):
    """
    Build training images (TI) based on grid masks and other parameters.

    Parameters:
    ----------
    grid_msk : ndarray
        Mask defining areas of interest in the grid.
    ti_ndisks : int
        Number of disks to generate for the training images.
    ti_pct_area : float
        Percentage of the grid area to cover with disks.
    ti_realid : int
        Realization ID.
    geolcd : bool, optional
        Flag indicating whether to include geological codes.
    xycv : bool, optional
        Flag indicating whether to include x and y coordinates.

    Returns:
    -------
    geocodes : ndarray
        Unique geological codes.
    ngeocodes : int
        Number of unique geological codes.
    tiMissingGeol : geone.img.Img
        Geostatistical image object representing the training images.
    cond_data : geone.img.Img or None
        Conditional data object if `geolcd` is False, otherwise None.
    """
    geocodes = np.unique(grid_geo)
    ngeocodes = len(geocodes)
    novalue = -9999999
    nz = 1
    sx = vec_x[1] - vec_x[0]
    sy = vec_y[1] - vec_y[0]
    sz = sx
    ox = vec_x[0]
    oy = vec_y[0]
    oz = 0.0

    if xycv == False:
        nv = 4
        varname = ['geo', 'grv', 'mag', 'lmp']

    else:
        nv = 6
        varname = ['geo', 'grv', 'mag', 'lmp', 'x', 'y']
        xx, yy = np.meshgrid(vec_x, vec_y, indexing='xy')
    name = path2ti + 'ti' + suffix + '-ndisks-' + str(ti_ndisks) + '-areapct-' + str(ti_pct_area) + '-r-' + str(
        ti_realid) + '-geolcd' + str(geolcd) + '-xycv' + str(xycv) + '.gslib'
        
        
    val = np.ones((nv, nz, ny, nx)) * np.nan
    grid_geo_masked = grid_geo + 0
    grid_geo_masked[grid_msk < 1] = novalue
    val[0, 0, :, :] = grid_geo_masked
    val[1, 0, :, :] = grid_grv
    val[2, 0, :, :] = grid_mag
    val[3, 0, :, :] = grid_lmp

    if xycv == True:
        val[4, 0, :, :] = xx
        val[5, 0, :, :] = yy
    # Create the Img class object
    tiMissingGeol = gn.img.Img(nx, ny, nz, sx, sy, sz, ox, oy, oz, nv, val, varname, name)

    if geolcd == False:
        val2 = val + 0 # Val2 is a copy of val but not at the same memory location
        val2[0, 0, :, :] = novalue
        cdname = path2cd + 'ti' + suffix + '-ndisks-' + str(ti_ndisks) + '-areapct-' + str(ti_pct_area) + '-r-' + str(
            ti_realid) + '-geolcd' + str(geolcd) + '-xycv' + str(xycv) + '.gslib'
        cond_data = gn.img.Img(nx, ny, nz, sx, sy, sz, ox, oy, oz, nv, val2, varname, cdname)

    else:
        cond_data = None
    gn.img.writeImageGslib(im=tiMissingGeol, filename=name, missing_value=None, fmt="%.10g")
    return geocodes, ngeocodes, tiMissingGeol, cond_data


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
        Vector of bins for lmp variable.
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


import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn import manifold
import pandas as pd


def plot_marginals(marg_mag, marg_grv, marg_lmp, suptitle):
    """
    Plot marginal histograms for magnetism, gravity, and lmp variables.

    Parameters:
    ----------
    marg_mag : ndarray
        Marginal histogram for magnetism.
    marg_grv : ndarray
        Marginal histogram for gravity.
    marg_lmp : ndarray
        Marginal histogram for lmp.
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
        Proportions of marginal distribution for lmp.
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
        Jensen-Shannon divergences for marginal distribution of lmp.
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


def main(ti_pct_area, ti_ndisks, ti_realid, mps_nreal, nthreads, geolcd=True, xycv=False, timesleep=0, verb=False,
         addtitle=''):
    """
    Main function for generating and analyzing training images (TI) in geostatistics.

    Parameters:
    ti_pct_area (float): Percentage of the area to be used for TI.
    ti_ndisks (int): Number of disks to be used for TI.
    ti_realid (int): Realization ID for TI.
    mps_nreal (int): Number of multiple-point statistics realizations.
    nthreads (int): Number of threads to use.
    geolcd (bool): Use geological codes (default is True).
    xycv (bool): Use cross-validation on x and y coordinates (default is False).
    timesleep (int): Time to sleep before starting (default is 0).
    verb (bool): Verbose mode, if True, will plot intermediate results (default is False).
    addtitle (str): Additional title for plots (default is empty).

    Returns:
    None
    """
    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)') + " - INIT")
    time.sleep(timesleep)

    # GENERATE TI MASK
    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)') + " - GENERATE MASK")
    grid_msk = gen_ti_mask(nx, ny, ti_pct_area, ti_ndisks, myseed + ti_realid)

    # PLOT TI MASK
    if verb:
        print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)') + " - PLOT MASK")
        plt.figure(dpi=300), plt.title('TI mask')
        plt.imshow(grid_msk, origin='lower', interpolation='none')
        plt.show()

    # BUILD TI
    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)') + " - BUILD TI")
    geocodes, ngeocodes, tiMissingGeol, cond_data = build_ti(grid_msk, ti_ndisks, ti_pct_area, ti_realid, geolcd, xycv)

    # COMPUTE TI INDICATORS
    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)') + " - COMPUTE TI INDICATORS")

    prop_ti = np.zeros(ngeocodes)
    prop_ref = np.zeros(ngeocodes)
    for i in range(ngeocodes):
        prop_ref[i] = np.sum(grid_geo == geocodes[i]) / np.prod(grid_msk.shape) * 100
        prop_ti[i] = np.sum((grid_geo * grid_msk) == geocodes[i]) / np.sum(grid_msk) * 100

    stats_check = pd.DataFrame(columns=['geocodes', 'prop_ref', 'prop_ti'])
    stats_check['geocodes'] = geocodes
    stats_check['prop_ref'] = prop_ref
    stats_check['prop_ti'] = prop_ti
    print(stats_check)
    print('TI coverage: %.1f%% of the total area.' % (np.sum(grid_msk) / np.prod(grid_msk.shape) * 100))

    # COUNT BASED ON REGULARLY SPACED BINS
    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)') + " - COUNT BASED ON REGULARLY SPACED BINS")
    vec_mag, vec_grv, vec_lmp = get_vec_bins(grid_msk, bintype='reg')
    class_hist_count_joint_dist = count_joint_dist(grid_mag[grid_msk == 1].flatten(),
                                                   grid_grv[grid_msk == 1].flatten(),
                                                   grid_lmp[grid_msk == 1].flatten(),
                                                   grid_geo[grid_msk == 1].flatten(),
                                                   vec_mag, vec_grv, vec_lmp, geocodes)

    [class_hist_count_marg_mag,
     class_hist_count_marg_grv,
     class_hist_count_marg_lmp,
     class_hist_count_joint_mag_grv] = count_joint_marginals(class_hist_count_joint_dist)
    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)') + " - COUNT BASED ON REGULARLY SPACED BINS - DONE")

    # COUNT BASED ON PERCENTILES
    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)') + " - COUNT BASED ON PERCENTILES")
    pctile_mag, pctile_grv, pctile_lmp = get_vec_bins(grid_msk, bintype='pct')
    class_hist_count_pct_joint_dist = count_joint_dist(grid_mag[grid_msk == 1].flatten(),
                                                       grid_grv[grid_msk == 1].flatten(),
                                                       grid_lmp[grid_msk == 1].flatten(),
                                                       grid_geo[grid_msk == 1].flatten(),
                                                       pctile_mag, pctile_grv, pctile_lmp, geocodes)

    [class_hist_count_pct_marg_mag,
     class_hist_count_pct_marg_grv,
     class_hist_count_pct_marg_lmp,
     class_hist_count_pct_joint_mag_grv] = count_joint_marginals(class_hist_count_pct_joint_dist)
    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)') + " - COUNT BASED ON PERCENTILES - DONE")

    # GET PROPORTIONS AND TOTAL
    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)') + " - GET PROPORTIONS AND TOTAL")
    class_hist_total_joint_dist, class_hist_prop_joint_dist = get_prop(class_hist_count_joint_dist)
    class_hist_total_marg_mag, class_hist_prop_marg_mag = get_prop(class_hist_count_marg_mag)
    class_hist_total_marg_grv, class_hist_prop_marg_grv = get_prop(class_hist_count_marg_grv)
    class_hist_total_marg_lmp, class_hist_prop_marg_lmp = get_prop(class_hist_count_marg_lmp)
    class_hist_total_joint_mag_grv, class_hist_prop_joint_mag_grv = get_prop(class_hist_count_joint_mag_grv)

    class_hist_total_pct_joint_dist, class_hist_prop_pct_joint_dist = get_prop(class_hist_count_pct_joint_dist)
    class_hist_total_pct_marg_mag, class_hist_prop_pct_marg_mag = get_prop(class_hist_count_pct_marg_mag)
    class_hist_total_pct_marg_grv, class_hist_prop_pct_marg_grv = get_prop(class_hist_count_pct_marg_grv)
    class_hist_total_pct_marg_lmp, class_hist_prop_pct_marg_lmp = get_prop(class_hist_count_pct_marg_lmp)
    class_hist_total_pct_joint_mag_grv, class_hist_prop_pct_joint_mag_grv = get_prop(class_hist_count_pct_joint_mag_grv)
    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)') + " - GET PROPORTIONS AND TOTAL - DONE")

    # PLOT MARGINALS IF VERBOSE
    if verb:
        plot_marginals(class_hist_count_marg_mag, class_hist_count_marg_grv, class_hist_count_marg_lmp,
                       'COUNT BASED ON REGULARLY SPACED BINS')
        plot_marginals(class_hist_count_pct_marg_mag, class_hist_count_pct_marg_grv, class_hist_count_pct_marg_lmp,
                       'COUNT BASED ON PERCENTILES')

    # COMPUTE ENTROPY
    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)') + " - ENTROPY BASED ON REGULARLY SPACED BINS")
    shannon_entropy_joint_dist = shannon_entropy(class_hist_prop_joint_dist)
    shannon_entropy_marg_mag = shannon_entropy(class_hist_prop_marg_mag)
    shannon_entropy_marg_grv = shannon_entropy(class_hist_prop_marg_grv)
    shannon_entropy_marg_lmp = shannon_entropy(class_hist_prop_marg_lmp)
    shannon_entropy_joint_mag_grv = shannon_entropy(class_hist_prop_joint_mag_grv)

    shannon_entropy_marg = np.vstack((shannon_entropy_marg_mag, shannon_entropy_marg_grv, shannon_entropy_marg_lmp))
    shannon_entropy_labl = ['mag', 'grv', '1vd']

    if verb:
        plot_shannon_entropy_marginals(shannon_entropy_marg, shannon_entropy_labl, shannon_entropy_joint_mag_grv)
    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)') + " - ENTROPY BASED ON REGULARLY SPACED BINS - DONE")

    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)') + " - ENTROPY BASED ON PERCENTILES")
    shannon_entropy_pct_joint_dist = shannon_entropy(class_hist_prop_pct_joint_dist)
    shannon_entropy_pct_marg_mag = shannon_entropy(class_hist_prop_pct_marg_mag)
    shannon_entropy_pct_marg_grv = shannon_entropy(class_hist_prop_pct_marg_grv)
    shannon_entropy_pct_marg_lmp = shannon_entropy(class_hist_prop_pct_marg_lmp)
    shannon_entropy_pct_joint_mag_grv = shannon_entropy(class_hist_prop_pct_joint_mag_grv)

    shannon_entropy_pct_marg = np.vstack(
        (shannon_entropy_pct_marg_mag, shannon_entropy_pct_marg_grv, shannon_entropy_pct_marg_lmp))
    shannon_entropy_labl = ['mag', 'grv', '1vd']

    if verb:
        plot_shannon_entropy_marginals(shannon_entropy_pct_marg, shannon_entropy_labl,
                                       shannon_entropy_pct_joint_mag_grv)
    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)') + " - ENTROPY BASED ON PERCENTILES - DONE")

    # COMPUTE HISTOGRAM DISSIMILARITY BETWEEN CLASSES FOR MARGINALS AND JOINT DISTRIBUTIONS
    print((datetime.now()).strftime(
        '%d-%b-%Y (%H:%M:%S)') + " - COMPUTE JENSEN SHANNON DIVERGENCE BETWEEEN DISTRIBUTIONS")
    [jsdist_joint_dist,
     jsdist_marg_mag,
     jsdist_marg_grv,
     jsdist_marg_lmp,
     jsdist_joint_mag_grv] = get_jsdist_all(class_hist_prop_joint_dist,
                                            class_hist_prop_marg_mag,
                                            class_hist_prop_marg_grv,
                                            class_hist_prop_marg_lmp,
                                            class_hist_prop_joint_mag_grv)
    print((datetime.now()).strftime(
        '%d-%b-%Y (%H:%M:%S)') + " - COMPUTE JENSEN SHANNON DIVERGENCE BETWEEEN DISTRIBUTIONS - DONE")

    if verb:
        # plot_jsdivmx_mds_hist(geocodes,jsdist_mx,prefix,class_hist_count=None)
        plot_jsdivmx_mds_hist(geocodes, jsdist_marg_mag, 'Mag', class_hist_count_marg_mag)
        plot_jsdivmx_mds_hist(geocodes, jsdist_marg_grv, 'Grv', class_hist_count_marg_grv)
        plot_jsdivmx_mds_hist(geocodes, jsdist_marg_lmp, '1vd', class_hist_count_marg_lmp)
        plot_jsdivmx_mds_hist(geocodes, jsdist_joint_dist, 'Joint-all')  # ,class_hist_count_joint_dist
        plot_jsdivmx_mds_hist(geocodes, jsdist_joint_mag_grv, 'Mag-Grv')  # ,class_hist_count_joint_mag_grv

    # TAKING INTO ACCOUNT TOPOLOGICAL DIFFERENCES BETWEEN CLASSES
    adj_mx = topological_adjacency(grid_geo, geocodes)
    tpl_dist = np.zeros((ngeocodes, ngeocodes))
    for i in range(ngeocodes):
        for j in range(i):
            ix = np.where((np.arange(ngeocodes) != i) & (np.arange(ngeocodes) != j))
            tpl_dist[i, j] = tpl_dist[j, i] = np.sqrt(np.sum((adj_mx[i, ix] - adj_mx[j, ix]) ** 2)) / np.sqrt(
                ngeocodes - 2)
    if verb:
        plot_jsdivmx_mds_hist(geocodes, tpl_dist, 'Topo only')
        # COMBINING TOPOLOGICAL AND JOINT DISTRIBUTION DIFFERENCES BETWEEN CLASSES
        wt = 0.25
        jsd_tpl = (1 - wt) * jsdist_joint_dist / jsdist_joint_dist.max() + wt * tpl_dist
        plot_jsdivmx_mds_hist(geocodes, jsd_tpl, 'Topology &')  # ,class_hist_count_joint_mag_grv
        # dendrogram plot
        y = jsd_tpl[np.triu_indices(ngeocodes, 1)]
        lnk = linkage(y, 'single')
        fig = plt.figure(figsize=(15, 5), dpi=300)
        dn = dendrogram(lnk, labels=geocodes)
        plt.show()

    # %% RUN DEESSE
    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)') + " - RUN DEESSE")
    deesse_output = run_deesse(tiMissingGeol, mps_nreal, nneighboringNode,
                               distanceThreshold, maxScanFraction, myseed + ti_realid,
                               nthreads, geolcd, cond_data)

    # %% COMPUTE REAL INDICATORS
    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)') + " - COMPUTE REAL INDICATORS")

    # Retrieve the results
    sim = deesse_output['sim']

    # Do some statistics on the realizations
    # ... gather all the realizations into one image
    all_sim = gn.img.gatherImages(sim)  # all_sim is one image with nreal variables
    # ... compute the pixel-wise proportion for the given categories
    all_sim_stats = gn.img.imageCategProp(all_sim, geocodes)
    realizations = np.ones((ny, nx, mps_nreal)) * np.nan
    for i in range(mps_nreal):
        ix = i * tiMissingGeol.nv
        realizations[:, :, i] = all_sim.val[ix, 0, :, :]

    geocode_entropy = entropy(realizations)

    # Local performance - errors
    error = np.zeros((ny, nx, ngeocodes + 1))
    for i in range(ngeocodes):
        tmp1 = realizations == geocodes[i]
        tmp2 = grid_geo == geocodes[i]  # np.tile(np.reshape(grid_geo==codelist[i],(ny,nx,1)), (1,1,nreal))
        error[:, :, i] = np.mean(tmp1, axis=2) - tmp2
    # error across all lithocode
    for r in range(mps_nreal):
        error[:, :, -1] += 1 * ((grid_geo - realizations[:, :, r]) != 0) / mps_nreal

    # confusion matrix, TP/FP/TN/FN and related indicators
    reference = grid_geo
    mask = 1 - grid_msk
    confusion_matrix, classes, training_size, testing_size = get_confusion_matrix(realizations, reference, mask)
    tfpn = get_true_false_pos_neg(confusion_matrix, classes, training_size, testing_size)
    classification_performance(tfpn)
    confusion_matrix_th, classes, thresholds = get_confusion_matrix_th(realizations, reference, mask, nthresholds_tpfn)
    TPR, FPR = get_tpr_fpr(confusion_matrix_th)

    if verb:
        # PLOT REAL
        plot_real_and_ref(realizations, grid_geo, grid_msk, addtitle=addtitle)
        # PLOT ENTROPY AND CONFUSION MATRIX
        plot_entropy_and_confusionmx(geocode_entropy, confusion_matrix, mps_nreal)

    # EXPORT / SAVE
    datafilepath = path2real + 'data' + suffix + '-ndisks-' + str(ti_ndisks) + '-areapct-' + str(
        ti_pct_area) + '-r-' + str(ti_realid) + '.pickle'
    with open(datafilepath, 'wb') as f:
        pickle.dump([realizations, grid_msk
                     ], f)

    datafilepath = path2ind + 'data' + suffix + '-ndisks-' + str(ti_ndisks) + '-areapct-' + str(
        ti_pct_area) + '-r-' + str(ti_realid) + '.pickle'
    with open(datafilepath, 'wb') as f:
        pickle.dump([class_hist_count_marg_mag, class_hist_count_marg_grv, class_hist_count_marg_lmp,
                     class_hist_count_pct_marg_mag, class_hist_count_pct_marg_grv, class_hist_count_pct_marg_lmp,
                     shannon_entropy_joint_dist, shannon_entropy_marg, shannon_entropy_joint_mag_grv,
                     shannon_entropy_pct_joint_dist, shannon_entropy_pct_marg, shannon_entropy_pct_joint_mag_grv,
                     jsdist_joint_dist, jsdist_marg_mag, jsdist_marg_grv, jsdist_marg_lmp, jsdist_joint_mag_grv,
                     tpl_dist,
                     geocode_entropy, error, confusion_matrix, mps_nreal, tfpn, TPR, FPR
                     ], f)

    # FINISHED
    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)') + " - FINISHED")
    return

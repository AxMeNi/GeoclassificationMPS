# -*- coding:utf-8 -*-
__projet__ = "GeoclassificationMPS"
__nom_fichier__ = "display_functions"
__author__ = "MENGELLE Axel"
__date__ = "aout 2024"

from sklearn import manifold
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt


def plot_entropy():


def plot_histogram_disimilarity(dist_hist, seed):
    """
    Plots a 2D Multi-Dimensional Scaling (MDS) representation of histogram dissimilarities.

    This function takes a dissimilarity matrix (e.g., derived from Jensen-Shannon divergence between histograms) 
    and performs Multi-Dimensional Scaling (MDS) to reduce the dimensionality to 2D for visualization. 
    The resulting 2D coordinates are plotted, with the points color-coded based on sample IDs.

    Parameters:
    -----------
    dist_hist : ndarray
        A precomputed dissimilarity matrix (e.g., Jensen-Shannon divergence) of shape (nsim, nsim).
        This matrix represents pairwise dissimilarities between histograms.
    seed : int
        A seed for the random state in the MDS algorithm to ensure reproducibility.

    Returns:
    --------
    None. Displays a scatter plot representing the 2D MDS positions of the samples.

    Notes:
    ------
    - MDS (Multi-Dimensional Scaling) is used to reduce the dimensionality of the dissimilarity matrix to 2D for 
      easier visualization.
    - Colors are assigned to points based on their sample ID, using a custom colormap that blends blue, green, and red.
    - The function displays the plot but does not return any value.
    """
    #Perform MDS (Multi-Dimensional Scaling) to reduce dimensionality to 2D
    mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed, dissimilarity="precomputed", n_jobs=1)

    #Apply MDS to the Jensen-Shannon divergence matrices
    mdspos_lc = mds.fit_transform(dist_hist)  # MDS for lithocode histograms
    
    #Create a colormap for plotting
    colors1 = plt.cm.Blues(np.linspace(0., 1, 512))
    colors2 = np.flipud(plt.cm.Greens(np.linspace(0, 1, 512)))
    colors3 = plt.cm.Reds(np.linspace(0, 1, 512))
    colors = np.vstack((colors1, colors2, colors3))
    mycmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
    
    s_id = np.arange(nsim)  #Sample IDs for color coding in scatter plots
    
    #Calculate limits for plotting
    lcMDSxmin = np.min(mdspos_lc[:, 0])
    lcMDSxmax = np.max(mdspos_lc[:, 0])
    lcMDSymin = np.min(mdspos_lc[:, 1])
    lcMDSymax = np.max(mdspos_lc[:, 1])
    
    s = 100  # Marker size
    fig = plt.figure()
    plt.subplot(231)
    plt.title('2D MDS Representation of hist. dissimilarities')
    plt.scatter(mdspos_lc[:, 0], mdspos_lc[:, 1], c=s_id, cmap=mycmap, s=s, label='lithocode hist', marker='+')
    plt.xlim(lcMDSxmin, lcMDSxmax)
    plt.ylim(lcMDSymin, lcMDSymax)
    plt.legend(scatterpoints=1, loc='best', shadow=False)
    cbar = plt.colorbar()
    cbar.set_label('sample #')

    fig.subplots_adjust(left=0.0, bottom=0.0, right=2.0, top=1.6, wspace=0.3, hspace=0.25)
    plt.show()
    

def plot_topological_adjacency

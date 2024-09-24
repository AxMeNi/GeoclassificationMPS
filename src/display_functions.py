# -*- coding:utf-8 -*-
__projet__ = "GeoclassificationMPS"
__nom_fichier__ = "display_functions"
__author__ = "MENGELLE Axel"
__date__ = "aout 2024"

from sklearn import manifold
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt


def plot_entropy(entropy):
    """
    Plot the 2D entropy visualization from a given entropy array.

    Parameters:
    -----------
    entropy : np.ndarray
        Input entropy array, which can be 2D or higher-dimensional. If higher-dimensional,
        it will be squeezed to 2D using `np.squeeze()`.

    Returns:
    --------
    None
        The function displays a plot of the entropy without returning any values.

    Notes:
    ------
    - The function uses the 'viridis' colormap for better contrast and readability.
    - The color bar on the side shows the range of entropy values across the 2D plot.
    - The input entropy array is expected to have been calculated in advance using an appropriate method.
    - Higher-dimensional input will be reduced to 2D using `np.squeeze()`, removing any singleton dimensions.
    """
    ent = np.squeeze(entropy)
    plt.figure(figsize=(10, 8))
    
    plt.title("Entropy 2D Visualization")
    plt.imshow(ent, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Entropy')
    
    plt.tight_layout()
    plt.show()


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
    

def plot_topological_adjacency():
        # MDS Visualization
    
    # Manual MDS implementation (simplified for 2D)
    nbsamples = nsim
    np.random.seed(852)
    mdspos_lc = np.random.rand(nsim, 2)  # Simulated MDS positions for lithocodes
    mdspos_sf = np.random.rand(nsim, 2)  # Simulated MDS positions for scalar fields

    # Prepare for plotting
    s_id = np.arange(nbsamples)
    colors1 = plt.cm.Blues(np.linspace(0., 1, 512))
    colors2 = np.flipud(plt.cm.Greens(np.linspace(0, 1, 512)))
    colors3 = plt.cm.Reds(np.linspace(0, 1, 512))
    colors = np.vstack((colors1, colors2, colors3))
    mycmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

    ix = np.tril_indices(nsim, k=-1)
    dist_hist_vals = dist_hist[ix]
    dist_topo_hamming_vals = dist_topo_hamming[ix]

    # Limits for the plots
    lcmin, lcmax = np.min(dist_hist_vals), np.max(dist_hist_vals)
    sfmin, sfmax = np.min(dist_topo_hamming_vals), np.max(dist_topo_hamming_vals)
    lcMDSxmin, lcMDSxmax = np.min(mdspos_lc[:, 0]), np.max(mdspos_lc[:, 0])
    lcMDSymin, lcMDSymax = np.min(mdspos_lc[:, 1]), np.max(mdspos_lc[:, 1])
    sfMDSxmin, sfMDSxmax = np.min(mdspos_sf[:, 0]), np.max(mdspos_sf[:, 0])
    sfMDSymin, sfMDSymax = np.min(mdspos_sf[:, 1]), np.max(mdspos_sf[:, 1])

    # Plot the results
    # s = 100
    # fig = plt.figure(figsize=(15, 10))
    
    # plt.subplot(231)
    # plt.title('2D MDS Representation of Jensen-Shannon Divergence')
    # plt.scatter(mdspos_lc[:, 0], mdspos_lc[:, 1], c=s_id, cmap=mycmap, s=s, label='Lithocode JS divergence', marker='+')
    # plt.xlim(lcMDSxmin, lcMDSxmax)
    # plt.ylim(lcMDSymin, lcMDSymax)
    # plt.legend(scatterpoints=1, loc='best', shadow=False)
    # cbar = plt.colorbar()
    # cbar.set_label('Sample #')
    
    # plt.subplot(234)
    # plt.title('2D MDS Representation of Topological Adjacency (Hamming)')
    # plt.scatter(mdspos_sf[:, 0], mdspos_sf[:, 1], c=s_id, cmap=mycmap, s=s, label='Scalar field Hamming', marker='x')
    # plt.xlim(sfMDSxmin, sfMDSxmax)
    # plt.ylim(sfMDSymin, sfMDSymax)
    # plt.legend(scatterpoints=1, loc='best', shadow=False)
    # cbar = plt.colorbar()
    # cbar.set_label('Sample #')
    
    # plt.subplot(232)
    # plt.hist(dist_hist_vals, bins=20, color='blue')
    # plt.title('Jensen-Shannon Distribution')

    # plt.subplot(233)
    # plt.scatter(dist_hist_vals, dist_topo_hamming_vals, color='green')
    # plt.xlim(lcmin, lcmax)
    # plt.ylim(sfmin, sfmax)
    # plt.title('JS Divergence vs Topological Adjacency')

    # plt.subplot(235)
    # plt.hexbin(dist_hist_vals, dist_topo_hamming_vals, gridsize=30, cmap='Greens')
    # plt.xlim(lcmin, lcmax)
    # plt.ylim(sfmin, sfmax)
    # plt.title('Hexbin of JS Divergence vs Adjacency')

    # plt.subplot(236)
    # plt.hist(dist_topo_hamming_vals, bins=20, color='red')
    # plt.title('Topological Adjacency (Hamming) Distribution')

    # fig.subplots_adjust(left=0.0, bottom=0.0, right=2.0, top=1.6, wspace=0.3, hspace=0.25)
    # plt.show()

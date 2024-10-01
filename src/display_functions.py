# -*- coding:utf-8 -*-
__projet__ = "GeoclassificationMPS"
__nom_fichier__ = "display_functions"
__author__ = "MENGELLE Axel"
__date__ = "aout 2024"

from sklearn import manifold

import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



def plot_entropy(entropy, background_image=None, categ_var_name=None):
    """
    Plot the 2D entropy visualization from a given entropy array.

    Parameters:
    -----------
    entropy : np.ndarray
        Input entropy array, which can be 2D or higher-dimensional. If higher-dimensional,
        it will be squeezed to 2D using `np.squeeze()`.
    background_image : np.ndarray (optional)
        2D array representing the background, which is treated as a categorical variable.
        Each unique value will have its own color, and the color legend will reflect those values.
    categ_var_name : string (optional)
        Name of the categorical variable superposed to the entropy

    Returns:
    --------
    None
        The function displays a plot of the entropy, with background image and contour if provided.
    """
    
    ent = np.squeeze(entropy)
    
    if background_image is not None:
        unique_values = np.unique(background_image)
        num_unique = len(unique_values)
        
        cmap = plt.get_cmap('tab20', num_unique)
        
        norm = mcolors.BoundaryNorm(boundaries=np.arange(num_unique+1)-0.5, ncolors=num_unique)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        ent_img = ax1.imshow(ent, cmap='gray', interpolation='nearest')
        if categ_var_name is not None :
            ax1.set_title(f"Entropy with {categ_var_name} contours")
        else :
            ax1.set_title("Entropy with categorical variable contours")
        
        contour_levels = np.arange(num_unique)
        ax1.contour(background_image, levels=contour_levels, colors='white', linewidths=1)
        
        cbar_entropy = plt.colorbar(ent_img, ax=ax1)
        cbar_entropy.set_label('Entropy')
        
        bg_img = ax2.imshow(background_image, cmap=cmap, norm=norm)
        if categ_var_name is not None :
            ax2.set_title(f'{categ_var_name}')
        else :
            ax2.set_title("Categorical variable")
        
        cbar_bg = plt.colorbar(bg_img, ax=ax2, ticks=np.arange(num_unique))
        cbar_bg.ax.set_yticklabels([str(val) for val in unique_values])
        if categ_var_name is not None :
            cbar_bg.set_label(f'{categ_var_name}')
        else :
            cbar_bg.set_label("Categories")
        
        if categ_var_name is not None :
            plt.suptitle(f'Superposition of entropy with {categ_var_name}')
        else :
            plt.suptitle('Superposition of entropy with categorical variable')
        plt.tight_layout()
        plt.show()
    
    else:
        plt.figure()
        plt.imshow(ent, cmap='gray', interpolation='nearest')
        plt.colorbar(label='Entropy')
        plt.title("Entropy 2D visualization")
        plt.tight_layout()
        plt.show()


def plot_histogram_disimilarity(dist_hist, seed, nsim, referenceIsPresent = False):
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

    mdspos_lc = mds.fit_transform(dist_hist)

    colors1 = plt.cm.Blues(np.linspace(0., 1, 512))
    colors2 = np.flipud(plt.cm.Greens(np.linspace(0, 1, 512)))
    colors3 = plt.cm.Reds(np.linspace(0, 1, 512))
    colors = np.vstack((colors1, colors2, colors3))
    mycmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
    
    s_id = np.arange(nsim)
    
    lcMDSxmin = np.min(mdspos_lc[:, 0])
    lcMDSxmax = np.max(mdspos_lc[:, 0])
    lcMDSymin = np.min(mdspos_lc[:, 1])
    lcMDSymax = np.max(mdspos_lc[:, 1])
    
    s = 100 
    fig, ax = plt.subplots()
    plt.title('2D MDS Representation of hist. dissimilarities')
   
    norm = plt.Normalize(vmin=0, vmax=nsim-1)
    if referenceIsPresent:  
        scatter = ax.scatter(mdspos_lc[:-1, 0], mdspos_lc[:-1, 1], c=s_id, cmap=mycmap, s=s, label='lithocode hist', marker='+')
        plt.scatter(mdspos_lc[-1, 0], mdspos_lc[-1, 1], c='red', s=50, label='reference hist', marker='o')
    else:
        scatter = ax.scatter(mdspos_lc[:, 0], mdspos_lc[:, 1], c=s_id, cmap=mycmap, s=s, label='lithocode hist', marker='+')
        
    plt.xlim(lcMDSxmin, lcMDSxmax)
    plt.ylim(lcMDSymin, lcMDSymax)
    plt.legend(scatterpoints=1, loc='best', shadow=False)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('sample #')

    plt.show()


def plot_pairwise_histograms(lithocode_all, nsim):
    """
    Plots a matrix of histograms showing pairwise comparisons of lithocode histograms.

    Parameters:
    -----------
    lithocode_all : ndarray
        4D array where each entry [:,:,:,i] corresponds to a simulation's lithocode data.
    nsim : int
        The number of simulations.

    Returns:
    --------
    None. Displays a matrix of pairwise histograms.
    """
    fig, axs = plt.subplots(nsim, nsim, figsize=(12, 12), constrained_layout=True)
    fig.suptitle('Matrix of Pairwise Histograms', fontsize=16)

    for i in range(nsim):
        for j in range(nsim):
            ax = axs[i, j]
            # Filter out NaN values
            lithocode_i = lithocode_all[:, :, :, i].flatten()
            lithocode_j = lithocode_all[:, :, :, j].flatten()
            
            lithocode_i = lithocode_i[~np.isnan(lithocode_i)]
            lithocode_j = lithocode_j[~np.isnan(lithocode_j)]
            
            if i == j:
                hist_lc, bins_lc = np.histogram(lithocode_i, bins='auto')
                ax.bar(bins_lc[:-1], hist_lc, width=np.diff(bins_lc), color='blue', label=f'Sim {i}')
            else:
                hist_lc_i, bins_lc_i = np.histogram(lithocode_i, bins='auto')
                hist_lc_j, bins_lc_j = np.histogram(lithocode_j, bins=bins_lc_i)
                
                bar_width = np.diff(bins_lc_i) / 3
                ax.bar(bins_lc_i[:-1], hist_lc_i, width=bar_width, color='blue', label=f'Sim {i}', align='edge')
                ax.bar(bins_lc_i[:-1] + bar_width, hist_lc_j, width=bar_width, color='green', label=f'Sim {j}', align='edge')
            
            ax.set_title(f'Sim {i} vs Sim {j}', fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=6)

    plt.show()
    

def plot_topological_adjacency(dist_hist, dist_topo_hamming, nsim):
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
    s = 100
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
    plt.title('2D MDS Representation of Topological Adjacency (Hamming)')
    plt.scatter(mdspos_sf[:, 0], mdspos_sf[:, 1], c=s_id, cmap=mycmap, s=s, label='Scalar field Hamming', marker='x')
    plt.xlim(sfMDSxmin, sfMDSxmax)
    plt.ylim(sfMDSymin, sfMDSymax)
    plt.legend(scatterpoints=1, loc='best', shadow=False)
    cbar = plt.colorbar()
    cbar.set_label('Sample #')
    
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
    plt.show()

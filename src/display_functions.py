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
import geone as gn



def plot_realization(deesse_output, varname='', index_real=0, show=False):
    """
    
    """
    plt.clf()
    plt.close()
    
    sim = deesse_output['sim']
    plt.subplots(1, 1, figsize=(17,10), sharex=True, sharey=True)
    gn.imgplot.drawImage2D(sim[index_real], iv=0, categ=True, title=f'Real #{index_real} {varname}')
    
    if show:
        plt.show()


def plot_proportions(sim, show=False):
    """
    WARNING this function is specific to the case of Mount 
    Isa data and cannot be used for other cases.
    """
    plt.clf()
    plt.close()
    
    all_sim = gn.img.gatherImages(sim)  
    categ_val = [1,2,3,4,5,6,7]
    
    prop_col = ['lightblue', 'blue', 'orange', 'green', 'red', 'purple', 'yellow']
    cmap = [gn.customcolors.custom_cmap(['white', c]) for c in prop_col]
    
    all_sim_stats = gn.img.imageCategProp(all_sim, categ_val)
    plt.subplots(1, 7, figsize=(17,5), sharey=True)
    
    for i in range(7):
        plt.subplot(1, 7, i+1)
        gn.imgplot.drawImage2D(all_sim_stats, iv=i, cmap=cmap[i],
                               title=f'Prop. of categ. {i}')
    if show:
        plt.show()


def plot_entropy(entropy, background_image=None, categ_var_name=None, show=False):
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
    plt.clf()
    plt.close()
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
            
        if show:
            plt.tight_layout()
            plt.show()
    
    else:
        plt.figure()
        plt.imshow(ent, cmap='gray', interpolation='nearest')
        plt.colorbar(label='Entropy')
        plt.title("Entropy 2D visualization")
        
        if show:
            plt.tight_layout()
            plt.show()


def plot_histogram_disimilarity(dist_hist, seed, nsim, referenceIsPresent=False, show=False):
    """
    Plots a 2D Multi-Dimensional Scaling (MDS) representation of histogram dissimilarities.

    This function takes a dissimilarity matrix (e.g., derived from Jensen-Shannon divergence between histograms) 
    and performs Multi-Dimensional Scaling (MDS) to reduce the dimensionality to 2D for visualization. 
    The resulting 2D coordinates are plotted, with the points color-coded based on simulation IDs.

    Parameters:
    -----------
    dist_hist : ndarray
        A precomputed dissimilarity matrix (e.g., Jensen-Shannon divergence) of shape (nsim, nsim).
        This matrix represents pairwise dissimilarities between histograms.
    seed : int
        A seed for the random state in the MDS algorithm to ensure reproducibility.
    nsim : int
        Number of simulations. Equivalent to number of points to represent minus the reference.
    referenceIsPresent : bool, optional
        Whether to display a reference point separately.


    Returns:
    --------
    None. Displays a scatter plot representing the 2D MDS positions of the simulations.

    Notes:
    ------
    - MDS (Multi-Dimensional Scaling) is used to reduce the dimensionality of the dissimilarity matrix to 2D for 
      easier visualization.
    - Colors are assigned to points based on their simulation ID, using a custom colormap that blends blue, green, and red.
    - The function displays the plot but does not return any value.
    """
    plt.clf()
    plt.close()
    #Perform MDS (Multi-Dimensional Scaling) to reduce dimensionality to 2D
    mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed, dissimilarity="precomputed", n_jobs=1)

    mdspos_lc = mds.fit_transform(dist_hist)

    mycmap = plt.get_cmap('tab20', nsim)
    
    s_id = np.arange(nsim)
    
    lcMDSxmin = np.min(mdspos_lc[:, 0])
    lcMDSxmax = np.max(mdspos_lc[:, 0])
    lcMDSymin = np.min(mdspos_lc[:, 1])
    lcMDSymax = np.max(mdspos_lc[:, 1])
    
    s = 100 
    fig, ax = plt.subplots()
    plt.title('2D MDS Representation of hist. dissimilarities')
   
    norm = mcolors.BoundaryNorm(boundaries=np.arange(nsim+1)-0.5, ncolors=nsim)
    if referenceIsPresent:  
        scatter = ax.scatter(mdspos_lc[:-1, 0], mdspos_lc[:-1, 1], c=s_id, cmap=mycmap, s=s, label='lithocode hist', marker='+')
        plt.scatter(mdspos_lc[-1, 0], mdspos_lc[-1, 1], c='red', s=50, label='reference hist', marker='o')
    else:
        scatter = ax.scatter(mdspos_lc[:, 0], mdspos_lc[:, 1], c=s_id, cmap=mycmap, s=s, label='lithocode hist', marker='+')
        
    plt.xlim(lcMDSxmin, lcMDSxmax)
    plt.ylim(lcMDSymin, lcMDSymax)
    plt.legend(scatterpoints=1, loc='best', shadow=False)
    
    cbar = plt.colorbar(scatter, ax=ax, ticks=np.arange(nsim))
    cbar.ax.set_yticklabels([str(val) for val in s_id])
    
    cbar.set_label('simulation #')
    
    if show:
        plt.tight_layout()
        plt.show()


def plot_simvar_histograms(simvar_all, nsim, show=False):
    """
    Display for each lithology the effectif of this lithology depending on the simulations

    Parameters:
    -----------
    simvar_all : ndarray
        4D array where each entry [:,:,:,i] corresponds to a simulation's simvar data.
    nsim : int
        The number of simulations.

    Returns:
    --------
    None. 
    """
    plt.clf()
    plt.close()
    n_subplots = len(np.unique(simvar_all[~np.isnan(simvar_all)]))  # Number of simvars
    cols = 5  # Adjust the number of columns 
    rows = n_subplots // cols

    if n_subplots % cols != 0:
        rows += 1

    positions = range(1, n_subplots + 1) 

    fig = plt.figure(figsize=(cols * 3, rows * 3))  # Each subplot will be of size 3x3

    unique_simvars = np.unique(simvar_all[~np.isnan(simvar_all)])
    
    # For each simvar
    for k, simvar in enumerate(unique_simvars):
        ax = fig.add_subplot(rows, cols, positions[k])
        
        simvar_counts = []
        
        # For each simulation
        for i in range(nsim):
            simvar_i = simvar_all[:, :, :, i].flatten()
            simvar_i = simvar_i[~np.isnan(simvar_i)]
            count = np.sum(simvar_i == simvar)
            simvar_counts.append(count)
            
        ax.bar(range(nsim), simvar_counts, color='blue', label=f'simvar {simvar}')

        ax.set_title(f'simvar {simvar}')
        ax.set_xlabel(f'Simulations')
        ax.tick_params(axis='both', which='major', labelsize=6)
    
    if show:
        plt.tight_layout()
        plt.show()


def plot_topological_adjacency(dist_hist, dist_topo_hamming, nsim, referenceIsPresent=False, show=False):
    """
    Plot a 2D MDS representation of topological adjacency based on Hamming distance.

    This function visualizes the relationships between simulations using a 2D Multi-Dimensional Scaling (MDS) representation, 
    with distances computed based on Hamming distance between categorical variables (simvars). Optionally, it can highlight 
    a reference simulation if present and save or display the plot.

    Parameters:
    -----------
    dist_hist : ndarray
        A lower triangular matrix (2D array) of distances between simulations based on histogram dissimilarity.
    dist_topo_hamming : ndarray
        A lower triangular matrix (2D array) of topological distances between simulations based on Hamming distance.
    nsim : int
        The number of simulations included in the distance matrices.
    referenceIsPresent : bool, optional (default: False)
        If True, highlights the reference simulation in red.
    save : bool, optional (default: False)
        If True, the plot will be saved to a file.
    show : bool, optional (default: False)
        If True, the plot will be displayed.
    fname : str, optional (default: '')
        The name of the file to save the plot to. If not provided and `save` is True, 
        the plot is saved as 'topological_adjacency.png'.

    Returns:
    --------
    None.
    """
    plt.clf()
    plt.close()
    
    np.random.seed(852)
    mdspos_lc = np.random.rand(nsim, 2)  # Simulated MDS positions for simvars
    s_id = np.arange(nsim)
       
    mycmap = plt.get_cmap('tab20', nsim)
    norm = mcolors.BoundaryNorm(boundaries=np.arange(nsim+1)-0.5, ncolors=nsim)
    
    ix = np.tril_indices(nsim, k=-1)
    dist_hist_vals = dist_hist[ix]
    dist_topo_hamming_vals = dist_topo_hamming[ix]
    
    lcmin, lcmax = np.min(dist_hist_vals), np.max(dist_hist_vals)
    sfmin, sfmax = np.min(dist_topo_hamming_vals), np.max(dist_topo_hamming_vals)
    
    s = 100
    plt.title('2D MDS Representation of Topological Adjacency (Hamming)')
    
    if referenceIsPresent:
        scatter = plt.scatter(mdspos_lc[:-1, 0], mdspos_lc[:-1, 1], c=s_id[:-1], cmap=mycmap, s=s, label='Scalar field Hamming', marker='x')
        plt.scatter(mdspos_lc[-1, 0], mdspos_lc[-1, 1], c='red', s=50, label='reference Hamming', marker='o')
    else:
        scatter = plt.scatter(mdspos_lc[:, 0], mdspos_lc[:, 1], c=s_id, cmap=mycmap, s=s, label='Scalar field Hamming', marker='x')
        
    plt.xlim(np.min(mdspos_lc[:, 0]), np.max(mdspos_lc[:, 0]))
    plt.ylim(np.min(mdspos_lc[:, 1]), np.max(mdspos_lc[:, 1]))
    plt.legend(scatterpoints=1, loc='best', shadow=False)
    
    cbar = plt.colorbar(scatter, ticks=np.arange(nsim))
    cbar.ax.set_yticklabels([str(val) for val in s_id])  # Label the ticks with sample IDs
    cbar.set_label('Simulation #')
    
    if show:
        plt.tight_layout()
        plt.show()

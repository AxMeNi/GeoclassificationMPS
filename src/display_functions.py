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

from utils import *



def plot_realizations(deesse_output, varname='', index_real=0, n_real=1, show=True):
    """
    Plot multiple 2D realizations from Deesse simulation output.

    This function clears and closes any previous plots, then extracts
    multiple realizations from the simulation output and displays them
    as 2D images in subplots.

    Parameters:
    ----------
    deesse_output : dict
        A dictionary containing the results of a Deesse simulation, where 'sim'
        holds the simulation realizations as a list or array.
    index_real : int
        The index of the first realization to plot within the 'sim' list or array in `deesse_output`.
    n_real : int
        The number of consecutive realizations to plot starting from `index_real`.
    varname : str
        The name of the variable being plotted, used in the plot titles.
    show : bool, optional
        If `True`, the plot is displayed immediately. Default is `True`.

    Returns:
    -------
    None

    Notes:
    -----
    - This function uses `gn.imgplot.drawImage2D` to render images with categorical values.
    - Ensure `deesse_output` has at least `index_real + n_real` realizations in `deesse_output['sim']`.
    """
    plt.clf()
    plt.close()
    
    sim = deesse_output['sim']

    n_cols = min(n_real, 4)
    n_rows = (n_real + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.array(axes).reshape(-1)  # Flatten in case it's a grid

    for i in range(n_real):
        iv = index_real + i
        ax = axes[i]
        img = np.squeeze(sim[iv].val)  # Convert (1, ny, nx) to (ny, nx)

        im = ax.imshow(img, cmap='viridis', origin='lower')
        ax.set_title(f'Real #{iv} {varname}')
        ax.axis('off')

    # Hide unused axes
    for j in range(n_real, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    if show:
        plt.show()


def plot_mask(mask, title=None, show=False, masking_strategy="unknown"):
    """
    Plot a mask over a background image or as a standalone image.
    This function visualizes a mask, either overlaid on a background image or as a standalone image.
    If a background image is provided, the mask is displayed with a specified transparency (alpha).
    If no background image is provided, the mask is displayed in grayscale.
    
    Parameters:
    ----------
    mask : array-like
        A 2D array representing the mask to be visualized. Values should be in the range [0, 1].    
    title : str, optional
        Title for the plot. If `None`, a default title based on the masking strategy is used.   
    show : bool, optional
        If `True`, the plot is displayed immediately. Default is `False`.
    masking_strategy : str, optional
        A string describing the masking strategy used, which will be included in the plot title if `title` is not provided. Default is "unknown".
    
    Returns:    
    -------
    None

    Notes:  
    -----
    - The plot is cleared and closed before creating a new one to avoid overlapping plots.
    - if masking_strategy is ReducedTIsG, the function is not yet capable to display the mask so returns nothing
    """
    if masking_strategy == "ReducedTiSg":
        return
    
    plt.clf()
    plt.close()
    
    plt.figure()

    plt.imshow(mask, cmap='gray')
    plt.colorbar(label='Mask value')

    if title is not None:
        plt.title(title)
    else:
        plt.title(f"Masking strategy:{masking_strategy}")
    plt.tight_layout()

    if show:
        plt.show()
    plt.close()


def plot_proportions(sim, show=False):
    """
    Plot category proportions for each of the seven categories in the Mount Isa dataset.

    This function visualizes the proportion of each category across simulation realizations. 
    The function is specific to the Mount Isa dataset and will not work for other datasets 
    due to its assumptions about category values and color schemes.

    Parameters:
    ----------
    sim : list or array
        A collection of simulation realizations to process, each containing categorical data.
    show : bool, optional
        If `True`, displays the plot immediately. Default is `False`.

    Returns:
    -------
    None

    Notes:
    -----
    - This function uses `gn.img.gatherImages` to combine images from the simulation list and 
      `gn.img.imageCategProp` to calculate the proportion of each category.
    - The color scheme is specific to seven categories and should not be modified unless the 
      categorical values or proportions change.
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
    Plot a 2D entropy map with an optional categorical background for visual context.

    This function displays a 2D map of entropy values and optionally overlays contours of a categorical variable. 
    If a background image is provided, it adds categorical boundaries on the entropy map and displays the 
    categorical variable in a separate subplot.

    Parameters:
    ----------
    entropy : array-like
        2D array of entropy values to visualize.
    background_image : array-like, optional
        2D array of categorical variable values for overlaying on the entropy plot. Default is `None`.
    categ_var_name : str, optional
        The name of the categorical variable for labeling and titling the background image plot. Default is `None`.
    show : bool, optional
        If `True`, the plot is displayed immediately. Default is `False`.

    Returns:
    -------
    None

    Notes:
    -----
    - If `background_image` is provided, this function displays both the entropy map and the categorical variable 
      with a shared colormap.
    - The categorical variable is shown with unique color-coded values, and contours are drawn over the entropy map.
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
        plt.tight_layout()
        
        if show:
            plt.show()
    
    else:
        plt.figure()
        plt.imshow(ent, cmap='gray', interpolation='nearest')
        plt.colorbar(label='Entropy')
        plt.title("Entropy 2D visualization")
        plt.tight_layout()
        
        if show:
            plt.show()


def plot_histogram_dissimilarity(dist_hist, nsim, referenceIsPresent=False, show=False):
    """
    Plots a 2D Multi-Dimensional Scaling (MDS) representation of histogram dissimilarities.

    This function takes a dissimilarity matrix (e.g., derived from Jensen-Shannon divergence between histograms) 
    and performs Multi-Dimensional Scaling (MDS) to reduce the dimensionality to 2D for visualization. 
    The resulting 2D coordinates are plotted, with the points color-coded based on simulation IDs.

    Parameters:
    -----------
    dist_hist : ndarray
        A precomputed dissimilarity matrix (Jensen-Shannon divergence) of shape (nsim, nsim) or 
        (nsim+1, nsim+1) if a reference is present.
        This matrix represents pairwise dissimilarities between histograms.
    nsim : int
        Number of simulations. Equivalent to number of points to represent minus the reference.
    referenceIsPresent : bool, optional
        Whether to display a reference point separately.
    show : bool, optional
        If `True`, the plot is displayed immediately. Default is `False`.


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
    
    #Apply MDS to compute 2D positions based on histogram dissimilarities
    mds = manifold.MDS(n_components=2,
                        max_iter=3000, 
                        eps=1e-9,
                        dissimilarity='precomputed',
                        random_state=852,
                        n_jobs=1)
    mds_positions = mds.fit_transform(dist_hist)
    
    mycmap = plt.get_cmap('tab20', nsim)
    s_id = np.arange(nsim)
    norm = mcolors.BoundaryNorm(boundaries=np.arange(nsim+1)-0.5, ncolors=nsim)
    
    plt.title('2D MDS Representation of hist. dissimilarities')
   
    if referenceIsPresent:  
        scatter = plt.scatter(mds_positions[:-1, 0], mds_positions[:-1, 1], c=s_id, cmap=mycmap, 
            s=100, label='Simulations hist', marker='+')
        plt.scatter(mds_positions[-1, 0], mds_positions[-1, 1], c='red', 
            s=100, label='Reference hist', marker='o')
    else:
        scatter = plt.scatter(mds_positions[:, 0], mds_positions[:, 1], c=s_id, cmap=mycmap,
            s=100, label='Simulations hist', marker='+')
        
    plt.xlim(np.min(mds_positions[:, 0]), np.max(mds_positions[:, 0]))
    plt.ylim(np.min(mds_positions[:, 1]), np.max(mds_positions[:, 1]))
    plt.legend(scatterpoints=1, loc='best', shadow=False)
    
    cbar = plt.colorbar(scatter, ticks=np.arange(nsim))
    cbar.ax.set_yticklabels([str(val) for val in s_id])
    cbar.set_label('simulation #')
    
    plt.tight_layout()
    
    if show:
        plt.show()


def plot_topological_adjacency(dist_topo_hamming, nsim, referenceIsPresent=False, show=False):
    """
    Plots a 2D Multi-Dimensional Scaling (MDS) representation of topological adjacency.

    This function takes a precomputed dissimilarity matrix (e.g., based on Hamming distances between geobodies) 
    and performs Multi-Dimensional Scaling (MDS) to project the data into 2D for visualization. 
    The resulting 2D coordinates are plotted, with points color-coded based on their simulation IDs.

    Parameters:
    -----------
    dist_topo_hamming : ndarray
        A precomputed topological adjacency matrix (e.g., Hamming distance) of shape (nsim, nsim) or 
        (nsim+1, nsim+1) if a reference is present.
    nsim : int
        Number of simulations, excluding the reference if present.
    referenceIsPresent : bool, optional
        If `True`, the plot includes a reference point (last row/column in matrices). Default is `False`.
    show : bool, optional
        If `True`, the plot is displayed immediately. Default is `False`.

    Returns:
    --------
    None. Displays a scatter plot representing the 2D MDS positions of the simulations.

    Notes:
    ------
    - Multi-Dimensional Scaling (MDS) is applied to reduce the dimensionality of the input matrices 
      to 2D for visualization purposes.
    - Simulations are color-coded by their IDs using a categorical colormap.
    - If `referenceIsPresent` is `True`, the reference is highlighted with a distinct red marker.
    """
    plt.clf()
    plt.close()

    # Apply MDS to compute 2D positions based on Hamming distances
    mds = manifold.MDS(n_components=2,
                        max_iter=3000, 
                        eps=1e-9,
                        dissimilarity='precomputed',
                        random_state=852,
                        n_jobs=1)
    mds_positions = mds.fit_transform(dist_topo_hamming)
    
    s_id = np.arange(nsim)
    mycmap = plt.get_cmap('tab20', nsim)
    norm = mcolors.BoundaryNorm(boundaries=np.arange(nsim + 1) - 0.5, ncolors=nsim)

    plt.title('2D MDS Representation of Topological Adjacency (Hamming)')
    
    if referenceIsPresent:
        scatter = plt.scatter(mds_positions[:-1, 0], mds_positions[:-1, 1], c=s_id, cmap=mycmap, 
            s=100, label='Simulations Hamming', marker='x')
        plt.scatter(mds_positions[-1, 0], mds_positions[-1, 1], c='red', 
            s=100, label='Reference Hamming', marker='o')
    else:
        scatter = plt.scatter(mds_positions[:, 0], mds_positions[:, 1], c=s_id, cmap=mycmap, 
            s=100, label='Simulations Hamming', marker='x')

    plt.xlim(np.min(mds_positions[:, 0]) - 0.1, np.max(mds_positions[:, 0]) + 0.1)
    plt.ylim(np.min(mds_positions[:, 1]) - 0.1, np.max(mds_positions[:, 1]) + 0.1)
    plt.legend(scatterpoints=1, loc='best', shadow=False)

    cbar = plt.colorbar(scatter, ticks=np.arange(nsim))
    cbar.ax.set_yticklabels([str(val) for val in s_id])
    cbar.set_label('Simulation #')

    plt.tight_layout()

    if show:
        plt.show()


def plot_general_MDS(global_dissimilarity_matrix, labels, indicator_name='unknown_indicator', show = False):
    """
    Visualizes the global MDS representation of a dissimilarity matrix.

    This function computes a 2D MDS (Multidimensional Scaling) embedding of a given global 
    dissimilarity matrix and visualizes it using a scatter plot. Points are colored based on 
    their corresponding matrix indices.

    Parameters:
    -----------
    global_dissimilarity_matrix : ndarray of shape (n_samples, n_samples)
        The global dissimilarity matrix to be embedded and visualized.
    labels : list or ndarray of shape (n_samples,)
        Labels corresponding to the matrix indices for coloring the scatter plot.
    title : str
        Title of the plot, typically indicating the type or context of the dissimilarity matrix.
    seed : int, optional
        Random seed for the MDS algorithm to ensure reproducibility. Default is 852.
    show : bool, optional
        Whether to display the plot interactively. Default is False.

    Returns:
    --------
    None
        The function creates and optionally displays the scatter plot. 
        The plot will need to be saved externally if required.
    """
    plt.clf()
    plt.close()
    
    global_mds = manifold.MDS(n_components=2,
                                max_iter=3000,
                                eps=1e-9,
                                dissimilarity='precomputed',
                                random_state=852,
                                n_jobs=1)
    global_mds_positions = global_mds.fit_transform(global_dissimilarity_matrix)

    # Step 5: Visualization
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(global_mds_positions[:, 0], global_mds_positions[:, 1], 
                           c=labels, cmap='hsv', marker='o')
    plt.colorbar(scatter, label='Matrix Index')
    plt.title(f"Global MDS Representation of {indicator_name}")
    
    plt.tight_layout()
    
    if show :
        plt.show()


def plot_simvar_histograms(simvar_all, nsim, show=False):
    """
    Plot bar charts of the occurrence counts of each unique simulation variable across multiple simulations.

    This function displays individual bar charts for each unique value of `simvar` within a grid of subplots. 
    Each chart represents the frequency of the `simvar` across simulations, allowing for visual comparison 
    across multiple simulations.

    Parameters:
    ----------
    simvar_all : numpy.ndarray
        4D array where each slice (last dimension) represents a different simulation. The array should 
        contain simulation variables, with NaNs used for any missing values.
    nsim : int
        Total number of simulations to analyze.
    show : bool, optional
        If `True`, the plot is displayed immediately. Default is `False`.

    Returns:
    -------
    None

    Notes:
    -----
    - Each subplot represents one unique simulation variable (`simvar`) across all simulations.
    - `NaN` values are ignored in the computation of counts.
    - The number of columns is fixed to 5, with rows calculated dynamically to fit all unique `simvar` values.
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
        
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.4, wspace=0.3)
    
    plt.tight_layout()
    if show:
        plt.show()


def plot_standard_deviation(std_array, realizations_range, indicator_name, show=False):
    """
    Plots the standard deviation of an indicator as a function of the number of realizations.

    This function visualizes the variability of a specified indicator (e.g., Jensen-Shannon Divergence, 
    Entropy, Topological Adjacency) by plotting its standard deviation against a range of realizations.

    Parameters:
    -----------
    std_array : ndarray
        An array of standard deviation values corresponding to the indicator, computed for different numbers 
        of realizations.
    realizations_range : ndarray or list
        A range of realization counts for which the standard deviation is computed.
    indicator_name : str
        The name of the indicator whose standard deviation is being plotted (e.g., "Entropy").
    show : bool, optional
        If `True`, the plot is displayed immediately. Default is `False`.

    Returns:
    --------
    None. Displays the plot of standard deviation against realizations.
    """
    
    plt.clf()
    plt.close()
    #Jensen-Shannon Divergence, Entropy, Topological Adjacency
    plt.figure(figsize=(10, 5))
    plt.plot(realizations_range, std_array, marker='o', color='blue')
    plt.title(f'Standard Deviation of {indicator_name}')
    plt.xlabel('Number of Realizations')
    plt.ylabel('Standard Deviation')
    plt.tight_layout()
    
    if show:
        plt.show()







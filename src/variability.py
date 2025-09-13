# -*- coding:utf-8 -*-
__projet__ = "GeoclassificationMPS"
__nom_fichier__ = "variability"
__author__ = "MENGELLE Axel"
__date__ = "sept 2024"


from loopui import *
from utils import *
from itertools import combinations
from sklearn import manifold

import os

import geone as gn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


###################################################################
### SOME PRE-REQUIREMENTS FOR THE COMPUTATION OF THE INDICATORS ###
###################################################################

def custom_jsdist_hist(img1, img2, nbins, base, plot=False, title="", lab1="img1", lab2="img2", iz_section=0):
    # nbins >1 : for continuous variables
    # otherwise for discrete variables
    tmp_min = np.min([np.nanmin(img1.flatten()), np.nanmin(img2.flatten())])
    tmp_max = np.max([np.nanmax(img1.flatten()), np.nanmax(img2.flatten())])

    if nbins > 1:
        binedges = np.linspace(tmp_min, tmp_max, num=int(nbins + 1))
    else:
        tmp_unique = np.unique(np.vstack((img1.flatten(), img2.flatten())))
        binedges = np.zeros(len(tmp_unique) + 1)
        binedges[0:-1] = tmp_unique - 1/2
        binedges[-1] = tmp_unique[-1] + 1/2

    # Compute histograms and normalize
    hist1, _ = np.histogram(img1, bins=binedges)
    p1 = hist1 / np.prod(img1.shape)

    hist2, _ = np.histogram(img2, bins=binedges)
    p2 = hist2 / np.prod(img2.shape)

    # Plotting logic
    if plot:
        ix = np.where((p1 > 0) | (p2 > 0))
        if nbins > 1:
            X = np.round((binedges[1:] + binedges[:-1]) / 2, 2)
        else:
            X = tmp_unique

        X_axis = np.arange(len(X[ix]))

        if len(img1.shape) == 3:
            map1 = img1[iz_section, :, :]
        else:
            map1 = img1

        if len(img2.shape) == 3:
            map2 = img2[iz_section, :, :]
        else:
            map2 = img2

        fig = plt.figure()
        gs = fig.add_gridspec(1, 9)
        ax0 = fig.add_subplot(gs[0, 0:2])
        ax1 = fig.add_subplot(gs[0, 2:4])
        ax2 = fig.add_subplot(gs[0, 4])
        ax3 = fig.add_subplot(gs[0, 5:])
        ax0.axis('off')
        ax1.axis('off')
        ax2.axis('off')

        axins02 = inset_axes(ax2, width="10%", height="90%", loc='center left')

        ax0.set_title('Map ' + lab1 + " - iz=" + str(iz_section))
        ax1.set_title('Map ' + lab2 + " - iz=" + str(iz_section))
        ax2.set_title(title)

        vmin = np.min([np.min(img1), np.min(img2)])
        vmax = np.max([np.max(img1), np.max(img2)])
        pos00 = ax0.imshow(map1, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        ax1.imshow(map2, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        fig.colorbar(pos00, cax=axins02)

        ax3.bar(X_axis - 0.2, p1[ix], 0.4, label=lab1)
        ax3.bar(X_axis + 0.2, p2[ix], 0.4, label=lab2)
        ax3.set_xticks(X_axis)
        ax3.set_xticklabels(X[ix])
        ax3.set_xlabel("Property Values")
        ax3.set_ylabel("Proportion")
        ax3.set_title,;
        

        fig.subplots_adjust(left=0.0, bottom=0.0, right=2.0, top=.65, wspace=0.1, hspace=0.2)
        plt.show()

    return kldiv(p1, p2, base, 'js')


def custom_topological_adjacency2D(img2D, categval, verb):
    [ny, nx] = img2D.shape
    topo_adjacency = np.zeros((len(categval), len(categval)))
    
    # Find boundaries of categorical geobodies in img2D
    img2DdxP = np.append(np.reshape(img2D[:, 0], [ny, 1]), img2D[:, :-1], axis=1) - img2D
    img2DdxN = img2D - np.append(img2D[:, 1:], np.reshape(img2D[:, -1], [ny, 1]), axis=1)
    img2DdyP = np.append(np.reshape(img2D[0, :], [1, nx]), img2D[:-1, :], axis=0) - img2D
    img2DdyN = img2D - np.append(img2D[1:, :], np.reshape(img2D[-1, :], [1, nx]), axis=0)
    img2Dbdy = ( (img2DdxN != 0) | (img2DdxP != 0) | (img2DdyN != 0) | (img2DdyP != 0) )
    
    for v in range(len(categval)):
        tmpiy, tmpix = np.where((1 * img2Dbdy) * (img2D == categval[v]))
        tmpiyNeighbors = np.concatenate((tmpiy, tmpiy, np.maximum(0, tmpiy - 1), np.minimum(tmpiy + 1, ny - 1)))  # xprev, xnext, yprev, ynext
        tmpixNeighbors = np.concatenate((np.maximum(0, tmpix - 1), np.minimum(tmpix + 1, nx - 1), tmpix, tmpix))  # xprev, xnext, yprev, ynext
        tmpNgbVal = np.unique(img2D[tmpiyNeighbors, tmpixNeighbors])
        # CHANGE
        tmpNgbVal =  tmpNgbVal[~np.isnan(tmpNgbVal)]
        
        tmpidNgbVal = np.ones(len(tmpNgbVal)) * np.nan

        for n in range(len(tmpNgbVal)):
            # CHANGE
            # Ensure the value is scalar, not a sequence
            custom_tmpidNgbValn = np.asarray(np.where(categval==tmpNgbVal[n]))  # Corrected to return a scalar value
            tmpidNgbVal[n] = custom_tmpidNgbValn.flatten()
            
        tmpidNgbVal = tmpidNgbVal.astype(int)
        topo_adjacency[v, tmpidNgbVal] = 1
        topo_adjacency[tmpidNgbVal, v] = 1
        topo_adjacency[v, v] = 0
    
    if verb:
        print('Unique values in boundaries: ' + str(np.unique(img2D[img2Dbdy])))
        print('Adjacency matrix: \n' + str(topo_adjacency))
    
    return topo_adjacency
 

def custom_topo_dist(img1, img2, npctiles=0, verb=0, plot=0, leg=" "):
    if npctiles > 0:
        if verb:
            categ1, categ2 = discretize_img_pair(img1, img2, npctiles)
    else:
        categ1 = img1
        categ2 = img2

    categval = np.unique(np.append(categ1, categ2))

    topo1adjacency = custom_topological_adjacency2D(categ1, categval, verb)
    topo2adjacency = custom_topological_adjacency2D(categ2, categval, verb)

    shd = structural_hamming_distance(topo1adjacency, topo2adjacency)
    lsgd = laplacian_spectral_graph_distance(topo1adjacency, topo2adjacency)

    if verb:
        print(leg + ' structural Hamming distance: ' + str(shd))
        print(leg + ' Laplacian spectral graph distance: ' + str(lsgd))

    if plot:
        if npctiles > 0:
            plot_topology_adjacency(categ1, categ2, topo1adjacency, topo2adjacency, leg, shd, lsgd, img1, img2)
        else:
            plot_topology_adjacency(categ1, categ2, topo1adjacency, topo2adjacency, leg, shd, lsgd)

    return shd, lsgd

    
#####################################
### COMPUTATION OF THE INDICATORS ###
#####################################

        # JENSEN SHANNON DIVERGENCE AND TOPOLOGICAL ADJACENCY ARE ONLY 
        # CALCULATED ON PAIRS OF REALIZATIONS. THUS, IT IS REQUIRED
        # TO ITERATE OVER ALL POSSIBLE PAIRS OF REALIZATIONS.
        # RK: FOR TOPOLOGICAL ADJACENCY, 
        #  - BECAUSE ALL THE DATA ARE IN 2D
        #  - BECAUSE GEONE STORE A 2D ARRAY AS BEING A 3D 
        #    ARRAY WITH THE Z DIMENSION BEING EQUAL TO ONE
        #  - BECAUSE LOOP UI IS MORE EFFICIENT ON 2D ARRAYS
        # IT HAS BEEN DECIDED TO REMOVE ALL DIMENSIONS EQUAL TO ONE 
        # USING NP.SQUEEZE

def calculate_indicators(deesse_output, n_sim_variables, reference_var=None,
                         referenceIsPresent=None, use_reference=True):
    """
    Computes entropy, Jensen-Shannon divergence, and topological adjacency indicators 
    for the simulations in one or multiple Deesse outputs. 

    Parameters
    ----------
    deesse_output : dict or list of dict
        Deesse output(s) containing the simulated variable(s). 
    n_sim_variables : int
        Number of simulated variables in each Deesse output. Must be 1.
    reference_var : numpy.ndarray, optional
        Reference variable for comparison. Default is None.
    referenceIsPresent : list of bool or bool, optional
        Indicates if reference_var is present in each deesse_output. 
        Default: [True] * len(deesse_output) if reference_var is provided.
    use_reference : bool, default True
        Whether to include reference_var in distance matrices.

    Returns
    -------
    ent, dist_hist, dist_topo_hamming, dist_topo_lapl_spec
    """

    if n_sim_variables > 1: 
        raise ValueError(f"The simulation was computed for {n_sim_variables} variables, "
                         "cannot compute indicators with more than 1 variable.")
    
    # --- 1. Handle input type: single output or list of outputs ---
    if isinstance(deesse_output, dict):  
        outputs = [deesse_output]
    elif isinstance(deesse_output, list):  
        outputs = deesse_output
    else:
        raise TypeError("deesse_output must be either a dict or a list of dicts")
    
    n_outputs = len(outputs)

    # --- 2. Handle referenceIsPresent ---
    if reference_var is not None and referenceIsPresent is None:
        referenceIsPresent = [True] * n_outputs
    elif isinstance(referenceIsPresent, bool):
        referenceIsPresent = [referenceIsPresent] * n_outputs
    elif len(referenceIsPresent) != n_outputs:
        raise ValueError("Length of referenceIsPresent must match number of outputs")

    # --- 3. Extract all simulations and combine them ---
    all_sims_list = []
    for out in outputs:
        sim = out['sim']
        all_sim_img = gn.img.gatherImages(sim)
        all_sim = all_sim_img.val
        all_sim = np.transpose(all_sim, (1, 2, 3, 0))  # adjust dimensions
        all_sims_list.append(all_sim)
    
    all_sims = np.concatenate(all_sims_list, axis=3)  # concatenate along simulation dimension
    ntot = all_sims.shape[3]  # total number of realizations

    # --- 4. Entropy ---
    ent = entropy(all_sims)

    # --- 5. Initialize distance matrices ---
    if reference_var is None or not use_reference:
        dist_hist = np.zeros((ntot, ntot))
        dist_topo_hamming = np.zeros((ntot, ntot))
        dist_topo_lapl_spec = np.zeros((ntot, ntot))
        use_ref_in_calc = False
    else:
        dist_hist = np.zeros((ntot + 1, ntot + 1))
        dist_topo_hamming = np.zeros((ntot + 1, ntot + 1))
        dist_topo_lapl_spec = np.zeros((ntot + 1, ntot + 1))
        use_ref_in_calc = True

    # --- 6. Compute distances ---
    for idx1 in range(ntot):
        sim1 = np.squeeze(all_sims[:, :, :, idx1])

        # Compare with reference if applicable
        if use_ref_in_calc:
            dist_hist[idx1, -1] = custom_jsdist_hist(sim1, reference_var, -1, base=np.e)
            dist_hist[-1, idx1] = dist_hist[idx1, -1]

            dist_topo_hamming[idx1, -1], dist_topo_lapl_spec[idx1, -1] = custom_topo_dist(sim1, reference_var, npctiles=-1)
            dist_topo_hamming[-1, idx1] = dist_topo_hamming[idx1, -1]
            dist_topo_lapl_spec[-1, idx1] = dist_topo_lapl_spec[idx1, -1]

        # Compare with other realizations
        for idx2 in range(idx1):
            sim2 = np.squeeze(all_sims[:, :, :, idx2])

            dist_hist[idx1, idx2] = custom_jsdist_hist(sim1, sim2, -1, base=np.e)
            dist_hist[idx2, idx1] = dist_hist[idx1, idx2]

            d_hamming, d_lapl = custom_topo_dist(sim1, sim2, npctiles=-1)
            dist_topo_hamming[idx1, idx2] = d_hamming
            dist_topo_hamming[idx2, idx1] = d_hamming
            dist_topo_lapl_spec[idx1, idx2] = d_lapl
            dist_topo_lapl_spec[idx2, idx1] = d_lapl

    return ent, dist_hist, dist_topo_hamming, dist_topo_lapl_spec



    
def calculate_std_deviation(indicator_map, min_realizations=1, max_realizations=1):
    indicator_map = np.squeeze(indicator_map)
    std_array = []
    
    for n in range(min_realizations, max_realizations + 1):
        truncated_array = indicator_map[:n, :n]
        std_array.append(np.std(indicator_map[np.triu_indices(n, 1)]))
    
    realizations_range = range(min_realizations, max_realizations + 1)
    
    return std_array, realizations_range


def analyze_global_MDS(dissimilarity_matrices, 
                        sim_names, 
                        simulation_log_path,
                        deesse_output_directory,
                        column_to_seek = "seed", 
                        n_points=4):
    """
    Analyzes and compares multiple dissimilarity matrices via a global MDS representation.

    This function extracts a set of `n_points` farthest from the centroid in each dissimilarity 
    matrix's MDS representation, combines them into a global dissimilarity matrix, and applies 
    MDS to compare these points across all matrices.

    Parameters:
    -----------
    dissimilarity_matrices : list of ndarray
        List of dissimilarity matrices (each of shape (n_samples, n_samples)) to process.
    sim_names : list of str
        List of the names give to the simulation. e.g. : ['1', '2', '3', '24']
        /!\ Must be the same size as dissimilarity_matrices
    simulation_log_path : str
        Path to the file logging the details of the simulations to compare
    column_to_seek : str, optional
        Header of the column of the logging file where to seek the simulation index. Default is "seed"
    n_points : int, optional
        Number of farthest points to extract from each MDS representation. Default is 4.

    Returns:
    --------
    global_mds_positions : ndarray of shape (n_points * len(dissimilarity_matrices), 2)
        The 2D coordinates of all selected points in the global MDS space.

    Notes:
    ------
    - MDS (Multi-Dimensional Scaling) is applied individually to each dissimilarity matrix and 
      globally to the selected points.
    - The function uses Euclidean distances to construct the global dissimilarity matrix.
    """
    #Step 1: Apply MDS to each dissimilarity matrix and extract farthest points    
    mds = manifold.MDS(n_components=2,
                        max_iter=3000, 
                        eps=1e-9,
                        dissimilarity='precomputed',
                        random_state=852,
                        n_jobs=1)
    
    labels = [] #labels is used in the plotting part to assign their colors to each group of simulation
    list_n_real = []

    for i, (sim_name, matrix) in enumerate(zip(sim_names, dissimilarity_matrices)):
        labels.extend([i] * n_points)
        
        mds_positions = mds.fit_transform(matrix)
        farthest_indices = find_farthest_points_from_centroid(mds_positions, n_points=n_points)   
    
        #Step 2: Retrieving the corresponding realizations
        #READING THE FILE
        df = pd.read_csv(simulation_log_path)
        file_name = df.loc[df[column_to_seek] == sim_name, 'File Name'].values[0]
        deesse_output = load_pickle_file(os.path.join(deesse_output_directory, file_name))
        
        #LOOKING FOR THE REALIZATIONS
        sim = deesse_output['sim']
        all_sim_img = gn.img.gatherImages(sim) #Using the inplace functin of geone to gather images
        all_sim = all_sim_img.val
        all_sim = np.transpose(all_sim,(1,2,3,0)) #Transposing the dimensions of the array to make it work with loop-ui...
        
        for idx_real in farthest_indices:
            list_n_real.append(np.squeeze(all_sim[:,:,:,idx_real]))
    
    #Step 3: Build global dissimilarity matrix
    nreal = len(list_n_real)
    dist_hist = np.zeros((nreal, nreal)) # Jensen Shannon indicators
    dist_topo_hamming = np.zeros((nreal, nreal)) # topological adjacency indicators
    dist_topo_lapl_spec = np.zeros((nreal, nreal))

    for idx1_real in range(nreal) :
        for idx2_real in range(idx1_real):
            dist_hist[idx1_real, idx2_real] = custom_jsdist_hist(list_n_real[idx1_real],list_n_real[idx2_real],-1,base=np.e)
            dist_hist[idx2_real, idx1_real] = dist_hist[idx1_real,idx2_real]
            
            dist_topo_hamming[idx1_real, idx2_real], dist_topo_lapl_spec[idx1_real, idx2_real] = custom_topo_dist(list_n_real[idx1_real],list_n_real[idx2_real],npctiles=-1,)
            dist_topo_hamming[idx2_real, idx1_real] = dist_topo_hamming[idx1_real, idx2_real]

    return dist_hist, dist_topo_hamming, labels

# -*- coding:utf-8 -*-
__projet__ = "GeoclassificationMPS"
__nom_fichier__ = "variability"
__author__ = "MENGELLE Axel"
__date__ = "sept 2024"


from loopui import *
import geone as gn
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors


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
        ax3.set_title("histogram of " + title)
        ax3.legend()

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
            print('discretize')
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
    
    
def calculate_indicators(deesse_output):
    """

    """
    sim = deesse_output['sim']
    all_sim_img = gn.img.gatherImages(sim) #Using the inplace functin of geone to gather images
    all_sim = all_sim_img.val
    all_sim = np.transpose(all_sim,(1,2,3,0)) #Transposing the dimensions of the array to make it work with loop-ui...
    nsim = len(sim)
    
    #1 ENTROPY   
    ent = entropy(all_sim)
    
    plt.figure(figsize=(10, 8))
    
    # If 2D, just plot the entropy matrix
    plt.title("Entropy 2D Visualization")
    plt.imshow(ent, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Entropy')
    
    plt.tight_layout()
    plt.show()
    
    # JENSEN SHANNON DIVERGENCE AND TOPOLOGICAL ADJACENCY IS ONLY 
    # CALCULATED ON PAIRS OF REALIZATIONS. THUS, IT IS REQUIRED
    # TO ITERATE OVER ALL POSSIBLE PAIRS OF REALIZATIONS.
    # RK: FOR TOPOLOGICAL ADJACENCY, 
    #  - BECAUSE ALL THE DATA ARE IN 2D
    #  - BECAUSE GEONE STORE A 2D ARRAY AS BEING A 3D 
    #    ARRAY WITH THE Z DIMENSION BEING EQUAL TO ONE
    #  - BECAUSE LOOP UI IS MORE EFFICIENT ON 2D ARRAYS
    # IT HAS BEEN DECIDED TO REMOVE ALL DIMENSIONS EQUAL TO ONE 
    # USING NP.SQUEEZE
    
    dist_hist = np.zeros((nsim, nsim)) # To store Jensen Shannon indicators
    dist_topo_hamming = np.zeros((nsim, nsim)) # To store topological adjacency indicators
    dist_topo_lapl_spec = np.zeros((nsim, nsim))
    
    for idx1_real in range(nsim):
        for idx2_real in range(idx1_real):
            
            #2 JENSEN SHANNON DIVERGENCE
            dist_hist[idx1_real, idx2_real] = custom_jsdist_hist(np.squeeze(all_sim[:,:,:,idx1_real]),np.squeeze(all_sim[:,:,:,idx2_real]),-1,base=np.e)
            dist_hist[idx2_real, idx1_real] = dist_hist[idx1_real,idx2_real]
            
            #3 TOPOLOGICAL ADJACENCY
            # NOTE: the use of np.squeeze is to tranform fake 3D data into 2D data
            dist_topo_hamming[idx1_real, idx2_real], dist_topo_lapl_spec[idx1_real, idx2_real] = custom_topo_dist(np.squeeze(all_sim[:,:,:,idx1_real]),np.squeeze(all_sim[:,:,:,idx2_real]),npctiles=-1,)
            dist_topo_hamming[idx2_real, idx1_real] = dist_topo_hamming[idx2_real, idx1_real]
            dist_topo_lapl_spec[idx2_real, idx1_real] = dist_topo_lapl_spec[idx1_real, idx2_real]   
    
    return ent, dist_hist, dist_topo_hamming


    
    
    
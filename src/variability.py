# -*- coding:utf-8 -*-
__projet__ = "GeoclassificationMPS"
__nom_fichier__ = "variability"
__author__ = "MENGELLE Axel"
__date__ = "sept 2024"

from sklearn import manifold
from loopui import *
import geone as gn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def jsdist2_hist(img1, img2, nbins, base, plot=False, title="", lab1="img1", lab2="img2", iz_section=0):
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

    # Return Jensen-Shannon divergence
    return kldiv(p1, p2, base, 'js')

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
    
    #2 JENSEN SHANNON DIVERGENCE
    dist_hist = np.zeros((nsim, nsim))
    for idx1_real in range(nsim):
        for idx2_real in range(idx1_real):
            dist_hist[idx1_real, idx2_real] = jsdist2_hist(all_sim[:,:,:,idx1_real],all_sim[:,:,:,idx2_real],-1,base=np.e)
            dist_hist[idx2_real, idx1_real] = dist_hist[idx1_real,idx2_real]
    
    # 3. Perform MDS (Multi-Dimensional Scaling) to reduce dimensionality to 2D
    mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=852, dissimilarity="precomputed", n_jobs=1)

    # Apply MDS to the Jensen-Shannon divergence matrices
    mdspos_lc = mds.fit_transform(dist_hist)  # MDS for lithocode histograms
    
    # Create a colormap for plotting
    colors1 = plt.cm.Blues(np.linspace(0., 1, 512))
    colors2 = np.flipud(plt.cm.Greens(np.linspace(0, 1, 512)))
    colors3 = plt.cm.Reds(np.linspace(0, 1, 512))
    colors = np.vstack((colors1, colors2, colors3))
    mycmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
    
    s_id = np.arange(nsim)  # Sample IDs for color coding in scatter plots
    
    # Calculate limits for plotting
    lcMDSxmin = np.min(mdspos_lc[:, 0])
    lcMDSxmax = np.max(mdspos_lc[:, 0])
    lcMDSymin = np.min(mdspos_lc[:, 1])
    lcMDSymax = np.max(mdspos_lc[:, 1])
    
    # Plot results
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

    # Adjust layout and display the plot
    fig.subplots_adjust(left=0.0, bottom=0.0, right=2.0, top=1.6, wspace=0.3, hspace=0.25)
    plt.show()
    
    return dist_hist
    
    
    
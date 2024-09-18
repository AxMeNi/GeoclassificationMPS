# -*- coding:utf-8 -*-
__projet__ = "GeoclassificationMPS"
__nom_fichier__ = "variability"
__author__ = "MENGELLE Axel"
__date__ = "sept 2024"

from loopui import *
import geone as gn
import numpy as np


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
            dist_hist[idx1_real, idx2_real] = jsdist_hist(all_sim[:,:,:,idx1_real],all_sim[:,:,:,idx2_real],-1,base=np.e)
            dist_hist[idx2_real, idx1_real] = dist_hist[idx1_real,idx2_real]
    
    return dist_hist
    
    
    
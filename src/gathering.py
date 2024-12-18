# -*- coding:utf-8 -*-
__projet__ = "GeoclassificationMPS"
__nom_fichier__ = "gathering"
__author__ = "MENGELLE Axel"
__date__ = "decembre 2024"


import pandas as pd
from utils import *
from variability import *
from display_functions import *
import os



def gather_dissimilarity_matrices(output_directory, simulation_log_path):
    df = pd.read_csv(simulation_log_path)
    all_hist=[]
    seeds=[]
    for i, row in df.iterrows():
        file_name = row['File Name']
        seed = row['seed']
        seeds.append(seed)
        deesse_output = load_pickle_file(os.path.join(output_directory, file_name))
        ent, dist_hist, dist_topo_hamming = calculate_indicators(deesse_output, n_sim_variables=1)
        all_hist.append(dist_hist)

    return all_hist, seeds



if __name__ == "__main__":
    all_hist, seeds = gather_dissimilarity_matrices(r"C:\Users\00115212\Documents\Kaya simulations\26-11-2024\deesse_output","C:/Users/00115212/Documents/Kaya simulations/26-11-2024/deesse_output/simulation_log.csv")
    dist_hist, dist_topo_hamming, labels = analyze_global_MDS(all_hist, 
                                                            seeds, 
                                                            "C:/Users/00115212/Documents/Kaya simulations/26-11-2024/deesse_output/simulation_log.csv",
                                                            r"C:\Users\00115212\Documents\Kaya simulations\26-11-2024\deesse_output",
                                                            column_to_seek = "seed", 
                                                            n_points=4)

    plot_general_MDS(dist_hist, labels, indicator_name='unknown_indicator', show = True)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
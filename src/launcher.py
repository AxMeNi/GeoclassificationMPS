# -*- coding:utf-8 -*-
__projet__ = "GeoclassificationMPS"
__nom_fichier__ = "launcher"
__author__ = "MENGELLE Axel"
__date__ = "juillet 2024"


from ti_mask_generation import *
from data_treatment import get_unique_names_and_types
from sg_mask_generation import *
from build_ti_cd import *
from saving import *
from display_functions import *
from time_logging import *
from variability import *

import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime 
from loopui import entropy


###
timelog = pd.DataFrame(columns=['Process', 'Start_Time', 'End_Time', 'Duration'])


def launcher(params,
            nvar, sim_var, auxTI_var, auxSG_var, condIm_var, names_var, types_var, outputVarFlag,
            nr, nc,
            verbose):
    """

    """
    global timelog
    
    seed = params['seed'] 
    ti_methods = params['ti_methods']
    ti_pct_area = params['ti_pct_area']
    ti_nshapes = params['ti_nshapes']
    pct_ti_sg_overlap = params['pct_ti_sg_overlap']
    pct_sg = params['pct_sg']
    pct_ti = params['pct_ti'] 
    cc_sg = params['cc_sg']
    rr_sg = params['rr_sg']
    cc_ti = params['cc_ti']
    rr_ti = params['rr_ti']
    nRandomTICDsets = params['nRandomTICDsets']
    nn = params['n_neighbouring_nodes']
    dt = params['distance_threshold']
    ms = params['max_scan_fraction']
    numberofmpsrealizations = params['n_mps_realizations']
    nthreads = params['n_threads']
    saveMask = params['saveMask']
    saveOutput = params['saveOutput']
    saveIndicators = params['saveIndicators']
    output_directory = params['output_directory']
    deesse_output_folder = params['deesse_output_folder']
    prefix_deesse_output = params['prefix_deesse_output']
    plot_output_folder = params['plot_output_folder']
    prefix_histogram_dissimilarity = params['prefix_histogram_dissimilarity']
    prefix_entropy = params['prefix_entropy']
    prefix_simvar_histograms = params['prefix_simvar_histograms']
    prefix_topological_adjacency = params['prefix_topological_adjacency']
    reference_var = params['reference_var']
    prefix_proportions = params['prefix_proportions']
    prefix_std_deviation = params['prefix_std_deviation']
    
    deesse_output_folder_complete = os.path.join(output_directory, deesse_output_folder)
    plot_output_folder_complete = os.path.join(output_directory, plot_output_folder)
    
    ti_list = []
    cd_list = []
    
    #Create a simulation grid mask based on no values of the auxiliary variables
    t0_sgticd = start_timer("Creation of SG, TI and CD")
    if verbose:
        print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S:%f)') + " <> INITIATED CREATION OF THE SIMULATION GRID, OF THE CONDITIONING DATA, AND OF THE TI")
        
    simgrid_mask_aux = create_sg_mask(auxTI_var, auxSG_var, nr, nc)

    #THREE PARAMETERS USED BELOW
    i_mask = seed
    nsim=numberofmpsrealizations
    n_sim_variables=1
    aux_var_names = "_".join(auxTI_var.keys())

    #---- METHOD 1 : for 1 set of TI and CD ----#
    
    if nRandomTICDsets == 1 :
        
        #Creation of the TI, the SG, the CD
        if "DependentCircles" in ti_methods :
            if verbose:
                print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S:%f)') + " <> USING METHOD DEPENDENT CIRCLES")
            ti_frame_DC, ntc_DC = gen_ti_frame_circles(nr, nc, ti_pct_area, ti_nshapes, seed)
            ti_list_DC, cd_list_DC = build_ti_cd(ti_frame_DC, ntc_DC, sim_var, nc, nr, auxTI_var, auxSG_var, names_var, simgrid_mask_aux, condIm_var)
            ti_list.extend(ti_list_DC)
            cd_list.extend(cd_list_DC)
            simgrid_mask = simgrid_mask_aux
            cc_sg, rr_sg = nc, nr
            if saveMask:
                plot_mask(simgrid_mask,masking_strategy="Dependent Circles")
                save_plot(fname=f"mask_DependentCircles__SEED{seed}_TIPCT{ti_pct_area}-TINSHP{ti_nshapes}-{aux_var_names}.png", output_directory=plot_output_folder_complete, comments=f'mask_dependentcircles_SEED{seed}_TIPCT{ti_pct_area}-TINSHP{ti_nshapes}-{aux_var_names}', params={"nsim":nsim})
                save_mask(simgrid_mask, output_directory=deesse_output_folder_complete, file_name=f"mask_DependentCircles_SEED{seed}_TIPCT{ti_pct_area}-TINSHP{ti_nshapes}-{aux_var_names}.npy",params={"nsim":nsim})
            
            
        if "DependentSquares" in ti_methods :
            if verbose:
                print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S:%f)') + " <> USING METHOD DEPENDENT SQUARES")
            ti_frame_DS, ntc_DS = gen_ti_frame_squares(nr, nc, ti_pct_area, ti_nshapes, seed)
            ti_list_DS, cd_list_DS = build_ti_cd(ti_frame_DS, ntc_DS, sim_var, nc, nr, auxTI_var, auxSG_var, names_var, simgrid_mask_aux, condIm_var)
            ti_list.extend(ti_list_DS)
            cd_list.extend(cd_list_DS)
            simgrid_mask = simgrid_mask_aux
            cc_sg, rr_sg = nc, nr
            if saveMask:
                plot_mask(simgrid_mask,masking_strategy="Dependent Squares")
                save_plot(fname=f"mask_DependentSquares__SEED{seed}_TIPCT{ti_pct_area}-TINSHP{ti_nshapes}-{aux_var_names}.png", output_directory=plot_output_folder_complete, comments=f'mask_dependentsquares_SEED{seed}_TIPCT{ti_pct_area}-TINSHP{ti_nshapes}-{aux_var_names}', params={"nsim":nsim})
                save_mask(simgrid_mask, output_directory=deesse_output_folder_complete, file_name=f"mask_DependentSquares_SEED{seed}_TIPCT{ti_pct_area}-TINSHP{ti_nshapes}-{aux_var_names}.npy",params={"nsim":nsim})

            
        if "IndependentSquares" in ti_methods :
            if verbose:
                print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S:%f)') + " <> USING METHOD INDEPENDENT SQUARES")
            ti_frame_IS, ntc_IS = gen_ti_frame_separatedSquares(nr, nc, ti_pct_area, ti_nshapes, seed)
            ti_list_IS, cd_list_IS = build_ti_cd(ti_frame_IS, ntc_IS, sim_var, nc, nr, auxTI_var, auxSG_var, names_var, simgrid_mask_aux, condIm_var)
            ti_list.extend(ti_list_IS)
            cd_list.extend(cd_list_IS)
            simgrid_mask = simgrid_mask_aux
            cc_sg, rr_sg = nc, nr
            if saveMask:
                plot_mask(simgrid_mask,masking_strategy="Independent Squares")
                save_plot(fname=f"mask_IndependentSquares__SEED{seed}_TIPCT{ti_pct_area}-TINSHP{ti_nshapes}-{aux_var_names}.png", output_directory=plot_output_folder_complete, comments=f'mask_independentsquares_SEED{seed}_TIPCT{ti_pct_area}-TINSHP{ti_nshapes}-{aux_var_names}', params={"nsim":nsim})
                save_mask(simgrid_mask, output_directory=deesse_output_folder_complete, file_name=f"mask_IndependentSquares_SEED{seed}_TIPCT{ti_pct_area}-TINSHP{ti_nshapes}-{aux_var_names}.npy",params={"nsim":nsim})

        if "ReducedTiSg" in ti_methods :
            if verbose:
                print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S:%f)') + " <> USING METHOD DEPENDENT REDUCED TI AND SG")
            ti_frame_RTS, ntc_RTS, simgrid_mask_RTS, cc_sg, rr_sg = gen_ti_frame_sg_mask(nr, nc, pct_ti_sg_overlap, pct_sg, pct_ti, cc_sg, rr_sg, cc_ti, rr_ti, seed)
            simgrid_mask_merged = merge_masks(simgrid_mask_RTS, simgrid_mask_aux)
            ti_list_RTS, cd_list_RTS = build_ti_cd(ti_frame_RTS, ntc_RTS, sim_var, cc_sg, rr_sg, auxTI_var, auxSG_var, names_var, simgrid_mask_merged, condIm_var)
            ti_list.extend(ti_list_RTS)
            cd_list.extend(cd_list_RTS)
            simgrid_mask = None
            if saveMask:
                if verbose:
                    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S:%f)') + " <> UNABLE TO SAVE MASK FOR METHOD REDUCED TI SG AS FUNCTION IS NOT YET IMPLEMENTED")
            
        timelog = end_timer_and_log(t0_sgticd, timelog)
        

        plt.subplot(1,4,1)   
        plt.hist(ti_list[0].val[0,0,:,:].flatten(), bins=50)
        plt.subplot(1,4,2) 
        plt.imshow(ti_list[0].val[0,0,:,:])
        plt.subplot(1,4,3)
        plt.hist(sim_var['grid_geo'].flatten(), bins=50)
        plt.subplot(1,4,4) 
        plt.imshow(sim_var['grid_geo'])
        plt.plot()
        plt.show()
        print(np.unique(ti_list[0].val[1,0,:,:].flatten())[0], np.unique(cd_list[0].val[0,0,:,:].flatten())[0])
        print(np.setdiff1d(np.unique(ti_list[0].val[1,0,:,:].flatten()), np.unique(cd_list[0].val[0,0,:,:].flatten())))
        print(np.unique(ti_list[0].val[1,0,:,:].flatten()) == np.unique(cd_list[0].val[0,0,:,:].flatten()))
        print(np.unique(ti_list[0].val[2,0,:,:].flatten()), np.unique(cd_list[0].val[1,0,:,:].flatten()))
        print(np.unique(ti_list[0].val[2,0,:,:].flatten()) == np.unique(cd_list[0].val[1,0,:,:].flatten()))
        exit()

        if verbose:
            print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S:%f)') + f" <> DATA DIMENSION : \n·····>> NUMBER OF ROWS : {nr} \n·····>> NUMBER OF COLUMNS : {nc}")
            print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S:%f)') + " <> FINISHED THE CREATION OF SG, CD AND TI")
        
            
        # im = gn.img.Img(nc, nr, 1, 1, 1, 1, 0, 0, 0, nv=0)
        # xx = im.xx()[0]
        # yy = im.yy()[0]
        # nTI = 2
        # pB = (np.minimum(np.maximum(xx, 190), 290) - 190) / 100
        # pA = 1.0 - pB
        # pdf_ti = np.zeros((2, 1, nr, nc))
        # pdf_ti[0,0,:,:] = pA
        # pdf_ti[1,0,:,:] = pB
        # im.append_var(pdf_ti, varname=['pA', 'pB'])
        # plt.subplots(1,2, figsize=(17,5), sharey=True) # 1 x 2 sub-plots
        # plt.subplot(1,2,1)
        # gn.imgplot.drawImage2D(im, iv=0, title='Probability to select TI A')
        # plt.subplot(1,2,2)
        # gn.imgplot.drawImage2D(im, iv=1, title='Probability to select TI B')
        # plt.show()
        
        nTI = len(ti_list)
        names, distance_types = get_unique_names_and_types(names_var, types_var)
        
        outputFlag = []
        for name in names:
            outputFlag.append(outputVarFlag[name])

        deesse_input = gn.deesseinterface.DeesseInput(
            nx=cc_sg, ny=rr_sg, nz=1,
            sx=1, sy=1, sz=1,
            ox=0, oy=0, oz=0,
            nv=nvar, varname=names,
            TI=ti_list,
            #pdfTI = pdf_ti,
            mask=simgrid_mask,
            dataImage=cd_list,
            distanceType=distance_types,
            nneighboringNode=nvar*[nn],
            distanceThreshold=nvar*[dt],
            maxScanFraction=nTI*[ms],
            outputVarFlag=outputFlag,
            npostProcessingPathMax=1,
            seed=seed,
            nrealization=numberofmpsrealizations
        ) 
        
        
        t0_sim = start_timer(f"simulation {seed}")
        
        if verbose:
            print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S:%f)') + " <> CREATED DEESSE INPUT, STARTING SIMULATION")        
        
        deesse_output = gn.deesseinterface.deesseRun(deesse_input, nthreads = nthreads)

        timelog = end_timer_and_log(t0_sim, timelog)
        
        if verbose:
            print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S:%f)') + " <> FINISHED SIMULATION")     
            
        if saveOutput:
            save_simulation(deesse_output, params, output_directory=deesse_output_folder_complete)
    
        #PARAMETERS FOR RETRIEVING THE SIMULATION
        sim = deesse_output['sim']
        all_sim_img = gn.img.gatherImages(sim) #Using the inplace function of geone to gather images
        all_sim = all_sim_img.val
        all_sim = np.transpose(all_sim,(1,2,3,0))
        
        #CALCULATION OF THE INDICATORS
        t0_indctr = start_timer(f"indicators")
        if verbose:
            print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S:%f)') + " <> CALCULATING INDICATORS")
        ent, dist_hist, dist_topo_hamming = calculate_indicators(deesse_output, n_sim_variables=n_sim_variables, reference_var=reference_var)
        timelog = end_timer_and_log(t0_indctr, timelog)
        if verbose:
            print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S:%f)') + " <> FINISHED THE CALCULATION OF THE INDICATORS, CALCULATING THE STANDARD DEVIATION")
        
        #CALCULATION OF THE STANDARD DEVIATION
        t0_plot = start_timer(f"indicators plotting")
        std_ent, realizations_range1 = calculate_std_deviation(ent, 1, numberofmpsrealizations)
        std_dist_hist, realizations_range2 = calculate_std_deviation(dist_hist, 1, numberofmpsrealizations)
        std_dist_hamming, realizations_range3 = calculate_std_deviation(dist_topo_hamming, 1, numberofmpsrealizations)
        if verbose:
            print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S:%f)') + " <> FINISHED THE CALCULATION OF THE STANDARD DEVIATIONS")

        if saveIndicators:
            save_indicators(indicators_dict={"ent":ent,
                                            "dist_hist":dist_hist,
                                            "dist_topo_hamming": dist_topo_hamming,
                                            #"std_ent": std_ent,
                                            "std_dist_hist": std_dist_hist,
                                            "std_dist_hamming": std_dist_hamming,
                                            "n_mps_real": numberofmpsrealizations
                                            #f"std_ent_{realizations_range1}": std_ent,
                                            #f"std_dist_hist_{realizations_range2}": std_dist_hist,
                                            #f"std_dist_hamming_{realizations_range3}": std_dist_hamming
                                            },
                            output_directory=output_directory ,#f'output/TIPCT{ti_pct_area}-TINSHP{ti_nshapes}-{aux_var_names}', 
                            comments='',  
                            params={"nsim":nsim})
                
        #1 ENTROPY
        plot_entropy(ent, background_image=reference_var, categ_var_name="Lithofacies")
        save_plot(fname=prefix_entropy+f"_SEED{seed}_TIPCT{ti_pct_area}-TINSHP{ti_nshapes}-{aux_var_names}.png", output_directory=plot_output_folder_complete, comments=f'TIPCT{ti_pct_area}-TINSHP{ti_nshapes}-{aux_var_names}', params={"nsim":nsim})
        
        #2 HISTOGRAM DISSIMILARITY
        plot_histogram_dissimilarity(dist_hist, nsim, referenceIsPresent=True)
        save_plot(fname=prefix_histogram_dissimilarity+f"_SEED{seed}_TIPCT{ti_pct_area}-TINSHP{ti_nshapes}-{aux_var_names}.png", output_directory=plot_output_folder_complete, comments=f'TIPCT{ti_pct_area}-TINSHP{ti_nshapes}-{aux_var_names}', params={"nsim":nsim})
        
        #3 HISTOGRAMS
        plot_simvar_histograms(all_sim, nsim)
        save_plot(fname=prefix_simvar_histograms+f"_SEED{seed}_TIPCT{ti_pct_area}-TINSHP{ti_nshapes}-{aux_var_names}.png", output_directory=plot_output_folder_complete, comments=f'TIPCT{ti_pct_area}-TINSHP{ti_nshapes}-{aux_var_names}', params={"nsim":nsim})
        
        #4 TOPOLOGICAL ADAJCENCY
        plot_topological_adjacency(dist_topo_hamming, nsim, referenceIsPresent=True)
        save_plot(fname=prefix_topological_adjacency+f"_SEED{seed}_TIPCT{ti_pct_area}-TINSHP{ti_nshapes}-{aux_var_names}.png", output_directory=plot_output_folder_complete, comments=f'TIPCT{ti_pct_area}-TINSHP{ti_nshapes}-{aux_var_names}', params={"nsim":nsim})
        
        #5 PROPORTIONS
        plot_proportions(sim)
        save_plot(fname=prefix_proportions+f"_SEED{seed}_TIPCT{ti_pct_area}-TINSHP{ti_nshapes}-{aux_var_names}.png", output_directory=plot_output_folder_complete, comments=f'TIPCT{ti_pct_area}-TINSHP{ti_nshapes}-{aux_var_names}', params={"nsim":nsim})
        
        #6 STANDARD DEVIATON
        plot_standard_deviation(std_ent, realizations_range1, indicator_name="Entropy")
        save_plot(fname=prefix_std_deviation+f"_entropy_TIPCT{ti_pct_area}-TINSHP{ti_nshapes}-{aux_var_names}.png", output_directory=plot_output_folder_complete, comments=f'entropy_TIPCT{ti_pct_area}-TINSHP{ti_nshapes}-{aux_var_names}', params={"nsim":nsim})
        plot_standard_deviation(std_dist_hist, realizations_range2, indicator_name="Jensen Shanon divergence")
        save_plot(fname=prefix_std_deviation+f"_dist_histogram_TIPCT{ti_pct_area}-TINSHP{ti_nshapes}-{aux_var_names}.png", output_directory=plot_output_folder_complete, comments=f'dist_histogram_TIPCT{ti_pct_area}-TINSHP{ti_nshapes}-{aux_var_names}', params={"nsim":nsim})
        plot_standard_deviation(std_dist_hamming, realizations_range3, indicator_name="Topological adjacency")
        save_plot(fname=prefix_std_deviation+f"_dist_topo_hamming_TIPCT{ti_pct_area}-TINSHP{ti_nshapes}-{aux_var_names}.png", output_directory=plot_output_folder_complete, comments=f'dist_topo_hamming_TIPCT{ti_pct_area}-TINSHP{ti_nshapes}-{aux_var_names}', params={"nsim":nsim})
        
        timelog = end_timer_and_log(t0_plot, timelog)
        
        time_folder = os.path.join(output_directory, plot_output_folder, "time")
        os.makedirs(time_folder, exist_ok=True)
        timelogname = os.path.join(time_folder, f"timing_log.csv") #_TIPCT{ti_pct_area}-TINSHP{ti_nshapes}-{aux_var_names}
        
        save_log_to_csv(timelog, filename=timelogname)
        
        if verbose:
            print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S:%f)') + " <> FINISHED PLOTTING AND SAVING THE INDICATORS")
    
    #---- METHOD 2 : for multiple sets of TI and CD ----#
    
    else:
        if len(ti_methods) > 1:
            raise ValueError(f"Cannot run the following methods: {', '.join(ti_methods)} for {nRandomTICDsets} random TI/CD sets. Please consider chosing only one method or only one set.")

        cd_lists, ti_lists, nc_sg, nr_sg, simgrid_mask_final = gen_n_random_ti_cd(nRandomTICDsets, nc, nr,
                        sim_var = sim_var, auxTI_var = auxTI_var, auxSG_var = auxSG_var,
                        names_var = names_var, 
                        simgrid_mask = simgrid_mask_aux,
                        condIm_var = condIm_var,
                        method = ti_methods[0],
                        ti_pct_area = ti_pct_area, ti_nshapes = ti_nshapes, 
                        pct_ti_sg_overlap = pct_ti_sg_overlap, 
                        pct_sg = pct_sg, pct_ti = pct_ti, 
                        cc_sg = cc_sg, rr_sg = rr_sg, 
                        cc_ti = cc_ti, rr_ti = rr_ti,
                        givenseed = seed)
                        
        for cd_list, ti_list in zip(cd_lists, ti_lists):
            nTI = len(ti_list)
            names, distance_types = get_unique_names_and_types(names_var, types_var)
            
            outputFlag = []
            for name in names:
                outputFlag.append(outputVarFlag[name])

            deesse_input = gn.deesseinterface.DeesseInput(
                nx=nc_sg, ny=nr_sg, nz=1,
                sx=1, sy=1, sz=1,
                ox=0, oy=0, oz=0,
                nv=nvar, varname=names,
                TI=ti_list,
                #pdfTI = pdf_ti,
                mask = simgrid_mask_final,
                dataImage=cd_list,
                distanceType=distance_types,
                nneighboringNode=nvar*[nn],
                distanceThreshold=nvar*[dt],
                maxScanFraction=nTI*[ms],
                outputVarFlag=outputFlag,
                npostProcessingPathMax=1,
                seed=seed,
                nrealization=numberofmpsrealizations
            ) 
            
            deesse_output = gn.deesseinterface.deesseRun(deesse_input)
            
            save_simulation(deesse_output, params, comments="", output_directory="output/")
                
            ent, dist_hist, dist_topo_hamming = calculate_indicators(deesse_output, n_sim_variables=1, reference_var = np.load(r"C:\Users\00115212\Documents\GeoclassificationMPS\data\grid_geo.npy"))
           
    return
    


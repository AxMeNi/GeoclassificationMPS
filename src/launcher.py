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
from variability import calculate_indicators

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime 
from loopui import entropy


#### COLORS PARAMETERS
# cm = plt.get_cmap('tab20')
# defaultclrs = np.asarray(cm.colors)[[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11], :]
# n_bin = 11
# cmap_name = 'my_tab20'
# defaultcmap = LinearSegmentedColormap.from_list(cmap_name, defaultclrs, N=n_bin)
# defaultticmap = LinearSegmentedColormap.from_list('ticmap', np.vstack(([0, 0, 0], defaultclrs)), N=n_bin + 1)



def launcher(params,
            nvar, sim_var, auxTI_var, auxSG_var, condIm_var, names_var, types_var, outputVarFlag,
            nr, nc,
            verbose):
    """

    """
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
    saveOutput = params['saveOutput']
    output_directory = params['output_directory']
    deesse_output_folder = params['deesse_output_folder']
    prefix_deesse_output = params['prefix_deesse_output']
    plot_output_folder = params['plot_output_folder']
    prefix_histogram_disimilarity = params['prefix_histogram_disimilarity']
    prefix_entropy = params['prefix_entropy']
    prefix_simvar_histograms = params['prefix_simvar_histograms']
    prefix_topological_adjacency = params['prefix_topological_adjacency']
    reference_var = params['reference_var']
    prefix_proportions = params['prefix_proportions']
    
    deesse_output_folder_complete = output_directory+r"\\"+deesse_output_folder
    plot_output_folder_complete = output_directory+r"\\"+plot_output_folder
    
    #variables initialization
    ti_list = []
    cd_list = []
    
    #Create a simulation grid mask based on no values of the auxiliary variables
    if verbose:
        print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S:%f)') + " <> INITIATE CREATION OF THE SIMULATION GRID, OF THE CONDITIONING DATA, AND OF THE TI")
        
    simgrid_mask_aux = create_sg_mask(auxTI_var, auxSG_var, nr, nc)

    if nRandomTICDsets == 1 :
        
        for i_mask in range(1,3):
            params['seed'] = seed+i_mask
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
                
            if "DependentSquares" in ti_methods :
                if verbose:
                    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S:%f)') + " <> USING METHOD DEPENDENT SQUARES")
                ti_frame_DS, ntc_DS = gen_ti_frame_squares(nr, nc, ti_pct_area, ti_nshapes, seed)
                ti_list_DS, cd_list_DS = build_ti_cd(ti_frame_DS, ntc_DS, sim_var, nc, nr, auxTI_var, auxSG_var, names_var, simgrid_mask_aux, condIm_var)
                ti_list.extend(ti_list_DS)
                cd_list.extend(cd_list_DS)
                simgrid_mask = simgrid_mask_aux
                cc_sg, rr_sg = nc, nr
                
            if "IndependentSquares" in ti_methods :
                if verbose:
                    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S:%f)') + " <> USING METHOD INDEPENDENT SQUARES")
                ti_frame_IS, ntc_IS = gen_ti_frame_separatedSquares(nr, nc, ti_pct_area, ti_nshapes, seed)
                ti_list_IS, cd_list_IS = build_ti_cd(ti_frame_IS, ntc_IS, sim_var, nc, nr, auxTI_var, auxSG_var, names_var, simgrid_mask_aux, condIm_var)
                ti_list.extend(ti_list_IS)
                cd_list.extend(cd_list_IS)
                simgrid_mask = simgrid_mask_aux
                cc_sg, rr_sg = nc, nr
                
            if "ReducedTiSg" in ti_methods :
                if verbose:
                    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S:%f)') + " <> USING METHOD DEPENDENT REDUCED TI AND SG")
                ti_frame_RTS, ntc_RTS, simgrid_mask_RTS, cc_sg, rr_sg = gen_ti_frame_sg_mask(nr, nc, pct_ti_sg_overlap, pct_sg, pct_ti, cc_sg, rr_sg, cc_ti, rr_ti, seed)
                simgrid_mask_merged = merge_masks(simgrid_mask_RTS, simgrid_mask_aux)
                ti_list_RTS, cd_list_RTS = build_ti_cd(ti_frame_RTS, ntc_RTS, sim_var, cc_sg, rr_sg, auxTI_var, auxSG_var, names_var, simgrid_mask_merged, condIm_var)
                ti_list.extend(ti_list_RTS)
                cd_list.extend(cd_list_RTS)
                simgrid_mask = None
                
            if verbose:
                print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S:%f)') + f" <> Data dimension : \n·····>> Number of rows : {nr} \n·····>> Number of columns : {nc}")
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
                mask = simgrid_mask,
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
            
            deesse_output = gn.deesseinterface.deesseRun(deesse_input, nthreads = nthreads)    
              
            if saveOutput:
                save_simulation(deesse_output, params, output_directory=deesse_output_folder_complete)        
            
            #TWO PARAMETERS USED BELOW
            nsim=numberofmpsrealizations
            n_sim_variables=1
            
            #PARAMETERS FOR RETRIEVING THE SIMULATION
            sim = deesse_output['sim']
            all_sim_img = gn.img.gatherImages(sim) #Using the inplace functin of geone to gather images
            all_sim = all_sim_img.val
            all_sim = np.transpose(all_sim,(1,2,3,0))
            
            #CALCULATION OF THE INDICATORS
            ent, dist_hist, dist_topo_hamming = calculate_indicators(deesse_output, n_sim_variables=n_sim_variables, reference_var=reference_var)
            
            #1 ENTROPY
            plot_entropy(ent, background_image=reference_var, categ_var_name="Lithofacies")
            save_plot(fname=prefix_entropy+f"_msk{i_mask}.png", output_directory=plot_output_folder_complete, comments=f'{i_mask}', params={"nsim":nsim})
            
            #2 HISTOGRAM DISIMILARITY
            plot_histogram_disimilarity(dist_hist, seed, nsim, referenceIsPresent=True)
            save_plot(fname=prefix_histogram_disimilarity+f"_msk{i_mask}.png", output_directory=plot_output_folder_complete, comments=f'{i_mask}', params={"nsim":nsim})
            
            #3 HISTOGRAMS
            plot_simvar_histograms(all_sim, nsim)
            save_plot(fname=prefix_simvar_histograms+f"_msk{i_mask}.png", output_directory=plot_output_folder_complete, comments=f'{i_mask}', params={"nsim":nsim})
            
            #4 TOPOLOGICAL ADAJCENCY
            plot_topological_adjacency(dist_hist,dist_topo_hamming, nsim, referenceIsPresent=True)
            save_plot(fname=prefix_topological_adjacency+f"_msk{i_mask}.png", output_directory=plot_output_folder_complete, comments=f'{i_mask}', params={"nsim":nsim})
            
            #5 PROPORTIONS
            plot_proportions(sim)
            save_plot(fname=prefix_proportions+f"_msk{i_mask}.png", output_directory=plot_output_folder_complete, comments=f'{i_mask}', params={"nsim":nsim})
            

            # plt.show()
    
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
            
            
            
            ###############################################################################
            # sim = deesse_output['sim']
            # all_sim = gn.img.gatherImages(sim)
            # categ_val = [1,2,3,4,5,6,7]
            # all_sim_stats = gn.img.imageCategProp(all_sim, categ_val)
            # prop_col = ['lightblue', 'blue', 'orange', 'green', 'red', 'purple', 'yellow']
            # cmap = [gn.customcolors.custom_cmap(['white', c]) for c in prop_col]
            # plt.subplots(1, 7, figsize=(17,5), sharey=True)
            # for i in range(7):
                # plt.subplot(1, 7, i+1) # select next sub-plot
                # gn.imgplot.drawImage2D(all_sim_stats, iv=i, cmap=cmap[i],
                                       # title=f'Prop. of categ. {i}')
            # plt.show()
            
    return
    


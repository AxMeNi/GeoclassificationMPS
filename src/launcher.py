# -*- coding:utf-8 -*-
__projet__ = "GeoclassificationMPS"
__nom_fichier__ = "launcher"
__author__ = "MENGELLE Axel"
__date__ = "juillet 2024"

from ti_mask_generation import *
from data_treatment import get_unique_names_and_types
from sg_mask_generation import *
from build_ti_cd import *
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




def launcher(seed, 
            ti_methods, 
            ti_pct_area, ti_nshapes,
            pct_ti_sg_overlap, pct_sg, pct_ti, cc_sg, rr_sg, cc_ti, rr_ti, nRandomTICDsets,
            nn, dt, ms, numberofmpsrealizations, nthreads,
            cm, myclrs, n_bin, cmap_name, mycmap, ticmap,
            nvar, sim_var, auxTI_var, auxSG_var, condIm_var, names_var, types_var, outputVarFlag,
            nr, nc
            ):
    """

    """
    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)') + " - INIT")
    
    #variables initialization
    ti_list = []
    cd_list = []
    
    #Create a simulation grid mask based on no values of the auxiliary variables
    simgrid_mask_aux = create_sg_mask(auxTI_var, auxSG_var, nr, nc)

    if nRandomTICDsets == 1 :
    
        #Creation of the TI and of the SG
        if "DependentCircles" in ti_methods :
            ti_frame_DC, ntc_DC = gen_ti_frame_circles(nr, nc, ti_pct_area, ti_nshapes, seed)
            ti_list_DC, cd_list_DC = build_ti_cd(ti_frame_DC, ntc_DC, sim_var, nc, nr, auxTI_var, auxSG_var, names_var, simgrid_mask_aux, condIm_var)
            ti_list.extend(ti_list_DC)
            cd_list.extend(cd_list_DC)
            simgrid_mask = simgrid_mask_aux
            cc_sg, rr_sg = nc, nr
            
        if "DependentSquares" in ti_methods :
            ti_frame_DS, ntc_DS = gen_ti_frame_squares(nr, nc, ti_pct_area, ti_nshapes, seed)
            ti_list_DS, cd_list_DS = build_ti_cd(ti_frame_DS, ntc_DS, sim_var, nc, nr, auxTI_var, auxSG_var, names_var, simgrid_mask_aux, condIm_var)
            ti_list.extend(ti_list_DS)
            cd_list.extend(cd_list_DS)
            simgrid_mask = simgrid_mask_aux
            cc_sg, rr_sg = nc, nr
            
        if "IndependentSquares" in ti_methods :
            ti_frame_IS, ntc_IS = gen_ti_frame_separatedSquares(nr, nc, ti_pct_area, ti_nshapes, seed)
            ti_list_IS, cd_list_IS = build_ti_cd(ti_frame_IS, ntc_IS, sim_var, nc, nr, auxTI_var, auxSG_var, names_var, simgrid_mask_aux, condIm_var)
            ti_list.extend(ti_list_IS)
            cd_list.extend(cd_list_IS)
            simgrid_mask = simgrid_mask_aux
            cc_sg, rr_sg = nc, nr
            
        if "ReducedTiSg" in ti_methods :
            ti_frame_RTS, ntc_RTS, simgrid_mask_RTS, cc_sg, rr_sg = gen_ti_frame_sg_mask(nr, nc, pct_ti_sg_overlap, pct_sg, pct_ti, cc_sg, rr_sg, cc_ti, rr_ti, seed)
            simgrid_mask_merged = merge_masks(simgrid_mask_RTS, simgrid_mask_aux)
            ti_list_RTS, cd_list_RTS = build_ti_cd(ti_frame_RTS, ntc_RTS, sim_var, cc_sg, rr_sg, auxTI_var, auxSG_var, names_var, simgrid_mask_merged, condIm_var)
            ti_list.extend(ti_list_RTS)
            cd_list.extend(cd_list_RTS)
            simgrid_mask = None
        
        
            
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
        
        deesse_output = gn.deesseinterface.deesseRun(deesse_input)

        sim = deesse_output['sim']
        
        ###############################################################################
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
        ###############################################################################
        
        # plt.subplots(1, 1, figsize=(17,10), sharex=True, sharey=True)
        
        # gn.imgplot.drawImage2D(sim[0], iv=0, categ=True, title=f'Real #{0} - {deesse_input.varname[0]}')
        
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

            sim = deesse_output['sim']
            
            ###############################################################################
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

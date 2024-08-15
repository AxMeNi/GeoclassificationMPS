# -*- coding:utf-8 -*-
__projet__ = "GeoclassificationMPS"
__nom_fichier__ = "interface"
__author__ = "MENGELLE Axel"
__date__ = "juillet 2024"

import numpy as np
import geone as gn
from time import time


    
def build_ti_cd(ti_frames_list, 
                need_to_cut, 
                sim_var, 
                nc_simgrid, nr_simgrid, 
                auxTI_var, auxSG_var,
                names_var,
                simgrid_mask = None,
                cond_var = None):
    """
    """
    # Building TI(s)
    ti_list = []
    
    for i in range(len(ti_frames_list)):
        ti_frame = ti_frames_list[i]
        ntc = need_to_cut[i]
        
        #Case for which a cut is needed
        if ntc:
        
            name = "TI{}_{}".format(i,time())
            ti = gn.img.Img(nv=0,name=name)
            
            #Reshape of the simulated var and integration to the TI
            for var_name, var_value in sim_var.items():
            
                var_value_masked = np.where(ti_frame == 1, var_value, np.nan)

                rows = np.any(ti_frame, axis=1)
                cols = np.any(ti_frame, axis=0)
                
                row_start, row_end = np.where(rows)[0][[0, -1]]
                col_start, col_end = np.where(cols)[0][[0, -1]]
                
                ti.set_grid(nx=col_end-col_start+1, ny=row_end-row_start+1, nz=1, sx=1, sy=1, sz=1, ox=0, oy=0, oz=0)
                
                var_value_cut = var_value_masked[row_start:row_end+1, col_start:col_end+1]
                
                ti.append_var(val=var_value_cut, varname=var_name)
                
            #Reshape of the auxiliary desciptive var and integration to the TI
            for var_name, var_value in auxTI_var.items():
                var_value_masked = np.where(ti_frame == 1, var_value, np.nan)

                rows = np.any(ti_frame, axis=1)
                cols = np.any(ti_frame, axis=0)

                row_start, row_end = np.where(rows)[0][[0, -1]]
                col_start, col_end = np.where(cols)[0][[0, -1]]
                
                ti.set_grid(nx=col_end-col_start+1, ny=row_end-row_start+1, nz=1, sx=1, sy=1, sz=1, ox=0, oy=0, oz=0)

                var_value_cut = var_value_masked[row_start:row_end+1, col_start:col_end+1]
                
                ti.append_var(val=var_value_cut, varname=var_name)
                           
            gn.img.writeImageTxt(f"TI{i}.txt", ti)      
            ti = gn.img.readImageTxt(f"TI{i}.txt")    
            ti_list.append(ti)
            
        #Case for which no cut is needed
        else:
        
            name = "TI{}_{}".format(i,time())
            ti = gn.img.Img(nv=0,name=name)
                       
            ti.set_grid(nx=nc_simgrid, ny=nr_simgrid, nz=1, sx=1, sy=1, sz=1, ox=0, oy=0, oz=0)
            
            #Integration of sim_var in the TI
            for var_name, var_value in sim_var.items() :
                var_value_masked = np.where(ti_frame == 1, var_value, np.nan)
   
                ti.append_var(val=var_value_masked, varname=var_name)
                
            
            #No application of the mask to the auxiliary var which have to be fully informed and integration to the TI
            for var_name, var_value in auxTI_var.items() :

                ti.append_var(val=var_value, varname=var_name)
            
            ti_list.append(ti)
          
        
    
    # Building conditioning AUXILIARY data
    cd_list = []

    name = "CondData{}_{}".format(i,time())
    cd = gn.img.Img(nv=0, name=name)
    
    # Integration of the auxiliary_var in the simulation grid to control the non stationarity
    # The values of the aux var here is to control the non stationarity of the data
    for var_name, var_value in auxSG_var.items():
        if var_value.shape != (nr_simgrid,nc_simgrid) :            
            var_value_masked = np.where(simgrid_mask == 1, var_value, np.nan)

            rows = np.any(simgrid_mask, axis=1)
            cols = np.any(simgrid_mask, axis=0)

            row_start, row_end = np.where(rows)[0][[0, -1]]
            col_start, col_end = np.where(cols)[0][[0, -1]]
            
            cd.set_grid(nx=col_end-col_start, ny=row_end-row_start, nz=1, sx=1, sy=1, sz=1, ox=0, oy=0, oz=0)

            var_value_cut = var_value_masked[row_start:row_end, col_start:col_end]   
            cd.append_var(val=var_value_cut, varname=var_name)
            
        else :
            var_value_masked = np.where(simgrid_mask == 1, var_value, np.nan)
            cd.set_grid(nx=nc_simgrid, ny=nr_simgrid, nz=1, sx=1, sy=1, sz=1, ox=0, oy=0, oz=0)
            cd.append_var(val=var_value_masked, varname=var_name)
        
    gn.img.writeImageTxt(f"CD.txt", cd)      
    cd = gn.img.readImageTxt(f"CD.txt")     
    
    cd_list.append(cd)       
            
    #Building conditioning SIMULATED data 
    
    return ti_list, cd_list
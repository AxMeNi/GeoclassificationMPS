# -*- coding:utf-8 -*-
__projet__ = "GeoclassificationMPS"
__nom_fichier__ = "main"
__author__ = "MENGELLE Axel"
__date__ = "juillet 2024"

from tests import *
from interface import *

def run_tests():
    """
    to run the tests
    """
    t_GetSimulationInfo =           False
    
    t_CheckTiMethods =              False
    t_CreateVariables =             False
    t_CountVariables =              False
    t_CheckVariables =              False
    t_GetSimGridDimensions =        False
    t_GetUniqueNamesAndTypes =      False
    
    t_generate_random_dimensions =  False #TODO
    t_chose_random_dimensions =     False #TODO
    t_generate_random_sg_origin =   False #TODO
    t_chose_random_sg_origin =      False #TODO
    t_chose_random_overlap_area =   False #TODO
    t_compute_row =                 False #TODO
    t_chose_random_overlap_origin = False #TODO
    t_get_ti_orign =                False #TODO
    t_check_ti_pos =                False #TODO
    t_get_ti_sg =                   False #TODO
    
    t_CreateSGMask =                False
    t_MergeMasks =                  False
    
    t_GenTiFrameCircles =           False
    t_GenTiFrameSquares =           False
    t_GenTiFrameSeparatedSquares =  False
    t_GenTiFrameCdMask =            False
    
    t_BuildTiCd =                   False #TODO faire le cas où il y a des conditioning data et tout le reste...
    t_GenNRandomTiCd =              False
    
    #----------------------------------------------------------------------------------------#
    
    if t_GetSimulationInfo : test_get_simulation_info()
    
    if t_CheckTiMethods : test_check_ti_methods()
    if t_CheckVariables : test_check_variables()
    if t_CountVariables : test_count_variables()
    if t_CreateVariables : test_create_variables() 
    if t_GetSimGridDimensions : test_get_sim_grid_dimensions()
    if t_GetUniqueNamesAndTypes : test_get_unique_names_and_types()
    
    if t_CreateSGMask : test_create_sg_mask()
    if t_MergeMasks : test_merge_masks()
    
    if t_GenTiFrameCircles : test_gen_ti_frame_circles()
    if t_GenTiFrameSquares : test_gen_ti_frame_squares()
    if t_GenTiFrameSeparatedSquares : test_gen_ti_frame_separated_squares(showCoord=False)
    if t_GenTiFrameCdMask : test_gen_ti_frame_cd_mask()
    
    if t_BuildTiCd : test_build_ti_cd()
    if t_GenNRandomTiCd : test_gen_n_random_ti_cd()


def run_simulations():
    launch_simulation()
    
    
    
if __name__ == '__main__':

    bool_run_simu = True
    bool_run_tests = False
    
    if bool_run_simu: run_simulation()
    if bool_run_tests: run_tests()
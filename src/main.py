# -*- coding:utf-8 -*-
__projet__ = "GeoclassificationMPS"
__nom_fichier__ = "main"
__author__ = "MENGELLE Axel"
__date__ = "juillet 2024"

from tests import *

def run_tests():
    """
    to run the tests
    """
    t_GetSimulationInfo = False
    t_CheckVariables = False
    t_CreateAuxiliaryAndSimultedVariable = False
    t_GetSimGridDimensions = False #TODO
    
    # t_generate_random_dimensions #TODO
    # t_chose_random_dimensions #TODO
    # generate_random_sg_origin #TODO
    # chose_random_sg_origin #TODO
    # chose_random_overlap_area #TODO
    # compute_row #TODO
    # chose_random_overlap_origin #TODO
    # get_ti_orign #TODO
    # check_ti_pos #TODO
    # get_ti_sg #TODO
    
    t_GenTiFrameCircles = False
    t_GenTiFrameSquares = False
    t_GenTiFrameSeparatedSquares = False
    t_GenTiFrameSingleRectangle = False
    t_BuildTi = True #TODO faire le cas où il y a des conditioning data et tout le reste...
    
    #----------------------------------------------------------------------------------------#
    
    if t_GetSimulationInfo : test_get_simulation_info()
    if t_CheckVariables : test_check_variables()
    if t_CreateAuxiliaryAndSimultedVariable : test_create_auxiliary_and_simulated_var() 
    if t_GetSimGridDimensions : test_get_sim_grid_dimensions()
    
    if t_GenTiFrameCircles : test_gen_ti_frame_circles()
    if t_GenTiFrameSquares : test_gen_ti_frame_squares()
    if t_GenTiFrameSeparatedSquares : test_gen_ti_frame_separated_squares(showCoord=True)
    if t_GenTiFrameSingleRectangle : test_gen_ti_frame_single_rectangle()
    if t_BuildTi : test_build_ti()


def run_simulations():
    launch_simulation()
    
    
    
if __name__ == '__main__':

    bool_run_simu = False
    bool_run_tests = True
    
    if bool_run_simu: run_simulations()
    if bool_run_tests: run_tests()
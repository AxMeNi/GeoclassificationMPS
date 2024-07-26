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
    t_GetSimGridDimensions = True #TODO
    t_GenTiFrameCircles = False
    t_GenTiFrameSquares = False
    t_GenTiFrameSeparatedSquares = False
    t_GenTiFrameSingleRectangle = False
    t_BuildTi = False #TODO
    
    #----------------------------------------------------------------------------------------#
    
    if t_GetSimulationInfo : test_get_simulation_info()
    if t_CheckVariables : test_check_variables()
    if t_CreateAuxiliaryAndSimultedVariable : test_create_auxiliary_and_simulated_var() 
    if t_GetSimGridDimensions : test_get_sim_grid_dimensions()
    if t_GenTiFrameCircles : test_gen_ti_frame_circles()
    if t_GenTiFrameSquares : test_gen_ti_frame_squares()
    if t_GenTiFrameSeparatedSquares : test_gen_ti_frame_separated_squares(showCoord=False)
    if t_GenTiFrameSingleRectangle : test_gen_ti_frame_single_rectangle()
    if t_BuildTi : test_build_ti()


def run_simulations():
    launch_simulation()
    
    
    
if __name__ == '__main__':

    bool_run_simu = False
    bool_run_tests = True
    
    if bool_run_simu: run_simulations()
    if bool_run_tests: run_tests()
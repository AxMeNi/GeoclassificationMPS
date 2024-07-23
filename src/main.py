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
    t_GenTiMaskCircles = True
    t_GenTiMaskSquares = True
    t_GenTiMaskSeparatedSquares = True
    t_GenTiMaskSingleSquare = True
    
    if t_GetSimulationInfo : test_get_simulation_info()
    if t_CheckVariables : test_check_variables()
    if t_GenTiMaskCircles : test_gen_ti_mask_circles()
    if t_GenTiMaskSquares : test_gen_ti_mask_squares()
    if t_GenTiMaskSeparatedSquares : test_gen_ti_mask_separated_squares(showCoord=False)
    if t_GenTiMaskSingleSquare : test_gen_ti_mask_single_square()


def run_simulations():
    launch_simulation()
    
    
    
if __name__ == '__main__':

    bool_run_simu = False
    bool_run_tests = True
    
    if bool_run_simu: run_simulations()
    if bool_run_tests: run_tests()
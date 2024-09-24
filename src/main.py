# -*- coding:utf-8 -*-
__projet__ = "GeoclassificationMPS"
__nom_fichier__ = "main"
__author__ = "MENGELLE Axel"
__date__ = "juillet 2024"

from tests import *
from interface import *
import sys

def run_tests():
    """
    to run the tests
    """
#INTERFACE.PY
    t_GetSimulationInfo =           False

#DATA_TREATMENT.PY
    t_CheckTiMethods =              False
    t_CreateVariables =             True
    t_CountVariables =              False
    t_CheckVariables =              False
    t_GetSimGridDimensions =        False
    t_GetUniqueNamesAndTypes =      False

#REDUCED_TI_SG.PY
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

#SG_MASK_GENERATION.PY
    t_CreateSGMask =                False
    t_MergeMasks =                  False

#TI_MASK_GENERATION.PY
    t_GenTiFrameCircles =           False
    t_GenTiFrameSquares =           False
    t_GenTiFrameSeparatedSquares =  False
    t_GenTiFrameCdMask =            False

#BUILD_TI_CD.PY
    t_BuildTiCd =                   False #TODO faire le cas où il y a des conditioning data et tout le reste...
    t_GenNRandomTiCd =              False

#VARIABILITY.PY
    t_CustJsdistHist =              False #TODO
    t_CustTopologicalAdjacency2D =  False #TODO
    t_CustomTopoDist =              False #TODO
    t_CalculateIndicators =         False #TODO

#DISPLAY_FUNCTIONS.PY
    t_PlotEntropy =                 False #TODO
    t_PlotHistogramDisimilarity =   False #TODO
    t_PlotTopologicalAdjacency =    False #TODO

#PROPORTIONS.PY
    t_GetBins =                     False

#UTILS.PY
    tCartesianProduct =             False

#SAVING.PY
    tSaveDeesseOutput =             False #TODO
    tSaveSimulation =               False #TODO
    
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
    if t_GenTiFrameCdMask : test_gen_ti_frame_sg_mask()
    
    if t_BuildTiCd : test_build_ti_cd()
    if t_GenNRandomTiCd : test_gen_n_random_ti_cd()
    
    if t_CustJsdistHist : test_custom_jsdist_hist()
    if t_CustTopologicalAdjacency2D : test_custom_topological_adjacency2D()
    if t_CustomTopoDist : test_custom_topo_dist()
    if t_CalculateIndicators : test_calculate_indicators()
    
    if t_PlotEntropy : test_plot_entropy()
    if t_PlotHistogramDisimilarity : test_plot_histogram_disimilarity()
    if t_PlotTopologicalAdjacency : test_plot_topological_adjacency()

    if t_GetBins : test_get_bins()
    
    if tCartesianProduct : test_cartesian_product()
    
    if tSaveDeesseOutput : test_save_deesse_output()
    if tSaveSimulation : test_save_simulation()
    
if __name__ == '__main__':
    arg = sys.argv[1]
    
    if arg == "-s" : run_simulation()
    if arg == "-t" : run_tests()
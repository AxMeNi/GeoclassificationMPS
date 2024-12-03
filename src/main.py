# -*- coding:utf-8 -*-
__projet__ = "GeoclassificationMPS"
__nom_fichier__ = "main"
__author__ = "MENGELLE Axel"
__date__ = "juillet 2024"

from tests import *
from interface import *
import sys



def run_tests(verbose):
    """
    This function allows selective execution of various tests defined across multiple modules in the project.
    The function initializes a series of test flags corresponding to specific features and functionality 
    within different modules. Each flag represents a test case, and setting a flag to True enables that 
    particular test to be run when the main function is executed with the parameter "-t".
    """
#INTERFACE.PY
    t_GetSimulationInfo =           False

#DATA_TREATMENT.PY
    t_CheckTiMethods =              False
    t_CreateVariables =             False
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
    t_CalculateIndicators =         False
    t_CalculateStandardDeviation =  False

#DISPLAY_FUNCTIONS.PY
    t_PlotRealization =             False
    t_PlotMask =                    False
    t_PlotProportions =             False #TODO
    t_PlotEntropy =                 False #TODO
    t_PlotHistogramDissimilarity =   False #TODO
    t_PlotSimvarHistogram =         False #TODO
    t_PlotTopologicalAdjacency =    False #TODO
    t_PlotStandardDeviation =       False

#PROPORTIONS.PY
    t_GetBins =                     False

#UTILS.PY
    t_FindFarthestPtsFrmCentroids = False
    t_LoadPickleFile =              True

#SAVING.PY
    t_SaveDeesseOutput =            False #TODO
    t_SaveSimulation =              False #TODO
    t_SavePlot =                    False #TODO
    t_SaveLogToCSV =                False #TODO

#TIME_LOGGING.PY  
    t_TimingSystem =                False #TODO
    
    
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
    if t_CalculateStandardDeviation : test_calculate_std_deviation()
    
    if t_PlotRealization :test_plot_realization()
    if t_PlotMask : test_plot_mask()
    if t_PlotProportions : test_plot_proportions()
    if t_PlotEntropy : test_plot_entropy()
    if t_PlotHistogramDissimilarity : test_plot_histogram_dissimilarity()
    if t_PlotSimvarHistogram : test_plot_simvar_histograms()
    if t_PlotTopologicalAdjacency : test_plot_topological_adjacency()
    if t_PlotStandardDeviation : test_plot_std_deviation()

    if t_GetBins : test_get_bins()
    
    if t_FindFarthestPtsFrmCentroids : test_find_farthest_points_from_centroid()
    if t_LoadPickleFile : test_load_pickle_file()
    
    if t_SaveDeesseOutput : test_save_deesse_output()
    if t_SaveSimulation : test_save_simulation()
    if t_SavePlot : test_save_plot()
    if t_SaveLogToCSV : test_save_log_to_csv()
    
    if t_TimingSystem : test_timing_system()
    
    
    
if __name__ == '__main__':
    verbose = False
    try: 
        arg = sys.argv[1]
    except IndexError:
        print("To use the program you must type \"python main.py\" followed by one of the following command:\n\
            >> -t to launch the test program for debugging\n\
            >> -s to launch the simulation\n\
            >> -v for verbose mode (optional with -t or -s)")
        exit()
    
    if "-v" in sys.argv:
        verbose = True
    if "-s" in sys.argv:
        run_simulation(verbose)
    elif "-t" in sys.argv:
        run_tests(verbose)
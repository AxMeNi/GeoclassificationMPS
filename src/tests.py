# -*- coding:utf-8 -*-
__projet__ = "GeoclassificationMPS"
__nom_fichier__ = "tests"
__author__ = "MENGELLE Axel"
__date__ = "juillet 2024"


from data_treatment import *
from interface import get_simulation_info
from ti_mask_generation import *
from sg_mask_generation import *
from build_ti_cd import *
# from proportions import *
from utils import *

from matplotlib.colors import *

import matplotlib.pyplot as plt
import os


##################################### TEST INTERFACE.PY

def test_get_simulation_info():
    # Test Case: Check if the function returns all the expected variables
    seed, \
    ti_method, \
    ti_pct_area, ti_shapes, \
    pct_ti_sg_overlap, pct_sg, pct_ti, cc_sg, rr_sg, cc_ti, rr_ti, \
    nn, dt, ms, numberofmpsrealizations, nthreads, \
    cm, myclrs, n_bin, cmap_name, mycmap, ticmap, \
    shorten, \
    sim_var, auxTI_var, auxSG_var, condIm_var, names_var, types_var, \
    nr, nc = get_simulation_info()

    assert isinstance(seed, int), "Seed should be an integer."
    assert isinstance(ti_method, list) and all(isinstance(m, str) for m in ti_method), "ti_method should be a list of strings."
    assert ti_pct_area is None or isinstance(ti_pct_area, float), "ti_pct_area should be None or a float."
    assert isinstance(ti_shapes, int), "ti_shapes should be an integer."
    assert isinstance(pct_ti_sg_overlap, int), "pct_ti_sg_overlap should be an integer."
    assert isinstance(pct_sg, int) and isinstance(pct_ti, int), "pct_sg and pct_ti should be integers."
    assert cc_sg is None or isinstance(cc_sg, int), "cc_sg should be None or an integer."
    assert rr_sg is None or isinstance(rr_sg, int), "rr_sg should be None or an integer."
    assert cc_ti is None or isinstance(cc_ti, int), "cc_ti should be None or an integer."
    assert rr_ti is None or isinstance(rr_ti, int), "rr_ti should be None or an integer."
    assert isinstance(nn, int), "nn should be an integer."
    assert isinstance(dt, float), "dt should be a float."
    assert isinstance(ms, float), "ms should be a float."
    assert isinstance(numberofmpsrealizations, int), "numberofmpsrealizations should be an integer."
    assert isinstance(nthreads, int), "nthreads should be an integer."
    assert isinstance(cm, ListedColormap), "cm should be a LinearSegmentedColormap instance."
    assert isinstance(myclrs, np.ndarray), "myclrs should be a numpy array."
    assert isinstance(n_bin, int), "n_bin should be an integer."
    assert isinstance(cmap_name, str), "cmap_name should be a string."
    assert isinstance(mycmap, LinearSegmentedColormap), "mycmap should be a LinearSegmentedColormap instance."
    assert isinstance(ticmap, LinearSegmentedColormap), "ticmap should be a LinearSegmentedColormap instance."
    assert isinstance(shorten, bool), "shorten should be a boolean."
    assert isinstance(sim_var, dict), "sim_var should be a dictionary."
    assert isinstance(auxTI_var, dict), "auxTI_var should be a dictionary."
    assert isinstance(auxSG_var, dict), "auxSG_var should be a dictionary."
    assert condIm_var is None or isinstance(condIm_var, dict), "condIm_var should be None or a dictionary."
    assert isinstance(names_var, list) and all(isinstance(name, str) for name in names_var[0]), "names_var should be a list of strings."
    assert isinstance(types_var, list) and all(isinstance(t, str) for t in types_var[0]), "types_var should be a list of strings."
    assert isinstance(nr, int) and isinstance(nc, int), "nr and nc should be integers."

    print("The function get_simulation_info is working correctly, all checks passed!")


##################################### TEST DATA_TREATMENT.PY

def test_check_ti_methods():
    try:
        check_ti_methods(["DependentCircles", "DependentSquares", "IndependentSquares", "ReducedTiSg"])
        print("Test case 1 passed.")
    except ValueError as e:
        print(f"Test case 1 failed: {e}")
    try:
        check_ti_methods(["DependentCircles", "IndependentSquares"])
        print("Test case 2 passed.")
    except ValueError as e:
        print(f"Test case 2 failed: {e}")
    try:
        check_ti_methods(["SomeOtherMethod"])
        print("Test case 3 failed: No error raised.")
    except ValueError as e:
        print(f"Test case 3 passed: {e}")
    try:
        check_ti_methods([])
        print("Test case 4 failed: No error raised.")
    except ValueError as e:
        print(f"Test case 4 passed: {e}")
    try:
        check_ti_methods(["ReducedTiSg"])
        print("Test case 5 passed.")
    except ValueError as e:
        print(f"Test case 5 failed: {e}")


def test_create_variables():
    """
    Test function for create_variables to ensure it properly categorizes variables
    and handles errors appropriately.
    """
    # Create a temporary directory to store the test numpy arrays
    temp_dir = "temp_test_dir"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Sample data for the test
    test_data = [
        {'var_name': 'sim_var1', 'categ_conti': 'categ', 'nature': 'sim', 'grid':'TI', 'path': f'{temp_dir}/array_sim1.npy'},
        {'var_name': 'auxTI_var1', 'categ_conti': 'conti', 'nature': 'auxTI', 'grid':'TI', 'path': f'{temp_dir}/array_auxTI1.npy'},
        {'var_name': 'auxSG_var1', 'categ_conti': 'categ', 'nature': 'auxSG', 'grid':'SG', 'path': f'{temp_dir}/array_auxSG1.npy'},
        {'var_name': 'condIm_var1', 'categ_conti': 'conti', 'nature': 'condIm', 'grid':'SG', 'path': f'{temp_dir}/array_condIm1.npy'}
    ]
    
    np.save(f'{temp_dir}/array_sim1.npy', np.array([1, 2, 3]))
    np.save(f'{temp_dir}/array_auxTI1.npy', np.array([4, 5, 6]))
    np.save(f'{temp_dir}/array_auxSG1.npy', np.array([7, 8, 9]))
    np.save(f'{temp_dir}/array_condIm1.npy', np.array([10, 11, 12]))
    
    df = pd.DataFrame(test_data)
    csv_file_path = f'{temp_dir}/test_data.csv'
    df.to_csv(csv_file_path, sep=';', index=False)
    
    sim_var, auxTI_var, auxSG_var, condIm_var, names_var, types_var = create_variables(csv_file_path)
    
    assert len(sim_var) == 1, "Expected 1 simulated variable."
    assert len(auxTI_var) == 1, "Expected 1 auxiliary TI variable."
    assert len(auxSG_var) == 1, "Expected 1 auxiliary SG variable."
    assert len(condIm_var) == 1, "Expected 1 conditioning image variable."
    
    assert np.array_equal(sim_var['sim_var1'], np.array([1, 2, 3])), "Unexpected data in simulated variable 'sim_var1'."
    assert np.array_equal(auxTI_var['auxTI_var1'], np.array([4, 5, 6])), "Unexpected data in auxiliary TI variable 'auxTI_var1'."
    assert np.array_equal(auxSG_var['auxSG_var1'], np.array([7, 8, 9])), "Unexpected data in auxiliary SG variable 'auxSG_var1'."
    assert np.array_equal(condIm_var['condIm_var1'], np.array([10, 11, 12])), "Unexpected data in conditioning image variable 'condIm_var1'."
    
    expected_names_var = [['sim_var1'], ['auxTI_var1'], ['auxSG_var1'], ['condIm_var1']]
    expected_types_var = [['categ'], ['conti'], ['categ'], ['conti']]
    
    assert names_var == expected_names_var, f"Unexpected names_var: {names_var}"
    assert types_var == expected_types_var, f"Unexpected types_var: {types_var}"
    
    os.remove(f'{temp_dir}/array_sim1.npy')
    os.remove(f'{temp_dir}/array_auxTI1.npy')
    os.remove(f'{temp_dir}/array_auxSG1.npy')
    os.remove(f'{temp_dir}/array_condIm1.npy')
    os.remove(csv_file_path)
    os.rmdir(temp_dir)

    print("Successfully passed test_create_variables.")


def test_count_variables():
    names_var1 = [["var1", "var2"],["var3", "var4"],["var5", "var6"],["var7", "var8"]]
    expected1 = 8
    result1 = count_variables(names_var1)
    assert result1 == expected1, f"Test case 1 failed: expected {expected1}, got {result1}"
    print(f"Test case 1 passed: {result1} unique variables.")
    names_var2 = [["var1", "var2"],["var3", "var2"],["var5", "var3"],["var6", "var7"]]
    expected2 = 6
    result2 = count_variables(names_var2)
    assert result2 == expected2, f"Test case 2 failed: expected {expected2}, got {result2}"
    print(f"Test case 2 passed: {result2} unique variables.")
    names_var3 = [["var1", "var1"],["var1", "var1"],["var1", "var1"],["var1", "var1"]]
    expected3 = 1
    result3 = count_variables(names_var3)
    assert result3 == expected3, f"Test case 3 failed: expected {expected3}, got {result3}"
    print(f"Test case 3 passed: {result3} unique variables.")
    names_var4 = [[],[],[],[]]
    expected4 = 0
    result4 = count_variables(names_var4)
    assert result4 == expected4, f"Test case 4 failed: expected {expected4}, got {result4}"
    print(f"Test case 4 passed: {result4} unique variables.")


def test_check_variables():
    # Test 1: Valid input (should pass)
    sim_var = {
        'var1': np.array([[1.0, 2.0], [3.0, -9999999]]),
        'var2': np.array([[4.0, 5.0], [6.0, 7.0]])
    }
    auxTI_var = {
        'aux1': np.array([[10.0, 20.0], [30.0, 40.0]])
    }
    auxSG_var = {
        'aux1': np.array([[50, 60], [70, 80]])
    }
    condIm_var = {
        'var1': np.array([[1.0, 2.0], [3.0, 4.0]])
    }
    names_var = [['var1', 'var2'], ['aux1'], ['aux1'], ['var1']]
    types_var = [['continuous', 'continuous'], ['continuous'], ['continuous'], ['continuous']]
    try:
        sim_var_out, auxTI_var_out, auxSG_var_out, condIm_var_out = check_variables(sim_var, auxTI_var, auxSG_var, condIm_var, names_var, types_var)
        print("Test 1 passed")
    except Exception as e:
        print(f"Test 1 failed: {e}")

    # Test 2: Type mismatch in simulated variable (should raise TypeError)
    sim_var['var1'] = np.array([['a', 'b'], ['c', 'd']])
    try:
        check_variables(sim_var, auxTI_var, auxSG_var, condIm_var, names_var, types_var)
        print("Test 2 failed: TypeError was expected")
    except TypeError as e:
        print(f"Test 2 passed: {e}")

    # Test 3: Shape inconsistency (should raise ValueError)
    sim_var['var1'] = np.array([[1.0, 2.0]])
    try:
        check_variables(sim_var, auxTI_var, auxSG_var, condIm_var, names_var, types_var)
        print("Test 3 failed: ValueError was expected")
    except ValueError as e:
        print(f"Test 3 passed: {e}")

    # Test 4: Name mismatch in conditioning variables (should raise NameError)
    sim_var = {
        'var1': np.array([[1.0, 2.0], [3.0, -9999999]]),
        'var2': np.array([[4.0, 5.0], [6.0, 7.0]])
    }
    names_var[3] = ['var3']
    condIm_var['var3'] = condIm_var['var1']
    try:
        check_variables(sim_var, auxTI_var, auxSG_var, condIm_var, names_var, types_var)
        print("Test 4 failed: NameError was expected")
    except NameError as e:
        print(f"Test 4 passed: {e}")

    # Test 5: Missing auxiliary conditioning variable (should raise NameError)
    names_var[3] = ['var1']
    names_var[2] = []
    try:
        check_variables(sim_var, auxTI_var, auxSG_var, condIm_var, names_var, types_var)
        print("Test 5 failed: NameError was expected")
    except NameError as e:
        print(f"Test 5 passed: {e}")

    # Test 6: Valid input with novalue handling (should pass and replace novalue with np.nan)
    names_var = [['var1', 'var2'], ['aux1'], ['aux1'], ['var1']]
    sim_var['var1'] = np.array([[1.0, 2.0], [3.0, -9999999]])
    try:
        sim_var_out, auxTI_var_out, auxSG_var_out, condIm_var_out = check_variables(sim_var, auxTI_var, auxSG_var, condIm_var, names_var, types_var)
        assert np.isnan(sim_var_out['var1'][1, 1]), "Test 6 failed: novalue was not replaced by np.nan"
        print("Test 6 passed")
    except Exception as e:
        print(f"Test 6 failed: {e}")


def test_get_sim_grid_dimensions():
    csv_file_path = r"C:\Users\00115212\Documents\GeoclassificationMPS\test\data_csv.csv"
    sim_var, auxTI_var, auxSG_var, condIm_var, names_var, types_var = create_variables(csv_file_path)
    nr, nc = get_sim_grid_dimensions(sim_var)
    print(nr,nc)
    
    return True


def test_get_unique_names_and_types():

    # Test Case 1: Simple case with unique names and corresponding types
    names_var_1 = [["var1", "var2"], ["var3", "var4"], ["var5", "var6"], ["var7", "var8"]]
    types_var_1 = [["categorical", "continuous"], ["continuous", "categorical"], ["categorical", "continuous"], ["continuous", "categorical"]]
    expected_unique_names_1 = ['var1', 'var2', 'var3', 'var4', 'var5', 'var6', 'var7', 'var8']
    expected_unique_types_1 = ['categorical', 'continuous', 'continuous', 'categorical', 'categorical', 'continuous', 'continuous', 'categorical']
    unique_names_1, unique_types_1 = get_unique_names_and_types(names_var_1, types_var_1)
    assert unique_names_1 == expected_unique_names_1, f"Test Case 1 Failed: {unique_names_1} != {expected_unique_names_1}"
    assert unique_types_1 == expected_unique_types_1, f"Test Case 1 Failed: {unique_types_1} != {expected_unique_types_1}"
    print("Test Case 1 Passed")

    # Test Case 2: Empty input
    names_var_2 = [[], [], [], []]
    types_var_2 = [[], [], [], []]
    expected_unique_names_2 = []
    expected_unique_types_2 = []
    unique_names_2, unique_types_2 = get_unique_names_and_types(names_var_2, types_var_2)
    assert unique_names_2 == expected_unique_names_2, f"Test Case 2 Failed: {unique_names_2} != {expected_unique_names_2}"
    assert unique_types_2 == expected_unique_types_2, f"Test Case 2 Failed: {expected_unique_types_2} != {expected_unique_types_2}"
    print("Test Case 2 Passed")

    # Test Case 3: All variables have the same name
    names_var_3 = [["var1", "var1"], ["var1", "var1"], ["var1", "var1"], ["var1", "var1"]]
    types_var_3 = [["categorical", "categorical"], ["categorical", "categorical"], ["categorical", "categorical"], ["categorical", "categorical"]]
    expected_unique_names_3 = ['var1']
    expected_unique_types_3 = ['categorical']
    unique_names_4, unique_types_3 = get_unique_names_and_types(names_var_3, types_var_3)
    assert unique_names_3 == expected_unique_names_3, f"Test Case 3 Failed: {unique_names_3} != {expected_unique_names_3}"
    assert unique_types_3 == expected_unique_types_3, f"Test Case 3 Failed: {unique_types_3} != {expected_unique_types_3}"
    print("Test Case 3 Passed")


##################################### TEST SG_MASK_GENERATION.PY

def test_create_sg_mask():
    """
    Test the create_sg_mask function to ensure it correctly generates a mask based on missing values.
    """
    nr, nc = 4, 5  
    sim_var = {
        'var1': np.array([[1, 2, np.nan, 4, 5], [1, np.nan, 3, 4, np.nan], [np.nan, 2, 3, np.nan, 5], [1, 2, 3, 4, 5]]),
    }
    auxTI_var = {
        'var2': np.array([[np.nan, 2, 3, 4, 5], [1, 2, np.nan, 4, 5], [1, np.nan, 3, 4, np.nan], [np.nan, 2, 3, 4, 5]]),
    }
    auxSG_var = {
        'var3': np.array([[1, 2, 3, 4, 5], [np.nan, 2, 3, 4, 5], [1, 2, np.nan, 4, 5], [1, np.nan, 3, 4, 5]]),
    }
    condIm_var = {
        'var4': np.array([[1, 2, 3, 4, 5], [1, 2, np.nan, 4, 5], [1, 2, 3, np.nan, 5], [np.nan, 2, 3, 4, 5]]),
    }
    expected_mask = np.array([[0., 1., 1., 1., 1.],
                              [0., 1., 0., 1., 1.],
                              [1., 0., 0., 1., 0.],
                              [0., 0., 1., 1., 1.]])

    result_mask = create_sg_mask(auxTI_var, auxSG_var, nr, nc)
    assert np.array_equal(result_mask, expected_mask), "Test failed: The mask does not match the expected output."
    print("Test passed: The mask matches the expected output.")
    
    return True


def test_merge_masks():
    """
    Test the merge_masks function to ensure it correctly merges two binary masks.
    """
    # Test case 1: 
    mask1 = np.array([[0, 0, 1], [1, 0, 1]])
    mask2 = np.array([[0, 1, 0], [0, 0, 1]])
    expected_result = np.array([[0, 0, 0], [0, 0, 1]])
    result = merge_masks(mask1, mask2)
    assert np.array_equal(result, expected_result), f"Test case 1 failed: {result}"

    # Test case 2:
    mask1 = np.array([[1, 0, 1], [0, 1, 0]])
    mask2 = np.array([[0, 0, 0], [0, 0, 0]])
    expected_result = np.array([[0, 0, 0], [0, 0, 0]])
    result = merge_masks(mask1, mask2)
    assert np.array_equal(result, expected_result), f"Test case 2 failed: {result}"

    # Test case 3: 
    mask1 = np.array([[0, 0, 0], [0, 0, 0]])
    mask2 = np.array([[1, 0, 1], [0, 1, 0]])
    expected_result = np.array([[0, 0, 0], [0, 0, 0]])
    result = merge_masks(mask1, mask2)
    assert np.array_equal(result, expected_result), f"Test case 3 failed: {result}"

    # Test case 4:
    mask1 = np.array([[1, 1, 0], [0, 1, 1]])
    mask2 = np.array([[1, 1, 0], [0, 1, 1]])
    expected_result = np.array([[1, 1, 0], [0, 1, 1]])
    result = merge_masks(mask1, mask2)
    assert np.array_equal(result, expected_result), f"Test case 4 failed: {result}"

    # Test case 5:
    mask1 = np.array([[1, 1], [0, 1]])
    mask2 = np.array([[1, 1, 0], [0, 1, 1]])
    try:
        result = merge_masks(mask1, mask2)
        print("Test case 5 failed: ValueError not raised.")
    except ValueError as e:
        print("Test case 5 passed: ", e)

    print("All test cases passed.")


##################################### TEST TI_GENERATION.PY

def test_gen_ti_frame_circles():
    nc = 3000 
    nr = 1000
    ti_pct_area = 50  # pourcentage de l'aire de la grille à couvrir
    ti_ndisks = 10 
    seed = 852 

    mask = gen_ti_frame_circles(nr, nc, ti_pct_area, ti_ndisks, seed)[0][0]
    plt.figure(figsize=(10, 10))
    plt.imshow(mask, cmap='gray', origin='lower')
    plt.title(f'TI mask generated with {ti_ndisks} disks covering {ti_pct_area}% of a grid of size {nc} x {nr}')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    
    handles = [plt.Line2D([0], [0], color='black', lw=4),
               plt.Line2D([0], [0], color='white', lw=4)]
    plt.legend(handles, ['Hidden (value = 0)', 'Used for Simulation (value = 1)'], loc='upper right', fontsize='medium', frameon=True, shadow=False)
    
    plt.show()


def test_gen_ti_frame_squares():
    nc = 3000
    nr = 1000
    ti_pct_area = 50
    ti_nsquares = 10 
    seed = 854
    
    mask_list, need_to_cut = gen_ti_frame_squares(nr, nc, ti_pct_area, ti_nsquares, seed)
    plt.figure(figsize=(10, 10))
    mask = mask_list[0]
    plt.imshow(mask, cmap='gray', origin = "lower")
    plt.title(f'TI mask generated with {ti_nsquares} squares covering {ti_pct_area}% of a grid of size {nc} x {nr}')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    
    handles = [plt.Line2D([0], [0], color='black', lw=4),
               plt.Line2D([0], [0], color='white', lw=4)]
    plt.legend(handles, ['Hidden (value = 0)', 'Used for Simulation (value = 1)'], loc='upper right', fontsize='medium', frameon=True, shadow=False)
    
    
    plt.show()
  
  
def test_gen_ti_frame_separated_squares(showCoord=True):
    print("\n##################################################################")
    print("\t\tTESTING GEN TI FRAME SEPARATED SQUARES")
    print("##################################################################\n")

    nc = 337
    nr = 529
    ti_pct_area = 90  # pourcentage de l'aire de la grille à couvrir
    ti_nsquares = 5  # nombre de carrés
    seed = 854 
    plot_size = nc
    
    squares, need_to_cut = gen_ti_frame_separatedSquares(nc, nr, ti_pct_area, ti_nsquares, seed)
    
    num_plots = len(squares)
    cols = 5  # Number of columns
    rows = (num_plots + cols - 1) // cols  # Calculate the required number of rows
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 4, rows * 4))
    
    if showCoord:
        for i, square in enumerate(squares):
            print(f"Square {i}: {square} \t")

    
    for i in range(rows * cols):
        ax = axs.flat[i] if rows * cols > 1 else axs
        if i < num_plots:
            square_plot = np.zeros((plot_size, plot_size))
            square = squares[i]
            square_plot = np.where((square == 1), 1, 0)
            ax.imshow(square_plot, cmap='gray', origin='lower')
            ax.set_title(f"Square {i}")
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def test_gen_ti_frame_sg_mask():
    print("\n##################################################################")
    print("\t\tTESTING GEN TI FRAME SINGLE RECTANGLE")
    print("##################################################################\n")
    
    nc, nr = 1000, 1000
    seed = 4
   
    ti_frame_list, need_to_cut, simgrid_mask, cc_sg, rr_sg = gen_ti_frame_sg_mask(nr, nc, pct_ti_sg_overlap = 10, cc_sg = 35, rr_sg = 80, cc_ti = 100, rr_ti = 50,seed=seed)
    ti_frame = ti_frame_list[0]
    
    plt.figure(figsize=(8, 8))
    plt.imshow(ti_frame, cmap='Blues', origin='lower', alpha=0.5)
    plt.imshow(simgrid_mask, cmap='Reds', origin='lower', alpha=0.5)
    plt.title("pct_ti_sg_overlap = 10, cc_sg = 35, rr_sg = 80, cc_ti = 100, rr_ti = 50")
    plt.colorbar(label="Presence")
    plt.axis('on')
    plt.show()
    
    ti_frame_list, need_to_cut, simgrid_mask, cc_sg, rr_sg = gen_ti_frame_sg_mask(nr, nc, pct_ti_sg_overlap = 25, pct_sg = 4, pct_ti = 5, seed=seed)
    ti_frame = ti_frame_list[0]
    
    plt.figure(figsize=(8, 8))
    plt.imshow(ti_frame, cmap='Blues', origin='lower', alpha=0.5, label='ti')
    plt.imshow(simgrid_mask, cmap='Reds', origin='lower', alpha=0.5)
    plt.title("pct_ti_sg_overlap = 25, pct_sg = 4, pct_ti = 5")
    plt.colorbar(label="Presence")
    plt.axis('on')
    plt.show()
    
    ti_frame_list, need_to_cut, simgrid_mask, cc_sg, rr_sg = gen_ti_frame_sg_mask(nr, nc, pct_ti_sg_overlap = 25, cc_sg = 300, rr_sg = 80, pct_ti = 25, seed = seed)
    ti_frame = ti_frame_list[0]
    
    plt.figure(figsize=(8, 8))
    plt.imshow(ti_frame, cmap='Blues', origin='lower', alpha=0.5, label='ti')
    plt.imshow(simgrid_mask, cmap='Reds', origin='lower', alpha=0.5)
    plt.title("pct_ti_sg_overlap = 25, cc_sg = 300, rr_sg = 80, pct_ti = 25")
    plt.colorbar(label="Presence")
    plt.axis('on')
    plt.show()
    
    ti_frame_list, need_to_cut, simgrid_mask, cc_sg, rr_sg = gen_ti_frame_sg_mask(nr, nc, pct_ti_sg_overlap = 25, pct_sg = 10, cc_ti = 100, rr_ti = 50, seed = seed)
    ti_frame = ti_frame_list[0]
    
    plt.figure(figsize=(8, 8))
    plt.imshow(ti_frame, cmap='Blues', origin='lower', alpha=0.5, label='ti')
    plt.imshow(simgrid_mask, cmap='Reds', origin='lower', alpha=0.5)
    plt.title("pct_ti_sg_overlap = 25, pct_sg = 10, cc_ti = 100, rr_ti = 50")
    plt.colorbar(label="Presence")
    plt.axis('on')
    plt.show()
    

def test_build_ti_cd():
    print("\n##################################################################")
    print("\t\t\tTESTING BUILD TI CD")
    print("##################################################################\n")
    
    import geone.imgplot as imgplt
    
    novalue = -9999999
    seed = 854
    csv_file_path = r"C:\Users\00115212\Documents\GeoclassificationMPS\test\data_csv.csv"
    sim_var, auxTI_var, auxSG_var, condIm_var, names_var, types_var = create_variables(csv_file_path)
    sim_var, auxTI_var, auxSG_var, condIm_var = check_variables(sim_var, auxTI_var, auxSG_var, condIm_var, names_var, types_var, novalue=novalue)    
    nr, nc = get_sim_grid_dimensions(auxSG_var)
    simgrid_mask1 = create_sg_mask(auxTI_var, auxSG_var, nr, nc)
    
    print(f"Data dimension : \n \t >> Number of rows : {nr} \n \t >> Number of columns : {nc}")
     
    print("**************************")
    
    ti_frame, need_to_cut, simgrid_mask2, cc_sg, rr_sg = gen_ti_frame_sg_mask(nr, nc, pct_ti_sg_overlap=50, pct_sg=20, pct_ti=65, cc_sg=None, rr_sg=None, cc_ti=None, rr_ti=None, seed=seed)
    simgrid_mask = merge_masks(simgrid_mask1, simgrid_mask2)
    ti_list, cd_list = build_ti_cd(ti_frame, need_to_cut, sim_var, cc_sg, rr_sg, auxTI_var, auxSG_var, names_var, simgrid_mask, condIm_var)

    # Check TI list length
    assert len(ti_list) == len(ti_frame), "TI list length mismatch."
          
    print(f"Number of TI : {len(ti_list)}, number of CD : {len(cd_list)}")
    
    for idx, ti in enumerate(ti_list):
        print(f"TI {idx + 1} shape: {ti.val.shape}")
        
    for idx, cd in enumerate(cd_list):
        print(f"CD {idx + 1} shape: {cd.val.shape}")
    
    for idx, ti in enumerate(ti_list):
        assert isinstance(ti, gn.img.Img), "TI is not of type Img."
        # print(f"Training Image {idx + 1}")
        # imgplt.drawImage2D(ti, iv=0, title=f"TI {idx + 1}", vmin=0)
        # plt.show()

    for idx, cd in enumerate(cd_list):
        assert isinstance(cd, gn.img.Img), "CD is not of type Img."
        # print(f"Conditioning Data {idx + 1}")
        # imgplt.drawImage2D(cd, iv=0, title=f"CD {idx + 1},{ti.varname[0]}")
        # plt.show()
        # imgplt.drawImage2D(cd, iv=1, title=f"CD {idx + 1},{ti.varname[1]}")
        # plt.show()
        # imgplt.drawImage2D(cd, iv=2, title=f"CD {idx + 1},{ti.varname[2]}")
        # plt.show()
        # imgplt.drawImage2D(cd, iv=3, title=f"CD {idx + 1},{ti.varname[3]}")
        # plt.show()
    
    print(">>>>> Test completed successfully with single TI frame.\n**************************")
    
    #ti_frame2, need_to_cut2 = gen_ti_frame_circles(nr, nc, ti_pct_area =87, ti_ndisks = 5, seed = seed)
    #ti_frame2, need_to_cut2 = gen_ti_frame_separatedSquares(nr, nc, 90, 5, seed)
    ti_frame2, need_to_cut2 = gen_ti_frame_squares(nr, nc, 90, 5, seed)
    ti_list2, cd_list2 = build_ti_cd(ti_frame2, need_to_cut2, sim_var, nc, nr, auxTI_var, auxSG_var, names_var, simgrid_mask1, condIm_var)

    assert len(ti_list2) == len(ti_frame2), "TI list length mismatch."

    print(f"Number of TI : {len(ti_list2)}, number of CD : {len(cd_list2)}")
    
    for idx, ti in enumerate(ti_list2):
        print(f"TI {idx + 1} shape: {ti.val.shape}")
        if np.any(ti.val == np.nan):
            print(f"NaN value in TI")
    
    for idx, cd in enumerate(cd_list2):
        print(f"CD {idx + 1} shape: {cd.val.shape}")
        
    # Visualize the Training Images (TIs)
    # for idx, ti in enumerate(ti_list2):
        # assert isinstance(ti, gn.img.Img), "TI is not of type Img."
        # imgplt.drawImage2D(ti, iv=0, categ=True, title=f"TI {idx + 1}, {ti.varname[0]}")
        # plt.show()
        # imgplt.drawImage2D(ti, iv=1, title=f"TI {idx + 1}, {ti.varname[1]}")
        # plt.show()
        # imgplt.drawImage2D(ti, iv=2, title=f"TI {idx + 1}, {ti.varname[2]}")
        # plt.show()
        # imgplt.drawImage2D(ti, iv=3, title=f"TI {idx + 1}, {ti.varname[3]}")
        # plt.show()

    # Visualize the Conditioning Data (CDs)
    # for idx, cd in enumerate(cd_list2):
        # assert isinstance(cd, gn.img.Img), "CD is not of type Img."
        # imgplt.drawImage2D(cd, iv=0, categ=False, title=f"CD {idx + 1},{cd.varname[0]}")
        # plt.show()
        # imgplt.drawImage2D(cd, iv=1, title=f"CD {idx + 1},{ti.varname[1]}")
        # plt.show()
        # imgplt.drawImage2D(cd, iv=2, title=f"CD {idx + 1},{ti.varname[2]}")
        # plt.show()
    
    print(">>>>> Test completed successfully with separated TI frames.\n**************************")

    
def test_gen_n_random_ti_cd():
    print("\n##################################################################")
    print("\t\t\tTESTING GEN N RANDOM TI CD")
    print("##################################################################\n")
    import random
    
    seed = 852
    random.seed(seed)
    n=5
    nc, nr = 500, 500
    sim_var = {
        'var1': np.random.randint(0, 25, size=(nr, nc)),
        'var2': np.random.randint(0, 25, size=(nr, nc))
    }
    auxTI_var = {
        'aux_var1': np.random.randint(0, 25, size=(nr, nc)),
        'aux_var2': np.random.randint(0, 25, size=(nr, nc))
    }
    auxSG_var = {
        'aux_var1': np.random.randint(0, 25, size=(nr, nc)),
        'aux_var2': np.random.randint(0, 25, size=(nr, nc))
    }
    names_var = [['var1', 'var2'], ['aux_var1', 'aux_var2'], ['aux_var1', 'aux_var2'], []]
    types_var = [['categorical','categorical'],['categorical','categorical'],['categorical','categorical'],[]]
    
    condIm_var = {} 
    
    sim_var, auxTI_var, auxSG_var, condIm_var = check_variables(sim_var, auxTI_var, auxSG_var, condIm_var, names_var, types_var, novalue=-9999999)
    
    simgrid_mask = np.ones((nr, nc)) 

    cd_lists, ti_lists, nr, nc, mask = gen_n_random_ti_cd(n=n, nc=nc,bnr=nr,
                                                        sim_var=sim_var,
                                                        auxTI_var=auxTI_var, auxSG_var=auxSG_var,
                                                        names_var=names_var,
                                                        simgrid_mask=simgrid_mask,
                                                        condIm_var=condIm_var,
                                                        method="ReducedTiSg",  
                                                        ti_pct_area=90, ti_nshapes=10,
                                                        pct_ti_sg_overlap=10,
                                                        pct_sg=10,pct_ti=30,
                                                        cc_sg=None,rr_sg=None,
                                                        cc_ti=None,rr_ti=None,
                                                        givenseed=seed)
    for ti_list, cd_list, i in zip(ti_lists, cd_lists, range(1,n+1)):
        print(f"\n\n\n----- Testing the set number {i}: -----")
        for cd, i in zip(cd_list, range(1, len(cd_list)+1)):
            for ti, j in zip(ti_list, range(1, len(ti_list)+1)):
                print(f"\nCD{i}, TI{j}")
                cd_vars = cd.varname
                ti_vars = ti.varname
                common_vars = [var for var in cd_vars if var in ti_vars]
            
                for var in common_vars:
                    print(f"> For the common variable {var}:")
                    cd_index = cd_vars.index(var)
                    ti_index = ti_vars.index(var)
                    
                    cd_values = cd.val[cd_index]
                    ti_values = ti.val[ti_index]
                    print(f">>>> Min CD : {np.nanmin(cd_values)}, Max CD : {np.nanmax(cd_values)}")
                    print(f">>>> Min TI : {np.nanmin(ti_values)}, Max CD : {np.nanmax(ti_values)}")

    print("Test passed!")


##################################### TEST PROPORTIONS.PY

def test_get_bins():
    nbins = 9
    eps = 0.01
    simgrid_mask = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]])
    
    auxTI_var = {
        'var1': np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        'var2': np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    }
    
    auxSG_var = {
        'var1': np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]])
    }
    
    condIm_var = {
        'varc1': np.array([[2, 2, 3], [3, 4, 5], [6, 6, 7]]),
        'varc2': np.array([[0.5, 0.6, 0.7], [0.8, 0.9, 1.0], [1.1, 1.2, 1.3]])
    }

    print("Testing 'reg' binning...")
    bins_aux_reg = get_bins(nbins, auxTI_var, auxSG_var, sim_var, simgrid_mask, eps, bintype='reg')
    print("Bins (regular):", bins_aux_reg)

    print("\nTesting 'pct' binning...")
    bins_aux_pct = get_bins(nbins, auxTI_var, auxSG_var, sim_var, simgrid_mask, eps, bintype='pct')
    print("Bins (percentile):", bins_aux_pct)

    print("\nTesting edge cases...")

    empty_test = get_bins(nbins, {}, {}, {}, simgrid_mask, eps, bintype='reg')
    print("Empty input test:", empty_test)

    auxTI_var_nan = {
        'var1': np.array([[np.nan, 2, 3], [4, np.nan, 6], [7, 8, np.nan]])
    }
    nan_test = get_bins(nbins, auxTI_var_nan, auxSG_var, sim_var, simgrid_mask, eps, bintype='reg')
    print("NaN values test:", nan_test)
    
    small_data_test = get_bins(nbins, {'var1': np.array([1, 2])}, {}, {}, np.array([1,1]), eps, bintype='reg')
    print("Small data test:", small_data_test)


##################################### TESTS UTILS.PY

def test_cartesian_product():
    a = [1, 2]
    b = [3, 4]
    c = [5, 6]
    combinations = cartesian_product(a, b, c)
    print(combinations)
    
    nbins = 8
    values = list(range(1, nbins + 1))
    combinations = cartesian_product(*([values]*3))
    print(combinations)
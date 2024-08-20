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
from matplotlib.colors import *

import matplotlib.pyplot as plt
import os

##################################### TEST INTERFACE.PY

def test_get_simulation_info():
    # Test Case: Check if the function returns all the expected variables
    seed, \
    ti_method, \
    ti_pct_area, ti_shapes, \
    ti_sg_overlap_percentage, pct_sg, pct_ti, cc_sg, rr_sg, cc_ti, rr_ti, \
    nn, dt, ms, numberofmpsrealizations, nthreads, \
    cm, myclrs, n_bin, cmap_name, mycmap, ticmap, \
    shorten, \
    sim_var, auxTI_var, auxSG_var, condIm_var, names_var, types_var, \
    nr, nc = get_simulation_info()

    assert isinstance(seed, int), "Seed should be an integer."
    assert isinstance(ti_method, list) and all(isinstance(m, str) for m in ti_method), "ti_method should be a list of strings."
    assert ti_pct_area is None or isinstance(ti_pct_area, float), "ti_pct_area should be None or a float."
    assert isinstance(ti_shapes, int), "ti_shapes should be an integer."
    assert isinstance(ti_sg_overlap_percentage, int), "ti_sg_overlap_percentage should be an integer."
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
    csv_file_path = r"C:\Users\Axel (Travail)\Documents\ENSG\CET\GeoclassificationMPS\test\data_csv.csv"
    sim_var, auxTI_var, auxSG_var, condIm_var, names_var, types_var = create_variables(csv_file_path)
    nr, nc = get_sim_grid_dimensions(sim_var)
    print(nr,nc)
    
    return True
    
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
    nc = 1000  # nombre de colonnes
    nr = 1000  # nombre de lignes
    ti_pct_area = 50  # pourcentage de l'aire de la grille à couvrir
    ti_ndisks = 10  # nombre de disques
    seed = 852  # graine pour le générateur de nombres aléatoires

    mask = gen_ti_frame_circles(nc, nr, ti_pct_area, ti_ndisks, seed)[0][0]
    plt.figure(figsize=(8, 8))
    plt.imshow(mask, cmap='gray', origin='lower')
    plt.title(f'Binary Mask Generated with {ti_ndisks} Disks')
    plt.xlabel('columns')
    plt.ylabel('rows')
    plt.show()

def test_gen_ti_frame_squares():
    nc = 337  # nombre de colonnes
    nr = 529  # nombre de lignes
    ti_pct_area = 90  # pourcentage de l'aire de la grille à couvrir
    ti_nsquares = 5  # nombre de carrés
    seed = 854  # graine pour le générateur de nombres aléatoires
    
    mask_list, need_to_cut = gen_ti_frame_squares(nc, nr, ti_pct_area, ti_nsquares, seed)
    mask = mask_list[0]
    plt.imshow(mask, cmap='gray')
    plt.show()
    
def test_gen_ti_frame_separated_squares(showCoord=True):
    print("\n##################################################################")
    print("\t\tTESTING GEN TI FRAME SEPARATED SQUARES")
    print("##################################################################\n")

    nc = 337  # nombre de colonnes
    nr = 529 # nombre de lignes
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
            ax.axis('off')  # Turn off unused subplots
    
    plt.tight_layout()
    plt.show()

def test_gen_ti_frame_cd_mask():
    print("\n##################################################################")
    print("\t\tTESTING GEN TI FRAME SINGLE RECTANGLE")
    print("##################################################################\n")
    
    nc, nr = 1000, 1000
    seed = 4
   
    ti_frame_list, need_to_cut, simgrid_mask, cc_sg, rr_sg = gen_ti_frame_cd_mask(nr, nc, ti_sg_overlap_percentage = 10, cc_sg = 35, rr_sg = 80, cc_ti = 100, rr_ti = 50,seed=seed)
    ti_frame = ti_frame_list[0]
    
    plt.figure(figsize=(8, 8))
    plt.imshow(ti_frame, cmap='Blues', origin='lower', alpha=0.5)
    plt.imshow(simgrid_mask, cmap='Reds', origin='lower', alpha=0.5)
    plt.title("ti_sg_overlap_percentage = 10, cc_sg = 35, rr_sg = 80, cc_ti = 100, rr_ti = 50")
    plt.colorbar(label="Presence")
    plt.axis('on')
    plt.show()
    
    ti_frame_list, need_to_cut, simgrid_mask, cc_sg, rr_sg = gen_ti_frame_cd_mask(nr, nc, ti_sg_overlap_percentage = 25, pct_sg = 4, pct_ti = 5, seed=seed)
    ti_frame = ti_frame_list[0]
    
    plt.figure(figsize=(8, 8))
    plt.imshow(ti_frame, cmap='Blues', origin='lower', alpha=0.5, label='ti')
    plt.imshow(simgrid_mask, cmap='Reds', origin='lower', alpha=0.5)
    plt.title("ti_sg_overlap_percentage = 25, pct_sg = 4, pct_ti = 5")
    plt.colorbar(label="Presence")
    plt.axis('on')
    plt.show()
    
    ti_frame_list, need_to_cut, simgrid_mask, cc_sg, rr_sg = gen_ti_frame_cd_mask(nr, nc, ti_sg_overlap_percentage = 25, cc_sg = 300, rr_sg = 80, pct_ti = 25, seed = seed)
    ti_frame = ti_frame_list[0]
    
    plt.figure(figsize=(8, 8))
    plt.imshow(ti_frame, cmap='Blues', origin='lower', alpha=0.5, label='ti')
    plt.imshow(simgrid_mask, cmap='Reds', origin='lower', alpha=0.5)
    plt.title("ti_sg_overlap_percentage = 25, cc_sg = 300, rr_sg = 80, pct_ti = 25")
    plt.colorbar(label="Presence")
    plt.axis('on')
    plt.show()
    
    ti_frame_list, need_to_cut, simgrid_mask, cc_sg, rr_sg = gen_ti_frame_cd_mask(nr, nc, ti_sg_overlap_percentage = 25, pct_sg = 10, cc_ti = 100, rr_ti = 50, seed = seed)
    ti_frame = ti_frame_list[0]
    
    plt.figure(figsize=(8, 8))
    plt.imshow(ti_frame, cmap='Blues', origin='lower', alpha=0.5, label='ti')
    plt.imshow(simgrid_mask, cmap='Reds', origin='lower', alpha=0.5)
    plt.title("ti_sg_overlap_percentage = 25, pct_sg = 10, cc_ti = 100, rr_ti = 50")
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
    csv_file_path = r"C:\Users\Axel (Travail)\Documents\ENSG\CET\GeoclassificationMPS\test\data_csv.csv"
    sim_var, auxTI_var, auxSG_var, condIm_var, names_var, types_var = create_variables(csv_file_path)
    sim_var, auxTI_var, auxSG_var, condIm_var = check_variables(sim_var, auxTI_var, auxSG_var, condIm_var, names_var, types_var, novalue=novalue)    
    nr, nc = get_sim_grid_dimensions(auxSG_var)
    simgrid_mask1 = create_sg_mask(auxTI_var, auxSG_var, nr, nc)
    
    print(f"Data dimension : \n \t >> Number of rows : {nr} \n \t >> Number of columns : {nc}")
     
    print("**************************")
    
    ti_frame, need_to_cut, simgrid_mask2, cc_sg, rr_sg = gen_ti_frame_cd_mask(nr, nc, ti_sg_overlap_percentage=50, pct_sg=20, pct_ti=65, cc_sg=None, rr_sg=None, cc_ti=None, rr_ti=None, seed=seed)
    simgrid_mask = merge_masks(simgrid_mask1, simgrid_mask2)
    ti_list, cd_list = build_ti_cd(ti_frame, need_to_cut, sim_var, cc_sg, rr_sg, auxTI_var, auxSG_var, names_var, simgrid_mask, condIm_var)

    # Check TI list length
    assert len(ti_list) == len(ti_frame), "TI list length mismatch."
          
    print(f"Number of TI : {len(ti_list)}, number of CD : {len(cd_list)}")
    
    for idx, ti in enumerate(ti_list):
        print(f"TI {idx + 1} shape: {ti.val.shape}")
        
    for idx, cd in enumerate(cd_list):
        print(f"CD {idx + 1} shape: {cd.val.shape}")
    
    # Visualize the Training Images (TIs)
    for idx, ti in enumerate(ti_list):
        assert isinstance(ti, gn.img.Img), "TI is not of type Img."
        # print(f"Training Image {idx + 1}")
        # imgplt.drawImage2D(ti, iv=0, title=f"TI {idx + 1}", vmin=0)
        # plt.show()

    # Visualize the Conditioning Data (CDs)
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

                
    # Check TI list length
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
    

    deesse_input = gn.deesseinterface.DeesseInput(
        nx=nc, ny=nr, nz=1,
        sx=1, sy=1, sz=1,
        ox=0, oy=0, oz=0,
        nv=2, varname=["grid_geo","grid_grv"],
        TI=ti_list2,
        #pdfTI = pdf_ti,
        mask = simgrid_mask1,
        dataImage=cd_list2,
        distanceType=['categorical',"continuous"],
        nneighboringNode=2*[24],
        distanceThreshold=2*[0.1],
        maxScanFraction=1*[0.5],
        npostProcessingPathMax=1,
        seed=seed,
        nrealization=1
    )  

    deesse_output = gn.deesseinterface.deesseRun(deesse_input)

    sim = deesse_output['sim']
    
    plt.subplots(1, 4, figsize=(17,10), sharex=True, sharey=True)
    
    gn.imgplot.drawImage2D(sim[0], iv=0, categ=True, title=f'Real #{0} - {deesse_input.varname[0]}')
    
    plt.show()

    
def test_gen_n_random_ti_cd():
    print("\n##################################################################")
    print("\t\t\tTESTING GEN N RANDOM TI CD")
    print("##################################################################\n")
    import random
    
    seed = 852
    random.seed(seed)
    
    nc, nr = 50, 50 
    sim_var = {
        'var1': np.random.randint(0, 5, size=(nr, nc)),
        'var2': np.random.randint(0, 5, size=(nr, nc))
    }
    auxTI_var = {
        'aux_var1': np.random.randint(0, 5, size=(nr, nc)),
        'aux_var2': np.random.randint(0, 5, size=(nr, nc))
    }
    auxSG_var = {
        'aux_var_sg1': np.random.randint(0, 5, size=(nr, nc)),
        'aux_var_sg2': np.random.randint(0, 5, size=(nr, nc))
    }
    names_var = ['var1', 'var2']
    simgrid_mask = np.ones((nr, nc))
    condIm_var = {} 
    
    # "DependentCircles", "DependentSquares", "IndependentSquares", "ReducedTiCd"
    method = "ReducedTiCd" 

    cd_lists, ti_lists = gen_n_random_ti_cd(
        n=20,
        nc=nc, nr=nr, 
        sim_var=sim_var, auxTI_var=auxTI_var, auxSG_var=auxSG_var, 
        names_var=names_var, simgrid_mask=simgrid_mask, 
        condIm_var=condIm_var, 
        method=method,
        ti_pct_area=90, ti_nshapes=10, 
        ti_sg_overlap_percentage=10, 
        pct_sg=10, pct_ti=30, 
        cc_sg=None, rr_sg=None, 
        cc_ti=None, rr_ti=None,
        seed=seed
    )

    for cd_list, ti_list in zip(cd_lists, ti_lists):
        for cd, ti in zip(cd_list, ti_list):
            print(np.unique(ti.val), np.unique(cd.val))

    appendFlags = [np.all(np.isin(np.unique(cd.val), np.unique(ti.val))) for cd in cd_lists[0] for ti in ti_lists[0]]

    # Verify if all conditions are True
    assert all(appendFlags), "The condition for exiting the while loop was not met."

    print("Test passed.")

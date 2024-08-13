# -*- coding:utf-8 -*-
__projet__ = "GeoclassificationMPS"
__nom_fichier__ = "tests"
__author__ = "MENGELLE Axel"
__date__ = "juillet 2024"


from ti_generation import *
from data_treatment import *
from interface import get_simulation_info

import matplotlib.pyplot as plt
import pickle  # For loading pickled data
import os

##################################### TEST INTERFACE.PY

def test_get_simulation_info():
     simulated_var, auxiliary_var, types_var, names_var, nn, dt, ms, numberofmpsrealizations, nthreads, config = get_simulation_info()

def test_check_variables():
    # Test 1: Valid input (should pass)
    sim_var = {
        'var1': np.array([[1.0, 2.0], [3.0, -9999999]]),
        'var2': np.array([[4.0, 5.0], [6.0, 7.0]])
    }
    auxdesc_var = {
        'aux1': np.array([[10.0, 20.0], [30.0, 40.0]])
    }
    auxcond_var = {
        'aux1': np.array([[50, 60], [70, 80]])
    }
    cond_var = {
        'var1': np.array([[1.0, 2.0], [3.0, 4.0]])
    }
    names_var = [['var1', 'var2'], ['aux1'], ['aux1'], ['var1']]
    types_var = [['continuous', 'continuous'], ['continuous'], ['continuous'], ['continuous']]
    try:
        sim_var_out, auxdesc_var_out, auxcond_var_out, cond_var_out = check_variables(sim_var, auxdesc_var, auxcond_var, cond_var, names_var, types_var)
        print("Test 1 passed")
    except Exception as e:
        print(f"Test 1 failed: {e}")

    # Test 2: Type mismatch in simulated variable (should raise TypeError)
    sim_var['var1'] = np.array([['a', 'b'], ['c', 'd']])
    try:
        check_variables(sim_var, auxdesc_var, auxcond_var, cond_var, names_var, types_var)
        print("Test 2 failed: TypeError was expected")
    except TypeError as e:
        print(f"Test 2 passed: {e}")

    # Test 3: Shape inconsistency (should raise ValueError)
    sim_var['var1'] = np.array([[1.0, 2.0]])
    try:
        check_variables(sim_var, auxdesc_var, auxcond_var, cond_var, names_var, types_var)
        print("Test 3 failed: ValueError was expected")
    except ValueError as e:
        print(f"Test 3 passed: {e}")

    # Test 4: Name mismatch in conditioning variables (should raise NameError)
    sim_var = {
        'var1': np.array([[1.0, 2.0], [3.0, -9999999]]),
        'var2': np.array([[4.0, 5.0], [6.0, 7.0]])
    }
    names_var[3] = ['var3']
    cond_var['var3'] = cond_var['var1']
    try:
        check_variables(sim_var, auxdesc_var, auxcond_var, cond_var, names_var, types_var)
        print("Test 4 failed: NameError was expected")
    except NameError as e:
        print(f"Test 4 passed: {e}")

    # Test 5: Missing auxiliary conditioning variable (should raise NameError)
    names_var[3] = ['var1']
    names_var[2] = []
    try:
        check_variables(sim_var, auxdesc_var, auxcond_var, cond_var, names_var, types_var)
        print("Test 5 failed: NameError was expected")
    except NameError as e:
        print(f"Test 5 passed: {e}")

    # Test 6: Valid input with novalue handling (should pass and replace novalue with np.nan)
    names_var = [['var1', 'var2'], ['aux1'], ['aux1'], ['var1']]
    sim_var['var1'] = np.array([[1.0, 2.0], [3.0, -9999999]])
    try:
        sim_var_out, auxdesc_var_out, auxcond_var_out, cond_var_out = check_variables(sim_var, auxdesc_var, auxcond_var, cond_var, names_var, types_var)
        assert np.isnan(sim_var_out['var1'][1, 1]), "Test 6 failed: novalue was not replaced by np.nan"
        print("Test 6 passed")
    except Exception as e:
        print(f"Test 6 failed: {e}")

def test_create_auxiliary_and_simulated_var():
    """
    Test function for create_auxiliary_and_simulated_var to ensure it properly categorizes variables
    and handles errors appropriately.
    """
    # Create a temporary directory to store the test numpy arrays
    temp_dir = "temp_test_dir"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Sample data for the test
    test_data = [
        {'var_name': 'sim_var1', 'categ_conti': 'categ', 'nature': 'sim', 'path': f'{temp_dir}/array_sim1.npy'},
        {'var_name': 'auxdesc_var1', 'categ_conti': 'conti', 'nature': 'auxdesc', 'path': f'{temp_dir}/array_auxdesc1.npy'},
        {'var_name': 'auxcond_var1', 'categ_conti': 'categ', 'nature': 'auxcond', 'path': f'{temp_dir}/array_auxcond1.npy'},
        {'var_name': 'cond_var1', 'categ_conti': 'conti', 'nature': 'cond', 'path': f'{temp_dir}/array_cond1.npy'}
    ]
    
    # Create numpy arrays and save them to the specified paths
    np.save(f'{temp_dir}/array_sim1.npy', np.array([1, 2, 3]))
    np.save(f'{temp_dir}/array_auxdesc1.npy', np.array([4, 5, 6]))
    np.save(f'{temp_dir}/array_auxcond1.npy', np.array([7, 8, 9]))
    np.save(f'{temp_dir}/array_cond1.npy', np.array([10, 11, 12]))
    
    # Create a DataFrame from the sample data and save it to a CSV file
    df = pd.DataFrame(test_data)
    csv_file_path = f'{temp_dir}/test_data.csv'
    df.to_csv(csv_file_path, sep=';', index=False)
    
    # Call the function to test
    sim_var, auxdesc_var, auxcond_var, cond_var, names_var, types_var = create_auxiliary_and_simulated_var(csv_file_path)
    
    # Check if the function outputs the expected results
    assert len(sim_var) == 1, "Expected 1 simulated variable."
    assert len(auxdesc_var) == 1, "Expected 1 auxiliary describing variable."
    assert len(auxcond_var) == 1, "Expected 1 auxiliary conditioning variable."
    assert len(cond_var) == 1, "Expected 1 conditioning variable."
    
    assert np.array_equal(sim_var['sim_var1'], np.array([1, 2, 3])), "Unexpected data in simulated variable 'sim_var1'."
    assert np.array_equal(auxdesc_var['auxdesc_var1'], np.array([4, 5, 6])), "Unexpected data in auxiliary describing variable 'auxdesc_var1'."
    assert np.array_equal(auxcond_var['auxcond_var1'], np.array([7, 8, 9])), "Unexpected data in auxiliary conditioning variable 'auxcond_var1'."
    assert np.array_equal(cond_var['cond_var1'], np.array([10, 11, 12])), "Unexpected data in conditioning variable 'cond_var1'."
    
    expected_names_var = [['sim_var1'], ['auxdesc_var1'], ['auxcond_var1'], ['cond_var1']]
    expected_types_var = [['categ'], ['conti'], ['categ'], ['conti']]
    
    assert names_var == expected_names_var, f"Unexpected names_var: {names_var}"
    assert types_var == expected_types_var, f"Unexpected types_var: {types_var}"
    
    # Clean up temporary files
    os.remove(f'{temp_dir}/array_sim1.npy')
    os.remove(f'{temp_dir}/array_auxdesc1.npy')
    os.remove(f'{temp_dir}/array_auxcond1.npy')
    os.remove(f'{temp_dir}/array_cond1.npy')
    os.remove(csv_file_path)
    os.rmdir(temp_dir)

    print("Successfully passed test_create_auxiliary_and_simulated_var.")

def test_get_sim_grid_dimensions():
    csv_file_path = r"C:\Users\Axel (Travail)\Documents\ENSG\CET\GeoclassificationMPS\test\data_csv.csv"
    simulated_var, auxiliary_var, names_var, types_var = create_auxiliary_and_simulated_var(csv_file_path)
    nr, nc = get_sim_grid_dimensions(simulated_var)
    print(nr,nc)
    
    return True

##################################### TEST TI_GENERATION.PY

def test_gen_ti_frame_circles():
    nc = 100  # nombre de colonnes
    nr = 100  # nombre de lignes
    ti_pct_area = 50  # pourcentage de l'aire de la grille à couvrir
    ti_ndisks = 10  # nombre de disques
    seed = 15  # graine pour le générateur de nombres aléatoires

    mask, need_to_cut = gen_ti_frame_circles(nc, nr, ti_pct_area, ti_ndisks, seed)[0]

    plt.figure(figsize=(8, 8))
    plt.imshow(mask, cmap='gray', origin='lower')
    plt.title(f'Binary Mask Generated with {ti_ndisks} Disks')
    plt.xlabel('columns')
    plt.ylabel('rows')
    plt.show()

def test_gen_ti_frame_squares():
    nc = 337  # nombre de colonnes
    nr = 529  # nombre de lignes
    ti_pct_area = 87  # pourcentage de l'aire de la grille à couvrir
    ti_nsquares = 15  # nombre de carrés
    seed = 852  # graine pour le générateur de nombres aléatoires
    
    mask_list, need_to_cut = gen_ti_frame_squares(nc, nr, ti_pct_area, ti_nsquares, seed)
    mask = mask_list[0]
    plt.imshow(mask, cmap='gray')
    plt.show()
    
def test_gen_ti_frame_separated_squares(showCoord=True):
    print("\n##################################################################")
    print("\t\tTESTING GEN TI FRAME SEPARATED SQUARES")
    print("##################################################################\n")

    nc = 1000  # nombre de colonnes
    nr = 1000  # nombre de lignes
    ti_pct_area = 10  # pourcentage de l'aire de la grille à couvrir
    ti_nsquares = 25  # nombre de carrés
    seed = 15  
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
            for idx in square:
                if 0 <= idx[0] < plot_size and 0 <= idx[1] < plot_size:
                    square_plot[idx[0], idx[1]] = 1
            ax.imshow(square_plot, cmap='gray', origin='lower')
            ax.set_title(f"Square {i}")
        else:
            ax.axis('off')  # Turn off unused subplots
    
    plt.tight_layout()
    plt.show()

def test_gen_ti_frame_single_rectangle():
    print("\n##################################################################")
    print("\t\tTESTING GEN TI FRAME SINGLE RECTANGLE")
    print("##################################################################\n")
    
    nc, nr = 624, 350
    seed = 4
   
    ti_frame_list, need_to_cut, simgrid_mask, cc_sg, rr_sg = gen_ti_frame_single_rectangle(nr, nc, ti_sg_overlap_percentage = 10, cc_sg = 35, rr_sg = 80, cc_ti = 100, rr_ti = 50,seed=seed)
    ti_frame = ti_frame_list[0]
    
    plt.figure(figsize=(8, 8))
    plt.imshow(ti_frame, cmap='Blues', origin='lower', alpha=0.5)
    plt.imshow(simgrid_mask, cmap='Reds', origin='lower', alpha=0.5)
    plt.title("ti_sg_overlap_percentage = 10, cc_sg = 35, rr_sg = 80, cc_ti = 100, rr_ti = 50")
    plt.colorbar(label="Presence")
    plt.axis('on')
    plt.show()
    
    ti_frame_list, need_to_cut, simgrid_mask, cc_sg, rr_sg = gen_ti_frame_single_rectangle(nr, nc, ti_sg_overlap_percentage = 25, pct_sg = 4, pct_ti = 5, seed=seed)
    ti_frame = ti_frame_list[0]
    
    plt.figure(figsize=(8, 8))
    plt.imshow(ti_frame, cmap='Blues', origin='lower', alpha=0.5, label='ti')
    plt.imshow(simgrid_mask, cmap='Reds', origin='lower', alpha=0.5)
    plt.title("ti_sg_overlap_percentage = 25, pct_sg = 4, pct_ti = 5")
    plt.colorbar(label="Presence")
    plt.axis('on')
    plt.show()
    
    ti_frame_list, need_to_cut, simgrid_mask, cc_sg, rr_sg = gen_ti_frame_single_rectangle(nr, nc, ti_sg_overlap_percentage = 25, cc_sg = 300, rr_sg = 80, pct_ti = 25, seed = seed)
    ti_frame = ti_frame_list[0]
    
    plt.figure(figsize=(8, 8))
    plt.imshow(ti_frame, cmap='Blues', origin='lower', alpha=0.5, label='ti')
    plt.imshow(simgrid_mask, cmap='Reds', origin='lower', alpha=0.5)
    plt.title("ti_sg_overlap_percentage = 25, cc_sg = 300, rr_sg = 80, pct_ti = 25")
    plt.colorbar(label="Presence")
    plt.axis('on')
    plt.show()
    
    ti_frame_list, need_to_cut, simgrid_mask, cc_sg, rr_sg = gen_ti_frame_single_rectangle(nr, nc, ti_sg_overlap_percentage = 25, pct_sg = 3, cc_ti = 100, rr_ti = 50, seed = seed)
    ti_frame = ti_frame_list[0]
    
    plt.figure(figsize=(8, 8))
    plt.imshow(ti_frame, cmap='Blues', origin='lower', alpha=0.5, label='ti')
    plt.imshow(simgrid_mask, cmap='Reds', origin='lower', alpha=0.5)
    plt.title("ti_sg_overlap_percentage = 25, pct_sg = 2, cc_ti = 100, rr_ti = 50")
    plt.colorbar(label="Presence")
    plt.axis('on')
    plt.show()
    

def test_build_ti():
    print("\n##################################################################")
    print("\t\t\tTESTING BUILD TI")
    print("##################################################################\n")
    
    import geone.imgplot as imgplt
    
    novalue = -9999999
    seed = 852
    csv_file_path = r"C:\Users\Axel (Travail)\Documents\ENSG\CET\GeoclassificationMPS\test\data_csv.csv"
    simulated_var_dirty, auxiliary_var_dirty, names_var, types_var = create_auxiliary_and_simulated_var(csv_file_path)
    simulated_var, auxiliary_var = check_variables(simulated_var_dirty, auxiliary_var_dirty, names_var, types_var, novalue=novalue)
    nr, nc = get_sim_grid_dimensions(simulated_var)
    print(f"Data dimension : \n \t >> Number of rows : {nr} \n \t >> Number of columns : {nc}")
     
    print("**************************")
    
    ti_frame, need_to_cut, simgrid_mask, cc_sg, rr_sg = gen_ti_frame_single_rectangle(nr, nc, ti_sg_overlap_percentage=10, pct_sg=10, pct_ti=30, cc_sg=None, rr_sg=None, cc_ti=None, rr_ti=None, seed=seed)
    ti_list, cd_list = build_ti(ti_frame, need_to_cut, simulated_var, cc_sg, rr_sg, auxiliary_var, names_var, simgrid_mask)

    # Check TI list length
    assert len(ti_list) == len(ti_frame), "TI list length mismatch."
          
    print(f"Number of TI : {len(ti_list)}, number of CD : {len(cd_list)}")
    
    for idx, ti in enumerate(ti_list):
        print(f"TI {idx + 1} shape: {ti.val.shape}")
        
    for idx, cd in enumerate(cd_list):
        print(f"CD {idx + 1} shape: {cd.val.shape}")
    
    # Visualize the Training Images (TIs)
    # for idx, ti in enumerate(ti_list):
        # assert isinstance(ti, gn.img.Img), "TI is not of type Img."
        # print(f"Training Image {idx + 1}")
        # imgplt.drawImage2D(ti, iv=0, title=f"TI {idx + 1}", vmin=0)
        # plt.show()

    # Visualize the Conditioning Data (CDs)
    # for idx, cd in enumerate(cd_list):
        # assert isinstance(cd, gn.img.Img), "CD is not of type Img."
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
    
    #ti_frame2, need_to_cut2 = gen_ti_frame_circles(nr, nc, ti_pct_area =70, ti_ndisks = 5, seed = seed)
    ti_frame2, need_to_cut2 = gen_ti_frame_separatedSquares(nr, nc, 50, 2, seed)
    #ti_frame2, need_to_cut2 = gen_ti_frame_squares(nr, nc, 87, 15, seed)
    ti_list2, cd_list2 = build_ti(ti_frame2, need_to_cut2, simulated_var, nc, nr, auxiliary_var, names_var)

    # Check TI list length
    assert len(ti_list2) == len(ti_frame2), "TI list length mismatch."

    print(f"Number of TI : {len(ti_list2)}, number of CD : {len(cd_list2)}")
    
    for idx, ti in enumerate(ti_list2):
        print(f"TI {idx + 1} shape: {ti.val.shape}")
        print(ti)
        if np.any(ti.val == np.nan):
            print(f"NaN value in TI")
    
    for idx, cd in enumerate(cd_list2):
        print(f"CD {idx + 1} shape: {cd.val.shape}")
        
    # Visualize the Training Images (TIs)
    for idx, ti in enumerate(ti_list2):
        assert isinstance(ti, gn.img.Img), "TI is not of type Img."
        imgplt.drawImage2D(ti, iv=0, title=f"TI {idx + 1}, {ti.varname[0]}")
        plt.show()
        imgplt.drawImage2D(ti, iv=1, title=f"TI {idx + 1}, {ti.varname[1]}")
        plt.show()
        imgplt.drawImage2D(ti, iv=2, title=f"TI {idx + 1}, {ti.varname[2]}")
        plt.show()
        imgplt.drawImage2D(ti, iv=3, title=f"TI {idx + 1}, {ti.varname[3]}")
        plt.show()

    # Visualize the Conditioning Data (CDs)
    # for idx, cd in enumerate(cd_list2):
        # assert isinstance(cd, gn.img.Img), "CD is not of type Img."
        # print(f"Conditioning Data {idx + 1}")
        # imgplt.drawImage2D(cd, iv=0, title=f"CD {idx + 1},{ti.varname[0]}")
        # plt.show()
        # imgplt.drawImage2D(cd, iv=1, title=f"CD {idx + 1},{ti.varname[1]}")
        # plt.show()
        # imgplt.drawImage2D(cd, iv=2, title=f"CD {idx + 1},{ti.varname[2]}")
        # plt.show()
        # imgplt.drawImage2D(cd, iv=3, title=f"CD {idx + 1},{ti.varname[3]}")
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
        nv=4, varname=["grid_geo","grid_grv","gris_lmp","grid_mag"],
        TI=ti_list2,
        #pdfTI = pdf_ti,
        dataImage=cd_list2,
        distanceType=['categorical',"continuous","continuous","continuous"],
        nneighboringNode=4*[12*4],
        distanceThreshold=4*[0.4],
        maxScanFraction=2*[1],
        npostProcessingPathMax=1,
        seed=seed,
        nrealization=1
    )  

    deesse_output = gn.deesseinterface.deesseRun(deesse_input)

    sim = deesse_output['sim']
    
    plt.subplots(1, 4, figsize=(17,10), sharex=True, sharey=True)
    
    gn.imgplot.drawImage2D(sim[0], iv=0, categ=True, title=f'Real #{0} - {deesse_input.varname[0]}')
    
    plt.show()
    
    gn.imgplot.drawImage2D(sim[0], iv=1, categ=True, title=f'Real #{0} - {deesse_input.varname[1]}')
    
    plt.show()
    

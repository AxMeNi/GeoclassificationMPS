# -*- coding:utf-8 -*-
__projet__ = "GeoclassificationMPS"
__nom_fichier__ = "launcher"
__author__ = "MENGELLE Axel"
__date__ = "juillet 2024"

"""
Script for processing and visualization of geospatial data.

This script imports necessary modules and initializes parameters for data processing
and visualization. It also loads pre-processed data from a pickle file.

Author: Guillaume Pirot
Date: Fri Jul 28 11:12:36 2023
"""
from simulation_functions_ter import *
from ti_generation import *
import matplotlib.pyplot as plt


#### COLORS PARAMETERS
cm = plt.get_cmap('tab20')
defaultclrs = np.asarray(cm.colors)[[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11], :]
n_bin = 11
cmap_name = 'my_tab20'
defaultcmap = LinearSegmentedColormap.from_list(cmap_name, defaultclrs, N=n_bin)
defaultticmap = LinearSegmentedColormap.from_list('ticmap', np.vstack(([0, 0, 0], defaultclrs)), N=n_bin + 1)




def launcher(simulated_var, 
        auxiliary_var, 
        var_names, 
        var_types, 
        ti_pct_area, 
        ti_ndisks, 
        ti_realid, 
        mps_nreal, 
        nthreads, 
        geolcd=True, 
        timesleep=0, 
        verb=False,
        addtitle='',
        seed=0,
        myclrs = defaultclrs,
        mycmap = defaultcmap,
        ticmap = defaultticmap
        ):
    """
    Main function for generating and analyzing training images (TI) in geostatistics.

    Parameters:
    ti_pct_area (float): Percentage of the area to be used for TI.
    ti_ndisks (int): Number of disks to be used for TI.
    ti_realid (int): Realization ID for TI.
    mps_nreal (int): Number of multiple-point statistics realizations.
    nthreads (int): Number of threads to use.
    geolcd (bool): Use geological codes (default is True).
    xycv (bool): Use cross-validation on x and y coordinates (default is False).
    timesleep (int): Time to sleep before starting (default is 0).
    verb (bool): Verbose mode, if True, will plot intermediate results (default is False).
    addtitle (str): Additional title for plots (default is empty).

    Returns:
    None
    """
    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)') + " - INIT")
    time.sleep(timesleep)
    
    create_directories()
    
    nx,ny,nsimvar,nauxvar = get_data_info(simulated_var, auxiliary_var, var_names, var_types)
    
    # GENERATE TI MASK
    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)') + " - GENERATE MASK")
    grid_msk = gen_ti_mask_circles(nx, ny, ti_pct_area, ti_ndisks, seed + ti_realid)

    # PLOT TI MASK
    if verb:
        print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)') + " - PLOT MASK")
        plt.figure(dpi=300), plt.title('TI mask')
        plt.imshow(grid_msk, origin='lower', interpolation='none')
        plt.show()

    # BUILD TI
    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)') + " - BUILD TI")
    geocodes, ngeocodes, tiMissingGeol, cond_data = build_ti(grid_msk, ti_ndisks, ti_pct_area, ti_realid, geolcd)

    # COMPUTE TI INDICATORS
    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)') + " - COMPUTE TI INDICATORS")

    prop_ti = np.zeros(ngeocodes)
    prop_ref = np.zeros(ngeocodes)
    for i in range(ngeocodes):
        prop_ref[i] = np.sum(grid_geo == geocodes[i]) / np.prod(grid_msk.shape) * 100
        prop_ti[i] = np.sum((grid_geo * grid_msk) == geocodes[i]) / np.sum(grid_msk) * 100

    stats_check = pd.DataFrame(columns=['geocodes', 'prop_ref', 'prop_ti'])
    stats_check['geocodes'] = geocodes
    stats_check['prop_ref'] = prop_ref
    stats_check['prop_ti'] = prop_ti
    print(stats_check)
    print('TI coverage: %.1f%% of the total area.' % (np.sum(grid_msk) / np.prod(grid_msk.shape) * 100))

    # COUNT BASED ON REGULARLY SPACED BINS
    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)') + " - COUNT BASED ON REGULARLY SPACED BINS")
    vec_mag, vec_grv, vec_lmp = get_vec_bins(grid_msk, bintype='reg')
    class_hist_count_joint_dist = count_joint_dist(grid_mag[grid_msk == 1].flatten(),
                                                   grid_grv[grid_msk == 1].flatten(),
                                                   grid_lmp[grid_msk == 1].flatten(),
                                                   grid_geo[grid_msk == 1].flatten(),
                                                   vec_mag, vec_grv, vec_lmp, geocodes)

    [class_hist_count_marg_mag,
     class_hist_count_marg_grv,
     class_hist_count_marg_lmp,
     class_hist_count_joint_mag_grv] = count_joint_marginals(class_hist_count_joint_dist)
    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)') + " - COUNT BASED ON REGULARLY SPACED BINS - DONE")

    # COUNT BASED ON PERCENTILES
    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)') + " - COUNT BASED ON PERCENTILES")
    pctile_mag, pctile_grv, pctile_lmp = get_vec_bins(grid_msk, bintype='pct')
    class_hist_count_pct_joint_dist = count_joint_dist(grid_mag[grid_msk == 1].flatten(),
                                                       grid_grv[grid_msk == 1].flatten(),
                                                       grid_lmp[grid_msk == 1].flatten(),
                                                       grid_geo[grid_msk == 1].flatten(),
                                                       pctile_mag, pctile_grv, pctile_lmp, geocodes)

    [class_hist_count_pct_marg_mag,
     class_hist_count_pct_marg_grv,
     class_hist_count_pct_marg_lmp,
     class_hist_count_pct_joint_mag_grv] = count_joint_marginals(class_hist_count_pct_joint_dist)
    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)') + " - COUNT BASED ON PERCENTILES - DONE")

    # GET PROPORTIONS AND TOTAL
    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)') + " - GET PROPORTIONS AND TOTAL")
    class_hist_total_joint_dist, class_hist_prop_joint_dist = get_prop(class_hist_count_joint_dist)
    class_hist_total_marg_mag, class_hist_prop_marg_mag = get_prop(class_hist_count_marg_mag)
    class_hist_total_marg_grv, class_hist_prop_marg_grv = get_prop(class_hist_count_marg_grv)
    class_hist_total_marg_lmp, class_hist_prop_marg_lmp = get_prop(class_hist_count_marg_lmp)
    class_hist_total_joint_mag_grv, class_hist_prop_joint_mag_grv = get_prop(class_hist_count_joint_mag_grv)

    class_hist_total_pct_joint_dist, class_hist_prop_pct_joint_dist = get_prop(class_hist_count_pct_joint_dist)
    class_hist_total_pct_marg_mag, class_hist_prop_pct_marg_mag = get_prop(class_hist_count_pct_marg_mag)
    class_hist_total_pct_marg_grv, class_hist_prop_pct_marg_grv = get_prop(class_hist_count_pct_marg_grv)
    class_hist_total_pct_marg_lmp, class_hist_prop_pct_marg_lmp = get_prop(class_hist_count_pct_marg_lmp)
    class_hist_total_pct_joint_mag_grv, class_hist_prop_pct_joint_mag_grv = get_prop(class_hist_count_pct_joint_mag_grv)
    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)') + " - GET PROPORTIONS AND TOTAL - DONE")

    # PLOT MARGINALS IF VERBOSE
    if verb:
        plot_marginals(class_hist_count_marg_mag, class_hist_count_marg_grv, class_hist_count_marg_lmp,
                       'COUNT BASED ON REGULARLY SPACED BINS')
        plot_marginals(class_hist_count_pct_marg_mag, class_hist_count_pct_marg_grv, class_hist_count_pct_marg_lmp,
                       'COUNT BASED ON PERCENTILES')

    # COMPUTE ENTROPY
    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)') + " - ENTROPY BASED ON REGULARLY SPACED BINS")
    shannon_entropy_joint_dist = shannon_entropy(class_hist_prop_joint_dist)
    shannon_entropy_marg_mag = shannon_entropy(class_hist_prop_marg_mag)
    shannon_entropy_marg_grv = shannon_entropy(class_hist_prop_marg_grv)
    shannon_entropy_marg_lmp = shannon_entropy(class_hist_prop_marg_lmp)
    shannon_entropy_joint_mag_grv = shannon_entropy(class_hist_prop_joint_mag_grv)

    shannon_entropy_marg = np.vstack((shannon_entropy_marg_mag, shannon_entropy_marg_grv, shannon_entropy_marg_lmp))
    shannon_entropy_labl = ['mag', 'grv', '1vd']

    if verb:
        plot_shannon_entropy_marginals(shannon_entropy_marg, shannon_entropy_labl, shannon_entropy_joint_mag_grv)
    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)') + " - ENTROPY BASED ON REGULARLY SPACED BINS - DONE")

    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)') + " - ENTROPY BASED ON PERCENTILES")
    shannon_entropy_pct_joint_dist = shannon_entropy(class_hist_prop_pct_joint_dist)
    shannon_entropy_pct_marg_mag = shannon_entropy(class_hist_prop_pct_marg_mag)
    shannon_entropy_pct_marg_grv = shannon_entropy(class_hist_prop_pct_marg_grv)
    shannon_entropy_pct_marg_lmp = shannon_entropy(class_hist_prop_pct_marg_lmp)
    shannon_entropy_pct_joint_mag_grv = shannon_entropy(class_hist_prop_pct_joint_mag_grv)

    shannon_entropy_pct_marg = np.vstack(
        (shannon_entropy_pct_marg_mag, shannon_entropy_pct_marg_grv, shannon_entropy_pct_marg_lmp))
    shannon_entropy_labl = ['mag', 'grv', '1vd']

    if verb:
        plot_shannon_entropy_marginals(shannon_entropy_pct_marg, shannon_entropy_labl,
                                       shannon_entropy_pct_joint_mag_grv)
    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)') + " - ENTROPY BASED ON PERCENTILES - DONE")

    # COMPUTE HISTOGRAM DISSIMILARITY BETWEEN CLASSES FOR MARGINALS AND JOINT DISTRIBUTIONS
    print((datetime.now()).strftime(
        '%d-%b-%Y (%H:%M:%S)') + " - COMPUTE JENSEN SHANNON DIVERGENCE BETWEEEN DISTRIBUTIONS")
    [jsdist_joint_dist,
     jsdist_marg_mag,
     jsdist_marg_grv,
     jsdist_marg_lmp,
     jsdist_joint_mag_grv] = get_jsdist_all(class_hist_prop_joint_dist,
                                            class_hist_prop_marg_mag,
                                            class_hist_prop_marg_grv,
                                            class_hist_prop_marg_lmp,
                                            class_hist_prop_joint_mag_grv)
    print((datetime.now()).strftime(
        '%d-%b-%Y (%H:%M:%S)') + " - COMPUTE JENSEN SHANNON DIVERGENCE BETWEEEN DISTRIBUTIONS - DONE")

    if verb:
        # plot_jsdivmx_mds_hist(geocodes,jsdist_mx,prefix,class_hist_count=None)
        plot_jsdivmx_mds_hist(geocodes, jsdist_marg_mag, 'Mag', class_hist_count_marg_mag)
        plot_jsdivmx_mds_hist(geocodes, jsdist_marg_grv, 'Grv', class_hist_count_marg_grv)
        plot_jsdivmx_mds_hist(geocodes, jsdist_marg_lmp, '1vd', class_hist_count_marg_lmp)
        plot_jsdivmx_mds_hist(geocodes, jsdist_joint_dist, 'Joint-all')  # ,class_hist_count_joint_dist
        plot_jsdivmx_mds_hist(geocodes, jsdist_joint_mag_grv, 'Mag-Grv')  # ,class_hist_count_joint_mag_grv

    # TAKING INTO ACCOUNT TOPOLOGICAL DIFFERENCES BETWEEN CLASSES
    adj_mx = topological_adjacency(grid_geo, geocodes)
    tpl_dist = np.zeros((ngeocodes, ngeocodes))
    for i in range(ngeocodes):
        for j in range(i):
            ix = np.where((np.arange(ngeocodes) != i) & (np.arange(ngeocodes) != j))
            tpl_dist[i, j] = tpl_dist[j, i] = np.sqrt(np.sum((adj_mx[i, ix] - adj_mx[j, ix]) ** 2)) / np.sqrt(
                ngeocodes - 2)
    if verb:
        plot_jsdivmx_mds_hist(geocodes, tpl_dist, 'Topo only')
        # COMBINING TOPOLOGICAL AND JOINT DISTRIBUTION DIFFERENCES BETWEEN CLASSES
        wt = 0.25
        jsd_tpl = (1 - wt) * jsdist_joint_dist / jsdist_joint_dist.max() + wt * tpl_dist
        plot_jsdivmx_mds_hist(geocodes, jsd_tpl, 'Topology &')  # ,class_hist_count_joint_mag_grv
        # dendrogram plot
        y = jsd_tpl[np.triu_indices(ngeocodes, 1)]
        lnk = linkage(y, 'single')
        fig = plt.figure(figsize=(15, 5), dpi=300)
        dn = dendrogram(lnk, labels=geocodes)
        plt.show()

    # %% RUN DEESSE
    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)') + " - RUN DEESSE")
    deesse_output = run_deesse(tiMissingGeol, mps_nreal, nneighboringNode,
                               distanceThreshold, maxScanFraction, myseed + ti_realid,
                               nthreads, geolcd, cond_data)

    # %% COMPUTE REAL INDICATORS
    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)') + " - COMPUTE REAL INDICATORS")

    # Retrieve the results
    sim = deesse_output['sim']

    # Do some statistics on the realizations
    # ... gather all the realizations into one image
    all_sim = gn.img.gatherImages(sim)  # all_sim is one image with nreal variables
    # ... compute the pixel-wise proportion for the given categories
    all_sim_stats = gn.img.imageCategProp(all_sim, geocodes)
    realizations = np.ones((ny, nx, mps_nreal)) * np.nan
    for i in range(mps_nreal):
        ix = i * tiMissingGeol.nv
        realizations[:, :, i] = all_sim.val[ix, 0, :, :]

    geocode_entropy = entropy(realizations)

    # Local performance - errors
    error = np.zeros((ny, nx, ngeocodes + 1))
    for i in range(ngeocodes):
        tmp1 = realizations == geocodes[i]
        tmp2 = grid_geo == geocodes[i]  # np.tile(np.reshape(grid_geo==codelist[i],(ny,nx,1)), (1,1,nreal))
        error[:, :, i] = np.mean(tmp1, axis=2) - tmp2
    # error across all lithocode
    for r in range(mps_nreal):
        error[:, :, -1] += 1 * ((grid_geo - realizations[:, :, r]) != 0) / mps_nreal

    # confusion matrix, TP/FP/TN/FN and related indicators
    reference = grid_geo
    mask = 1 - grid_msk
    confusion_matrix, classes, training_size, testing_size = get_confusion_matrix(realizations, reference, mask)
    tfpn = get_true_false_pos_neg(confusion_matrix, classes, training_size, testing_size)
    classification_performance(tfpn)
    confusion_matrix_th, classes, thresholds = get_confusion_matrix_th(realizations, reference, mask, nthresholds_tpfn)
    TPR, FPR = get_tpr_fpr(confusion_matrix_th)

    if verb:
        # PLOT REAL
        plot_real_and_ref(realizations, grid_geo, grid_msk, addtitle=addtitle)
        # PLOT ENTROPY AND CONFUSION MATRIX
        plot_entropy_and_confusionmx(geocode_entropy, confusion_matrix, mps_nreal)

    # EXPORT / SAVE
    datafilepath = path2real + 'data' + suffix + '-ndisks-' + str(ti_ndisks) + '-areapct-' + str(
        ti_pct_area) + '-r-' + str(ti_realid) + '.pickle'
    with open(datafilepath, 'wb') as f:
        pickle.dump([realizations, grid_msk
                     ], f)

    datafilepath = path2ind + 'data' + suffix + '-ndisks-' + str(ti_ndisks) + '-areapct-' + str(
        ti_pct_area) + '-r-' + str(ti_realid) + '.pickle'
    with open(datafilepath, 'wb') as f:
        pickle.dump([class_hist_count_marg_mag, class_hist_count_marg_grv, class_hist_count_marg_lmp,
                     class_hist_count_pct_marg_mag, class_hist_count_pct_marg_grv, class_hist_count_pct_marg_lmp,
                     shannon_entropy_joint_dist, shannon_entropy_marg, shannon_entropy_joint_mag_grv,
                     shannon_entropy_pct_joint_dist, shannon_entropy_pct_marg, shannon_entropy_pct_joint_mag_grv,
                     jsdist_joint_dist, jsdist_marg_mag, jsdist_marg_grv, jsdist_marg_lmp, jsdist_joint_mag_grv,
                     tpl_dist,
                     geocode_entropy, error, confusion_matrix, mps_nreal, tfpn, TPR, FPR
                     ], f)

    # FINISHED
    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)') + " - FINISHED")
    return

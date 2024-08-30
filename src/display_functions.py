# -*- coding:utf-8 -*-
__projet__ = "GeoclassificationMPS"
__nom_fichier__ = "display_functions"
__author__ = "MENGELLE Axel"
__date__ = "aout 2024"



def plot_marginals(marg_mag, marg_grv, marg_lmp, suptitle):
    """
    Plot marginal histograms for magnetism, gravity, and lmp variables.

    Parameters:
    ----------
    marg_mag : ndarray
        Marginal histogram for magnetism.
    marg_grv : ndarray
        Marginal histogram for gravity.
    marg_lmp : ndarray
        Marginal histogram for lmp.
    suptitle : str
        Title for the plot.
    """
    nbins = marg_mag.shape[0]
    plt.subplots(1, 3, figsize=(12, 5), dpi=300)

    plt.subplot(1, 3, 1), plt.title('Marginal hist count mag'), plt.imshow(marg_mag, origin='lower',
                                                                           cmap='Blues'), plt.xlabel(
        'lithocode'), plt.ylabel('bin number'), plt.ylim((0.5, nbins - 0.5))
    plt.subplot(1, 3, 2), plt.title('Marginal hist count grv'), plt.imshow(marg_grv, origin='lower',
                                                                           cmap='Blues'), plt.xlabel(
        'lithocode'), plt.ylabel('bin number'), plt.ylim((0.5, nbins - 0.5))
    plt.subplot(1, 3, 3), plt.title('Marginal hist count 1vd'), plt.imshow(marg_lmp, origin='lower',
                                                                           cmap='Blues'), plt.xlabel(
        'lithocode'), plt.ylabel('bin number'), plt.ylim((0.5, nbins - 0.5))

    plt.suptitle(suptitle)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.show()


def plot_shannon_entropy_marginals(shannon_entropy_marg, shannon_entropy_labl, shannon_entropy_joint):
    """
    Plot Shannon's entropy for marginals and joint distributions.

    Parameters:
    ----------
    shannon_entropy_marg : ndarray
        Shannon's entropy for marginals.
    shannon_entropy_labl : list
        Labels for Shannon's entropy marginals.
    shannon_entropy_joint : ndarray
        Shannon's entropy for joint distributions.
    """
    nbins = shannon_entropy_marg.shape[1]
    bins = np.linspace(1, nbins, nbins)

    plt.subplots(1, 2, figsize=(8, 3.5), dpi=300)

    plt.subplot(1, 2, 1), plt.title("Marginal Shannon's entropy")
    plt.plot(bins, shannon_entropy_marg.T)
    plt.legend(shannon_entropy_labl, loc='best')
    plt.xlabel('marginal distribution bins'), plt.ylabel("Shannon's entropy"), plt.ylim((0, 1))

    ax = plt.subplot(1, 2, 2)
    plt.title("Joint Shannon's entropy")
    im = plt.imshow(shannon_entropy_joint, origin='lower', extent=[0.5, nbins - 0.5, 0.5, nbins - 0.5], cmap='Blues',
                    vmin=0, vmax=1)
    plt.xlabel('grv bins'), plt.ylabel('mag bins')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    plt.subplots_adjust(bottom=0.1, right=1.0, top=0.9)
    plt.show()
    

def plot_jsdivmx_mds_hist(geocodes, jsdist_mx, prefix, class_hist_count=None):
    """
    Plot Jensen-Shannon divergence matrix using Multi-Dimensional Scaling (MDS).

    Parameters:
    ----------
    geocodes : ndarray
        Geological codes.
    jsdist_mx : ndarray
        Jensen-Shannon divergence matrix.
    prefix : str
        Prefix for plot title.
    class_hist_count : ndarray, optional
        Histogram counts for classification.

    Returns:
    -------
    None
    """
    ngeocodes = len(geocodes)
    mdspos_jsdist_mx = mds.fit(jsdist_mx).embedding_

    bx = (mdspos_jsdist_mx[:, 0].max() - mdspos_jsdist_mx[:, 0].min()) / 100
    by = (mdspos_jsdist_mx[:, 1].max() - mdspos_jsdist_mx[:, 1].min()) / 100

    plt.subplots(1, 3, figsize=(13, 3.5), dpi=300)

    plt.subplot(1, 3, 1), plt.title(prefix + ' - Jensen Shannon Divergence'), plt.xlabel('lithocodes'), plt.ylabel(
        'lithocodes')
    plt.imshow(jsdist_mx, origin='upper', cmap='Reds', extent=[0.5, ngeocodes + 0.5, ngeocodes + 0.5, 0.5])

    ax = plt.subplot(1, 3, 2)
    plt.title(prefix + ' - Multi-Dimensional-Scaling'), plt.xlabel('MDS component 1'), plt.ylabel('MDS component 2')
    im = plt.scatter(mdspos_jsdist_mx[:, 0], mdspos_jsdist_mx[:, 1], c=geocodes, cmap=mycmap,
                     s=100, label='lithocode hist', marker='+', vmin=0.5, vmax=11.5)

    for i in range(ngeocodes):
        plt.text(mdspos_jsdist_mx[i, 0] + bx, mdspos_jsdist_mx[i, 1] + by, str(geocodes[i]), size=12,
                 color=myclrs[i, :])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    plt.subplot(1, 3, 3)
    if class_hist_count is None:
        plt.axis('off')
    elif len(class_hist_count.shape) == 2:
        plt.title(prefix + ' - Histograms'), plt.xlabel('lithocodes'), plt.ylabel('bins')
        plt.imshow(class_hist_count, origin='lower', cmap='Blues', extent=[0.5, ngeocodes + 0.5, 0.5, nbins + 0.5])
    else:
        plt.axis('off')

    plt.show()


def plot_real_and_ref(realizations, reference, mask, nrealmax=3, addtitle=''):
    """
    Trace les réalisations simulées et la référence.

    Paramètres:
    realizations (np.array): Réalisations simulées.
    reference (np.array): Référence pour la classification.
    mask (np.array): Masque définissant les zones d'intérêt.
    nrealmax (int, optionnel): Nombre maximum de réalisations à tracer. Par défaut, 3.
    addtitle (str, optionnel): Titre additionnel à ajouter aux sous-titres des graphes.

    Retour:
    None: Affiche les graphes.
    """
    shrinkfactor = 0.55
    plot_msk2 = 1 - mask
    nr2plot = np.min([nrealmax, 3])
    ncol = nr2plot + 1
    plt.figure(figsize=(5 * ncol, 10), dpi=300)

    for i in range(nr2plot):
        plt.subplot(1, ncol, i + 1)
        plt.title('Real #' + str(i) + ' ' + addtitle)
        tmp = realizations[:, :, i]
        im = plt.imshow(tmp, origin='lower', cmap=mycmap, interpolation='none', vmin=0.5, vmax=11.5)
        plt.imshow(plot_msk2, origin='lower', cmap='gray', alpha=0.3)
        plt.axis('off')
        plt.colorbar(im, shrink=shrinkfactor)

    plt.subplot(1, ncol, nr2plot + 1)
    plt.title('Reference')
    im = plt.imshow(reference, origin='lower', cmap=mycmap, interpolation='none', vmin=0.5, vmax=11.5)
    plt.imshow(plot_msk2, origin='lower', cmap='gray', alpha=0.3)
    plt.axis('off')
    plt.colorbar(im, shrink=shrinkfactor)
    plt.show()


def plot_entropy_and_confusionmx(geocode_entropy, confusion_matrix, mps_nreal):
    """
    Trace l'entropie des codes géologiques et la matrice de confusion.

    Paramètres:
    geocode_entropy (np.array): Entropie des codes géologiques.
    confusion_matrix (np.array): Matrice de confusion des lithocodes.
    mps_nreal (int): Nombre de réalisations MPS.

    Retour:
    None: Affiche les graphes.
    """
    plt.subplots(1, 2, figsize=(12, 8), dpi=300)
    shrinkfactor = 0.8

    plt.subplot(1, 2, 1)
    im = plt.imshow(geocode_entropy, origin='lower', extent=grid_ext, cmap='Reds')
    plt.xlabel('Easting')
    plt.ylabel('Northing')
    plt.title('Geological code Entropy over ' + str(mps_nreal) + ' realizations')
    plt.colorbar(im, shrink=shrinkfactor)

    plt.subplot(1, 2, 2)
    im = plt.imshow(np.log10(confusion_matrix), interpolation='none', cmap='Greens')
    plt.title('Lithocode log$_{10}$ confusion matrix')
    plt.xlabel('predicted lithocode')
    plt.ylabel('reference lithocode')
    cb = plt.colorbar(im, shrink=shrinkfactor)
    cb.set_label('log$_{10}$ count')
    plt.show()
    return

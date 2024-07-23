# -*- coding:utf-8 -*-
__projet__ = "GeoclassificationMPS"
__nom_fichier__ = "ti_generation"
__author__ = "MENGELLE Axel"
__date__ = "juillet 2024"

import numpy as np  # NumPy for numerical operations
from skimage.draw import disk  # For drawing shapes
from skimage.draw import rectangle
from skimage.morphology import binary_dilation  # For morphological operations
from time import *

def gen_ti_mask_circles(nx, ny, ti_pct_area, ti_ndisks, seed):
    """
    Generate a binary mask representing multiple disks within a grid.

    Parameters:
    ----------
    nx : int
        Number of columns in the grid.
    ny : int
        Number of rows in the grid.
    ti_pct_area : float
        Percentage of the grid area to cover with disks.
    ti_ndisks : int
        Number of disks to generate.
    seed : int
        Seed for the random number generator.

    Returns:
    -------
    mask : ndarray
        Binary mask with 1s indicating disk positions within the grid.
    """
    rng = np.random.default_rng(seed=seed)
    rndy = rng.integers(low=0, high=ny, size=ti_ndisks)
    rndx = rng.integers(low=0, high=nx, size=ti_ndisks)
    radius = np.floor(np.sqrt((nx * ny * ti_pct_area / 100 / ti_ndisks) / np.pi))
    mask = np.zeros((ny, nx))
    for i in range(ti_ndisks):
        rr, cc = disk((rndy[i], rndx[i]), radius, shape=(ny, nx))
        mask[rr, cc] = 1
    check_pct = np.sum(mask.flatten()) / (nx * ny) * 100
    while check_pct < ti_pct_area:
        mask = binary_dilation(mask)
        check_pct = np.sum(mask.flatten()) / (nx * ny) * 100
    return mask
    
def gen_ti_mask_squares(nx, ny, ti_pct_area, ti_nsquares, seed):
    """
    Generate a binary mask representing multiple squares within a grid.

    Parameters:
    ----------
    nx : int
        Number of columns in the grid.
    ny : int
        Number of rows in the grid.
    ti_pct_area : float
        Percentage of the grid area to cover with squares.
    ti_nsquares : int
        Number of squares to generate.
    seed : int
        Seed for the random number generator.

    Returns:
    -------
    mask : ndarray
        Binary mask with 1s indicating square positions within the grid.
    """
    rng = np.random.default_rng(seed=seed)
    rndy = rng.integers(low=0, high=ny, size=ti_nsquares)
    rndx = rng.integers(low=0, high=nx, size=ti_nsquares)
    side_length = int(np.sqrt((nx * ny * ti_pct_area / 100) / ti_nsquares))
    mask = np.zeros((ny, nx))
    for i in range(ti_nsquares):
        rr_start = max(0, rndy[i] - side_length // 2)
        cc_start = max(0, rndx[i] - side_length // 2)
        rr_end = min(ny - 1, rr_start + side_length - 1)
        cc_end = min(nx - 1, cc_start + side_length - 1)
        rr, cc = rectangle(start=(rr_start, cc_start), end=(rr_end, cc_end))
        mask[rr, cc] = 1
    check_pct = np.sum(mask.flatten()) / (nx * ny) * 100
    while check_pct < ti_pct_area:
        mask = np.pad(mask, 1, mode='constant', constant_values=0)
        mask = binary_dilation(mask)
        mask = mask[1:-1, 1:-1]
        check_pct = np.sum(mask.flatten()) / (nx * ny) * 100
    return mask
    
def gen_ti_mask_separatedSquares(nx, ny, ti_pct_area, ti_nsquares, seed):
    """
    Generate a binary mask representing multiple squares within a grid and return each square as a separate array.

    Parameters:
    ----------
    nx : int
        Number of columns in the grid.
    ny : int
        Number of rows in the grid.
    ti_pct_area : float
        Percentage of the grid area to cover with squares.
    ti_nsquares : int
        Number of squares to generate.
    seed : int
        Seed for the random number generator.

    Returns:
    -------
    squares : list of ndarrays
        List of arrays with 1s indicating square positions within the grid.
    """
    rng = np.random.default_rng(seed=seed)
    rndy = rng.integers(low=0, high=ny, size=ti_nsquares)
    rndx = rng.integers(low=0, high=nx, size=ti_nsquares)
    side_length = int(np.sqrt((nx * ny * ti_pct_area / 100) / ti_nsquares))
    squares = []
    mask = np.zeros((ny, nx))

    for i in range(ti_nsquares):
        rr_start = max(0, rndy[i] - side_length // 2)
        cc_start = max(0, rndx[i] - side_length // 2)
        rr_end = min(ny, rr_start + side_length)
        cc_end = min(nx, cc_start + side_length)

        # Adjust start and end coordinates if the end exceeds the grid bounds
        if rr_end == ny:
            rr_start = max(0, ny - side_length)
        if cc_end == nx:
            cc_start = max(0, nx - side_length)
        
        rr_end = min(ny, rr_start + side_length)
        cc_end = min(nx, cc_start + side_length)
        
        #Rows and columns
        rr, cc = rectangle(start=(rr_start, cc_start), end=(rr_end, cc_end))
        
        # Ensure rr and cc are within bounds
        rr = np.clip(rr, 0, ny-1)
        cc = np.clip(cc, 0, nx-1)

        square_mask = np.zeros((ny, nx))
        square_mask[rr, cc] = 1
        
        square_indexes = np.argwhere(square_mask == 1)
        squares.append(square_indexes)
        

        current_pct = np.sum(mask) / (nx * ny) * 100
    
    return squares
    

def gen_ti_mask_single_square(nx, ny, simgrid_pct, ti_pct_area, seed, nseedGenerations=100, ntryPerSeed=1000, tolerance = 5):
    """
    Generate a binary mask with two overlapping squares within a grid. 
    The first square covers a percentage of the grid area, and the second square covers a percentage of the first square's area.

    Parameters:
    ----------
    nx : int
        Number of columns in the grid.
    ny : int
        Number of rows in the grid.
    simgrid_pct : float
        Percentage of the grid area to cover with the first square.
    ti_pct_area : float
        Percentage of the first square's area to cover with the second square.
    seed : int
        Seed for the random number generator.
    nseedGenerations : int (Optional, default: 100)
        Number of maximum seed to try for finding the best overlapping mask for the TI.
    ntryPerSeed : int (Optional, default: 1000)
        Number of attempts to place the squares to ensure the desired overlap.
    tolerance : int (Optional, default: 5)
        In percentage, the tolerance for the ti_pct_area.

    Returns:
    -------
    square1 : ndarray
        Binary mask of the first square.
    square2 : ndarray
        Binary mask of the second square.
    
    """
    for i in range(nseedGenerations):
        rng = np.random.default_rng(seed=seed)

        # Calculate side length of the first square
        first_square_area = (nx * ny * simgrid_pct) / 100
        side_length1 = int(np.sqrt(first_square_area))
        
        # Ensure side_length1 is at least 1
        side_length1 = max(side_length1, 1)
        
        # Generate random position for the first square
        rr_start1 = rng.integers(low=0, high=ny - side_length1 + 1)
        cc_start1 = rng.integers(low=0, high=nx - side_length1 + 1)
        rr_end1 = rr_start1 + side_length1
        cc_end1 = cc_start1 + side_length1

        # Generate the area of the second square
        first_square_area = side_length1 ** 2
        second_square_area = (first_square_area * ti_pct_area) / 100
        side_length2 = int(np.sqrt(second_square_area))
        
        # Ensure side_length2 is at least 1
        side_length2 = max(side_length2, 1)
        
        for i in range(ntryPerSeed) :
            rr_start2 = rng.integers(low=max(0, rr_start1 + side_length1 // 2 - side_length2),
                                     high=min(ny - side_length2, rr_start1 + side_length1 // 2 + side_length2))
            cc_start2 = rng.integers(low=max(0, cc_start1 + side_length1 // 2 - side_length2),
                                     high=min(nx - side_length2, cc_start1 + side_length1 // 2 + side_length2))
            rr_end2 = rr_start2 + side_length2
            cc_end2 = cc_start2 + side_length2
            
            # Generate masks
            square1 = np.zeros((ny, nx))
            square2 = np.zeros((ny, nx))
            
            rr1, cc1 = rectangle(start=(rr_start1, cc_start1), end=(rr_end1, cc_end1), shape=(ny, nx))
            rr2, cc2 = rectangle(start=(rr_start2, cc_start2), end=(rr_end2, cc_end2), shape=(ny, nx))
            
            square1[rr1, cc1] = 1
            square2[rr2, cc2] = 2
            
            # Check overlap and external area
            overlap_area = np.sum(square1[rr2, cc2] == 1)
            external_area = np.sum(square2) - overlap_area*2
            
            
            if overlap_area > 0 and external_area > 0:
                total_area_square1 = np.sum(square1 == 1)
                shared_pixels = np.sum(np.logical_and(square1 == 1, square2 == 2))
                overlap_percentage = (shared_pixels / total_area_square1) * 100

                if (abs(overlap_percentage - ti_pct_area) <= tolerance ):
                    return square1, square2, overlap_percentage
                    
            elif overlap_area <= 0 or external_area <= 0:
                raise ValueError(f"{ntryPerSeed} attempts for the seed {seed} were made to generate a mask without finding a suitable configuration for the training image. Please consider modifying the ntry value, modifying the ti_pct_area value or modifiying the simgrid_pct.")
        seed+=100
    raise ValueError(f"{n_seedGenerations} attempts were made to generate a mask without finding a suitable configuration for the training image, consider increasing the tolerance")



def build_ti(grid_msk, ti_ndisks, ti_pct_area, ti_realid, geolcd=True):
    """
    Build training images (TI) based on grid masks and other parameters.

    Parameters:
    ----------
    grid_msk : ndarray
        Mask defining areas of interest in the grid.
    ti_ndisks : int
        Number of disks to generate for the training images.
    ti_pct_area : float
        Percentage of the grid area to cover with disks.
    ti_realid : int
        Realization ID.
    geolcd : bool, optional
        Flag indicating whether to include geological codes.
    xycv : bool, optional
        Flag indicating whether to include x and y coordinates.

    Returns:
    -------
    geocodes : ndarray
        Unique geological codes.
    ngeocodes : int
        Number of unique geological codes.
    tiMissingGeol : geone.img.Img
        Geostatistical image object representing the training images.
    cond_data : geone.img.Img or None
        Conditional data object if `geolcd` is False, otherwise None.
    """
    geocodes = np.unique(grid_geo)
    ngeocodes = len(geocodes)
    novalue = -9999999
    nz = 1
    sx = vec_x[1] - vec_x[0]
    sy = vec_y[1] - vec_y[0]
    sz = sx
    ox = vec_x[0]
    oy = vec_y[0]
    oz = 0.0

    nv = 4
    varname = ['geo', 'grv', 'mag', 'lmp']

    #else:
    nv = 6
    varname = ['geo', 'grv', 'mag', 'lmp', 'x', 'y']
    xx, yy = np.meshgrid(vec_x, vec_y, indexing='xy')
    name = path2ti + 'ti' + suffix + '-ndisks-' + str(ti_ndisks) + '-areapct-' + str(ti_pct_area) + '-r-' + str(
        ti_realid) + '-geolcd' + str(geolcd) + '-xycv' + str(xycv) + '.gslib'
    val = np.ones((nv, nz, ny, nx)) * np.nan
    grid_geo_masked = grid_geo + 0
    grid_geo_masked[grid_msk < 1] = novalue
    val[0, 0, :, :] = grid_geo_masked
    val[1, 0, :, :] = grid_grv
    val[2, 0, :, :] = grid_mag
    val[3, 0, :, :] = grid_lmp


    # Create the Img class object
    tiMissingGeol = gn.img.Img(nx, ny, nz, sx, sy, sz, ox, oy, oz, nv, val, varname, name)

    if geolcd == False:
        val2 = val + 0 # Val2 is a copy of val but not at the same memory location
        val2[0, 0, :, :] = novalue
        cdname = path2cd + 'ti' + suffix + '-ndisks-' + str(ti_ndisks) + '-areapct-' + str(ti_pct_area) + '-r-' + str(
            ti_realid) + '-geolcd' + str(geolcd) + '-xycv' + str(xycv) + '.gslib'
        cond_data = gn.img.Img(nx, ny, nz, sx, sy, sz, ox, oy, oz, nv, val2, varname, cdname)

    else:
        cond_data = None
    gn.img.writeImageGslib(im=tiMissingGeol, filename=name, missing_value=None, fmt="%.10g")
    return geocodes, ngeocodes, tiMissingGeol, cond_data
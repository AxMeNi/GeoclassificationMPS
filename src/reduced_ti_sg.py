# -*- coding:utf-8 -*-
__projet__ = "GeoclassificationMPS"
__nom_fichier__ = "reduced_ti_sg"
__author__ = "MENGELLE Axel"
__date__ = "août 2024"

from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import random as rd


 
    
def generate_random_dimensions(cc_dg, rr_dg, area_to_cover):
    """
    Generate possible dimensions for rectangles that can cover a specified area within the data_grid.

    Parameters:
    ----------
    cc_dg : int
        Maximum number of columns for the rectangles = number of columns of the data grid.
    rr_dg : int
        Maximum number of rows for the rectangles = number of rows of the data grid.
    area_to_cover : int
        Total area that needs to be covered by the rectangles.

    Returns:
    -------
    cc_list : ndarray
        Array of possible numbers of columns for the rectangles.
    rr_list : ndarray
        Array of possible numbers of rows for the rectangles.
    """
    rr_list = [i for i in range(1,area_to_cover+1)]

    cc_list = [int(area_to_cover/i) for i in range(1,len(rr_list)+1)]

    rr_list, cc_list = zip(*[(rr, cc) for rr, cc in zip(rr_list, cc_list) if rr <= rr_dg and cc <= cc_dg])

    cc_list = np.array(cc_list)
    rr_list = np.array(rr_list)

    return cc_list, rr_list

def chose_random_dimensions(cc_list, rr_list):
    """
    Randomly select a pair of dimensions from the given lists and remove the selected pair from the lists.

    Parameters:
    ----------
    cc_list : ndarray
        Array of possible numbers of columns for rectangles.
    rr_list : ndarray
        Array of possible numbers of rows for rectangles.

    Returns:
    -------
    cc : int
        The chosen number of columns for a rectangle.
    rr : int
        The chosen number of rows for a rectangle.
    cc_list : ndarray
        Updated array of possible numbers of columns after removing the selected one.
    rr_list : ndarray
        Updated array of possible numbers of rows after removing the selected one.
    """
    rd_index = np.random.choice(len(cc_list))
    
    cc = cc_list[rd_index]
    rr = rr_list[rd_index]

    cc_list = np.delete(cc_list, rd_index)
    rr_list = np.delete(rr_list, rd_index)
    
    return cc, rr, cc_list, rr_list

def generate_random_sg_origin(cc_dg, rr_dg, cc_sg, rr_sg):
    """
    Generate all possible starting positions for a simulation grid within the data grid.

    Parameters:
    ----------
    cc_dg : int
        Number of columns in the data grid.
    rr_dg : int
        Number of rows in the data grid.
    cc_sg : int
        Number of columns in the simulation grid.
    rr_sg : int
        Number of rows in the simulation grid.

    Returns:
    -------
    positions_sg : ndarray
        Array of shape (n, 2) containing all possible (column, row) starting positions for the simulation grid.
    """
    c_sg_list = np.arange(0, cc_dg - cc_sg + 1)
    r_sg_list = np.arange(0, rr_dg - rr_sg + 1)

    c_sg_grid, r_sg_grid = np.meshgrid(c_sg_list, r_sg_list, indexing='ij')

    positions_sg = np.stack((c_sg_grid, r_sg_grid), axis=-1).reshape(-1, 2)
    
    return positions_sg

def chose_random_sg_origin(positions_sg):
    """
    Choose a random origin for the simulation grid from a list of possible positions and update the list.

    Parameters:
    ----------
    positions_sg : ndarray
        Array of possible positions for the simulation grid origin, each represented as a pair of coordinates.

    Returns:
    -------
    c_sg : int
        Column index of the chosen simulation grid origin.
    r_sg : int
        Row index of the chosen simulation grid origin.
    positions_sg : ndarray
        Updated array of positions, with the chosen position removed.
    """
    rd_index_position_sg = np.random.choice(len(positions_sg))
            
    position_sg = positions_sg[rd_index_position_sg]

    c_sg = position_sg[0]
    r_sg = position_sg[1]

    positions_sg = np.delete(positions_sg, rd_index_position_sg,axis=0)
    return c_sg, r_sg, positions_sg

def chose_random_overlap_area(c_overlap_list, c_sg, r_sg, cc_sg, rr_sg, ti_sg_overlap_percentage):
    """
    Choose a random overlap area and compute the position and dimensions of the overlap.

    Parameters:
    ----------
    c_overlap_list : array-like
        List of possible overlap column positions.
    c_sg : int
        Column position of the simulation grid.
    r_sg : int
        Row position of the simulation grid.
    cc_sg : int
        Number of columns in the simulation grid.
    rr_sg : int
        Number of rows in the simulation grid.
    ti_sg_overlap_percentage : int
        The SG should cover this percentage of the TI.

    Returns:
    -------
    cc_overlap : int
        Number of columns in the overlap area.
    rr_overlap : int
        Number of rows in the overlap area.
    c_overlap_list : array-like
        Updated list of possible overlap column positions.
    positions_overlap : ndarray
        Array of coordinates representing the positions of the overlap area.
    """
    rd_index_c_overlap_temp = np.random.choice(len(c_overlap_list))
            
    c_overlap = c_overlap_list[rd_index_c_overlap_temp]
    r_overlap = compute_row(c_overlap, c_sg, r_sg, cc_sg, rr_sg, ti_sg_overlap_percentage)
  
    c_overlap_list = np.delete(c_overlap_list, rd_index_c_overlap_temp)
    
    cc_overlap = cc_sg - c_overlap + c_sg
    rr_overlap = rr_sg - r_overlap + r_sg
    
    # Definitive choice of the origin of the overlap area
    positions_overlap = np.array(
        [(c, r_sg) for c in range(c_sg, c_overlap + 1)] +
        [(c, r_overlap) for c in range(c_sg, c_overlap + 1)] +
        [(c_sg, r) for r in range(r_sg + 1, r_overlap)] +
        [(c_overlap, r) for r in range(r_sg + 1, r_overlap)]
    )
    
    return cc_overlap, rr_overlap, c_overlap_list, positions_overlap

def compute_row(col, c_sg, r_sg, cc_sg, rr_sg, ti_sg_overlap_percentage):
    """
    Compute the row position based on the column position and overlap percentage.

    Parameters:
    ----------
    col : int
        Current column position.
    c_sg : int
        Column position of the simulation grid.
    r_sg : int
        Row position of the simulation grid.
    cc_sg : int
        Number of columns in the simulation grid.
    rr_sg : int
        Number of rows in the simulation grid.
    ti_sg_overlap_percentage : float
        Percentage of overlap between the training image and simulation grid.

    Returns:
    -------
    int
        Computed row position based on the column position and overlap percentage.
    """
    row = int(rr_sg+r_sg-(((ti_sg_overlap_percentage/100)*cc_sg*rr_sg)/(cc_sg-(col-c_sg))))
    return row


def chose_random_overlap_origin(positions_overlap):
    """
    Choose a random origin for the overlap area from the available positions.

    Parameters:
    ----------
    positions_overlap : ndarray
        Array of available positions for the overlap area, where each position is a pair of coordinates (column, row).

    Returns:
    -------
    c_overlap : int
        Column position of the chosen overlap origin.
    r_overlap : int
        Row position of the chosen overlap origin.
    positions_overlap : ndarray
        Updated array of positions after removing the chosen overlap origin.
    """
    rd_index_c_overlap_def = np.random.choice(len(positions_overlap))
                    
    position_overlap = positions_overlap[rd_index_c_overlap_def] 

    c_overlap = position_overlap[0]
    r_overlap = position_overlap[1]

    positions_overlap = np.delete(positions_overlap, rd_index_c_overlap_def,axis=0)
    
    return c_overlap, r_overlap, positions_overlap

def get_ti_orign(cc_dg, rr_dg, 
                c_sg, r_sg, 
                cc_sg, rr_sg, 
                c_overlap, r_overlap, 
                cc_overlap, rr_overlap,
                cc_ti = None, rr_ti = None):
    """
    Determine the origin of the training image (TI) based on the overlap area 
    between the simulation grid (SG) and the data grid (DG).

    Parameters:
    ----------
    cc_dg : int
        Number of columns in the data grid.
    rr_dg : int
        Number of rows in the data grid.
    c_sg : int
        Column position of the simulation grid origin.
    r_sg : int
        Row position of the simulation grid origin.
    cc_sg : int
        Number of columns in the simulation grid.
    rr_sg : int
        Number of rows in the simulation grid.
    c_overlap : int
        Column position of the overlap area origin.
    r_overlap : int
        Row position of the overlap area origin.
    cc_overlap : int
        Number of columns in the overlap area.
    rr_overlap : int
        Number of rows in the overlap area.
    cc_ti : int, optional
        Number of columns in the training image (default is None).
    rr_ti : int, optional
        Number of rows in the training image (default is None).

    Returns:
    -------
    c_ti : int
        Column position of the training image origin.
    r_ti : int
        Row position of the training image origin.
    position : str
        A string indicating the relative position of the TI origin within the overlap area.
        Possible values are "TOPRIGHT", "RIGHT", "BOTTOMRIGHT", "BOTTOM", "BOTTOMLEFT",
        "LEFT", "TOPLEFT", "TOP".

    Raises:
    ------
    ValueError
        If the origin of the TI cannot be determined based on the input values.
    """

    if (c_overlap + cc_overlap == c_sg + cc_sg) and (r_overlap + rr_overlap == r_sg + rr_sg):
        c_ti = c_overlap
        r_ti = r_overlap
        return c_ti, r_ti, "TOPRIGHT"
      
    if (c_overlap + cc_overlap == c_sg + cc_sg) and (r_sg < r_overlap) and (r_overlap + rr_overlap < r_sg + rr_sg):
        c_ti = c_overlap
        r_ti = r_overlap
        return c_ti, r_ti, "RIGHT"
        
    if (c_overlap + cc_overlap == c_sg + cc_sg)  and (r_overlap == r_sg):
        c_ti = c_overlap
        r_ti = r_overlap + rr_overlap - rr_ti
        return c_ti, r_ti, "BOTTOMRIGHT"
    
    if (c_overlap + cc_overlap < c_sg + cc_sg) and (c_sg < c_overlap) and (r_overlap == r_sg):
        c_ti = c_overlap
        r_ti = r_overlap + rr_overlap - rr_ti
        return c_ti, r_ti, "BOTTOM"
        
    if (c_overlap == c_sg) and (r_overlap == r_sg):
        c_ti = c_overlap + cc_overlap - cc_ti
        r_ti = r_overlap + rr_overlap - rr_ti
        return c_ti, r_ti, "BOTTOMLEFT"
        
    if (c_overlap == c_sg) and (r_overlap + rr_overlap < r_sg + rr_sg) and (r_sg < r_overlap):
        c_ti = c_overlap + cc_overlap - cc_ti
        r_ti = r_overlap
        return c_ti, r_ti, "LEFT"
                
    if (c_overlap == c_sg) and (r_overlap + rr_overlap == r_sg + rr_sg):
        c_ti = c_overlap + cc_overlap - cc_ti
        r_ti = r_overlap
        return c_ti, r_ti, "TOPLEFT"
        
    if (r_overlap + rr_overlap == r_sg + rr_sg) and (c_sg < c_overlap) and (c_overlap + cc_overlap < c_sg + cc_sg):
        c_ti = c_overlap
        r_ti = r_overlap
        return c_ti, r_ti, "TOP"
    print(ValueError(f"Error while trying to find the origin of the TI: \n Values : c_sg={c_sg}, cc_sg={cc_sg}, r_sg={r_sg},rr_dg={rr_sg}, c_overlap={c_overlap}, cc_overlap={cc_overlap}, r_overlap={r_overlap}, rr_overlap={rr_overlap}, cc_ti={cc_ti}, rr_ti={rr_ti} "))
    return None

def check_ti_pos(c_ti, r_ti, cc_ti, rr_ti, cc_dg, rr_dg, c_sg, r_sg, cc_sg, rr_sg, c_overlap, r_overlap, cc_overlap, rr_overlap, pos):
    """
    Check if the position and dimensions of a training image (TI) are valid given the constraints of the data grid (DG), simulation grid (SG), and overlap area.

    Parameters:
    ----------
    c_ti : int
        Column position of the TI origin.
    r_ti : int
        Row position of the TI origin.
    cc_ti : int
        Number of columns in the TI.
    rr_ti : int
        Number of rows in the TI.
    cc_dg : int
        Number of columns in the data grid.
    rr_dg : int
        Number of rows in the data grid.
    c_sg : int
        Column position of the simulation grid origin.
    r_sg : int
        Row position of the simulation grid origin.
    cc_sg : int
        Number of columns in the simulation grid.
    rr_sg : int
        Number of rows in the simulation grid.
    c_overlap : int
        Column position of the overlap area origin.
    r_overlap : int
        Row position of the overlap area origin.
    cc_overlap : int
        Number of columns in the overlap area.
    rr_overlap : int
        Number of rows in the overlap area.
    pos : str
        Relative position of the TI origin with respect to the overlap area. Possible values are "TOPRIGHT", "RIGHT", "BOTTOMRIGHT", "BOTTOM", "BOTTOMLEFT", "LEFT", "TOPLEFT", "TOP".

    Returns:
    -------
    bool
        True if the TI position and dimensions are valid based on the constraints, False otherwise.

    Notes:
    -----
    - The function performs boundary checks to ensure that the TI fits within the data grid and matches the specified overlap position.
    """
    if (c_ti < 0) or (r_ti < 0) or (c_ti + cc_ti > cc_dg) or (r_ti + rr_ti > rr_dg):
        return False

    if pos == "TOPRIGHT":
        if (c_ti + cc_ti < c_overlap + cc_overlap) or (r_ti + rr_ti < r_overlap + rr_overlap):
            return False

    if pos == "RIGHT":
        if (r_ti + rr_ti != r_overlap + rr_overlap) or (c_ti + cc_ti < c_overlap + cc_overlap):
            return False

    if pos == "BOTTOMRIGHT":
        if (r_ti + rr_ti != r_overlap + rr_overlap) or (c_ti + cc_ti < c_overlap + cc_overlap):
            return False

    if pos == "BOTTOM":
        if (r_ti + rr_ti != r_overlap + rr_overlap) or (c_ti + cc_ti != c_overlap + cc_overlap):
            return False

    if pos == "BOTTOMLEFT":
        if (r_ti + rr_ti != r_overlap + rr_overlap) or (c_ti + cc_ti != c_overlap + cc_overlap):
            return False

    if pos == "LEFT":
        if (r_ti + rr_ti != r_overlap + rr_overlap) or (c_ti + cc_ti != c_overlap + cc_overlap):
            return False

    if pos == "TOPLEFT":
        if (r_ti + rr_ti < r_overlap + rr_overlap) or (c_ti + cc_ti != c_overlap + cc_overlap):
            return False

    if pos == "TOP":
        if (r_ti + rr_ti < r_overlap + rr_overlap) or (c_ti + cc_ti != c_overlap + cc_overlap):
            return False
    return True

############################################################################
############################################################################
############################################################################
    
def get_ti_sg(cc_dg, rr_dg, 
              cc_sg=None, rr_sg=None, pct_sg=10, 
              cc_ti=None, rr_ti=None, pct_ti=30, 
              ti_sg_overlap_percentage=0, seed=None):
    """
    Generate the position and dimensions of a training image (TI) and a simulation grid (SG) based on the dimensions of the data grid (DG) and specified overlap area.

    Parameters:
    ----------
    cc_dg : int
        Number of columns in the data grid (DG).
    rr_dg : int
        Number of rows in the data grid (DG).
    cc_sg : int, optional
        Number of columns in the simulation grid (SG). If not provided, it will be calculated based on `pct_sg`.
    rr_sg : int, optional
        Number of rows in the simulation grid (SG). If not provided, it will be calculated based on `pct_sg`.
    pct_sg : int, optional
        Percentage of the data grid area to be used for the simulation grid (SG) if `cc_sg` and `rr_sg` are not provided. Default is 10%.
    cc_ti : int, optional
        Number of columns in the training image (TI). If not provided, it will be calculated based on `pct_ti`.
    rr_ti : int, optional
        Number of rows in the training image (TI). If not provided, it will be calculated based on `pct_ti`.
    pct_ti : int, optional
        Percentage of the data grid area to be used for the training image (TI) if `cc_ti` and `rr_ti` are not provided. Default is 30%.
    ti_sg_overlap_percentage : int, optional
        Percentage of the simulation grid (SG) area to be overlapped with the training image (TI). Default is 0%.
    seed : int, optional
        Seed for random number generation to ensure reproducibility. If not provided, a random seed will be generated.

    Returns:
    -------
    tuple
        A tuple containing the following elements:
        - c_sg : int : Column position of the simulation grid origin.
        - cc_sg : int : Number of columns in the simulation grid.
        - r_sg : int : Row position of the simulation grid origin.
        - rr_sg : int : Number of rows in the simulation grid.
        - c_overlap : int : Column position of the overlap area origin.
        - cc_overlap : int : Number of columns in the overlap area.
        - r_overlap : int : Row position of the overlap area origin.
        - rr_overlap : int : Number of rows in the overlap area.
        - c_ti : int : Column position of the training image origin.
        - cc_ti : int : Number of columns in the training image.
        - r_ti : int : Row position of the training image origin.
        - rr_ti : int : Number of rows in the training image.

    Raises:
    ------
    ValueError
        - If only one of the dimensions (`cc_sg`, `rr_sg` or `cc_ti`, `rr_ti`) is provided instead of both.
        - If the percentages provided are higher than 100
        - If the dimensions of the simulation or the training image are larger than the size of the auxiliary variable.

    Notes:
    -----
    - This function uses random sampling to generate positions and dimensions for the simulation grid and training image based on the data grid constraints.
    - Boundary checks are performed to ensure that the generated positions and dimensions are valid.
    - The overlap area between the training image and simulation grid is optional and controlled by `ti_sg_overlap_percentage`.
    - If no valid position is found, the function will exit with an error message.
    """
    if (cc_sg is None and rr_sg is not None) or (cc_sg is not None and rr_sg is None) or (cc_ti is None and rr_ti is not None) or (cc_ti is not None and rr_ti is None):
        print(ValueError(f"TI size and SG size must be precised : please consider chosing a size (columns AND rows) or a percentage of the Simulation Grid."))
        exit()
        
    if seed is None:
        seed = int(rd.randint(1,2**32-1))
        print(f"Seed used to generate the TI and the simulation grid : {seed}")
    np.random.seed(seed)
        
    if (pct_ti is not None):
        if (pct_ti > 100):
            raise ValueError(f"The percentage of the grid covered by the TI provided is higher than 100! Please consider chosing a percentage lower than 100.")
    
    if  (pct_sg is not None):
        if (pct_ti > 100):
            raise ValueError(f"The percentage of the grid covered by the simulation grid provided is higher than 100! Please consider chosing a percentage lower than 100.")
    
    
    if (cc_sg is None) and (rr_sg is None):
        area_sg = int(pct_sg/100 * (cc_dg * rr_dg))
        cc_sg_list, rr_sg_list = generate_random_dimensions(cc_dg, rr_dg, area_sg)
    else:
        if (cc_sg > cc_dg) or (rr_sg > rr_dg):
            raise ValueError(f"The dimensions of the simulation grid are too large! Please enter the dimensions within the auxiliary grid limits: nmax_columns = {cc_dg}, nmax_rows = {rr_dg}.")
        cc_sg_list, rr_sg_list = np.array([cc_sg]), np.array([rr_sg])
        
    if (cc_ti is None) and (rr_ti is None):
        area_ti = int(pct_ti/100 * (cc_dg * rr_dg))
    else:
        if (cc_ti > cc_dg) or (rr_ti > rr_dg):
            raise ValueError(f"The dimensions of the TI are too large! Please enter the dimensions within the auxiliary grid limits: nmax_columns = {cc_dg}, nmax_rows = {rr_dg}.")
    
    while cc_sg_list.size > 0:
    
        cc_sg, rr_sg, cc_sg_list, rr_sg_list = chose_random_dimensions(cc_sg_list, rr_sg_list)
        
        positions_sg = generate_random_sg_origin(cc_dg, rr_dg, cc_sg, rr_sg)

        while positions_sg.size > 0:
            
            c_sg, r_sg, positions_sg = chose_random_sg_origin(positions_sg)
            

            c_overlap_list = np.arange(c_sg, int(c_sg+cc_sg-(ti_sg_overlap_percentage/100)*cc_sg))

            while c_overlap_list.size > 0:
                cc_overlap, rr_overlap, c_overlap_list, positions_overlap = chose_random_overlap_area(c_overlap_list, c_sg, r_sg, cc_sg, rr_sg, ti_sg_overlap_percentage)

                while positions_overlap.size > 0:
                    c_overlap, r_overlap, positions_overlap = chose_random_overlap_origin(positions_overlap)
                    
                    if (cc_ti is None) and (rr_ti is None):
                        cc_ti_list, rr_ti_list = generate_random_dimensions(cc_dg, rr_dg, area_ti)
                    else:
                        cc_ti_list, rr_ti_list = np.array([cc_ti]), np.array([rr_ti])
                    while cc_ti_list.size > 0:
                        
                        cc_ti, rr_ti, cc_ti_list, rr_ti_list = chose_random_dimensions(cc_ti_list, rr_ti_list)
                        
                        c_ti, r_ti, pos = get_ti_orign(cc_dg, rr_dg, 
                                                c_sg, r_sg, 
                                                cc_sg, rr_sg, 
                                                c_overlap, r_overlap, 
                                                cc_overlap, rr_overlap, 
                                                cc_ti = cc_ti, rr_ti = rr_ti)
                        
                        valid_ti_pos = check_ti_pos(c_ti, r_ti, cc_ti, rr_ti, cc_dg, rr_dg, c_sg, r_sg, cc_sg, rr_sg, c_overlap, r_overlap, cc_overlap, rr_overlap, pos)

                        if valid_ti_pos :
                            return c_sg, cc_sg, r_sg ,rr_sg, c_overlap, cc_overlap, r_overlap, rr_overlap, c_ti, cc_ti, r_ti, rr_ti
    
    print("No position matched with the paramaters given to create the simulation grid and the TI, please change the parameters.")
    exit()
    
    


if __name__ == "__main__":

    ##### For the data_grid :

    cc_dg = 350
    rr_dg = 624

    ##### For the simulation grid : 

    pct_sg = 10
    cc_sg = 50
    rr_sg = 50

    ##### For the TI:

    cc_ti = 75
    rr_ti = 90

    ### For the overlap area

    ti_sg_overlap_percentage = 10
    
    
    #output = get_ti_sg(cc_dg, rr_dg, ti_sg_overlap_percentage, cc_sg = cc_sg, rr_sg = rr_sg, cc_ti = cc_ti, rr_ti = rr_ti)
    #output = get_ti_sg(cc_dg, rr_dg, ti_sg_overlap_percentage, pct_sg = pct_sg, pct_ti = pct_ti)
    #output = get_ti_sg(cc_dg, rr_dg, ti_sg_overlap_percentage, pct_ti = pct_ti, cc_sg = cc_sg, rr_sg = rr_sg)
    output = get_ti_sg(cc_dg, rr_dg, ti_sg_overlap_percentage = 10, pct_sg = 23, cc_ti = 100, rr_ti = 50, seed=55)
    c_sg, cc_sg, r_sg ,rr_sg, c_overlap, cc_overlap, r_overlap, rr_overlap, c_ti, cc_ti, r_ti, rr_ti = output
    print(f"c_sg={c_sg}, cc_sg={cc_sg}, r_sg={r_sg}, rr_sg={rr_sg}, c_overlap={c_overlap}, cc_overlap={cc_overlap}, r_overlap={r_overlap}, rr_overlap={rr_overlap}, c_ti={c_ti}, cc_ti={cc_ti}, r_ti={r_ti}, rr_ti={rr_ti} ")


    plt.figure(figsize=(10, 6))

    sim_grid        = plt.Rectangle((c_sg,r_sg), cc_sg, rr_sg, fill=None, edgecolor='blue', linewidth=2, label=r'simulation_grid')
    data_grid       = plt.Rectangle((0,0), cc_dg, rr_dg, fill=None, edgecolor='green', linewidth=2, linestyle='--', label='data_grid')
    TI              = plt.Rectangle((c_ti, r_ti), cc_ti, rr_ti, fill=None, edgecolor='red', linewidth=2, linestyle='-.', label='TI')
    overlap_rect    = plt.Rectangle((c_overlap, r_overlap), cc_overlap, rr_overlap, fill=None, edgecolor='orange', linewidth=2, linestyle='-.', label='overlap_area')

    plt.gca().add_patch(sim_grid)
    plt.gca().add_patch(data_grid)
    plt.gca().add_patch(TI)
    plt.gca().add_patch(overlap_rect)

    plt.xlim(-10, 1100)
    plt.ylim(-10, 1100)
    plt.xlabel('cc')
    plt.ylabel('rr')
    plt.title('Rectangles Plot')
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.show()











# -*- coding:utf-8 -*-
__projet__ = "GeoclassificationMPS"
__nom_fichier__ = "interface"
__author__ = "MENGELLE Axel"
__date__ = "juillet 2024"

import numpy as np
import pandas as pd

def check_variables(simulated_var, auxiliary_var, names_var, types_var, novalue=-9999999):
    """
    Check the validity of input variables used in a simulation or analysis.

    Parameters:
    simulated_var : dict
        Dictionary containing simulated variables with arrays as values.
    auxiliary_var : dict
        Dictionary containing auxiliary variables with arrays as values.
    names_var : numpy.ndarray
        Array containing names corresponding to variables in simulated_var and auxiliary_var.
    types_var : numpy.ndarray
        Array containing expected types for each variable in simulated_var and auxiliary_var.
        Expected types are "continuous" for numerical types and "categorical" for integer types.
    novalue : float (Optional)
        The value to set when a cell is empty

    Returns:
    dict, dict:
        Modified dictionaries simulated_var and auxiliary_var where None values are replaced by -9999999.

    Raises:
    ValueError:
        - If the number of variable names does not match the number of variables in simulated_var or auxiliary_var.
        - If simulated_var and auxiliary_var do not have the same XY dimensions.
        - If auxiliary_var contains NaN values.
        - If simulated_var or auxiliary_var contains -9999999 values.
    TypeError:
        - If the type of any variable in simulated_var or auxiliary_var does not match the expected type in types_var.
    """

    # Check for no value in the auxiliary variable
    for key in auxiliary_var:
        if np.isnan(auxiliary_var[key]).any() or np.any(auxiliary_var[key] == novalue):
            raise ValueError(f"auxiliary_var contains NaN values or {novalue} in '{key}', but it must be fully informed.")
            
            
    # Replace novalue by "nan"
    for key in simulated_var:
        if np.any(simulated_var[key] == novalue):
            if np.issubdtype(simulated_var[key].dtype, np.floating):
                simulated_var[key] = np.where(simulated_var[key] == novalue, np.nan, simulated_var[key])
            if np.issubdtype(simulated_var[key].dtype, np.integer):
                raise ValueError(f"simulated_var {key} cannot contain novalue {novalue} since it is of type {np.integer}.")


    
    # Check dimensions XY
    for key in simulated_var:
        for key2 in auxiliary_var:
            if simulated_var[key].shape != auxiliary_var[key2].shape:
                raise ValueError(f"simulated_var and auxiliary_var do not have the same dimensions XY for '{key}' and '{key2}'.")
    
   
    # Check for novalue values in simulated_var and auxiliary_var
    for key in simulated_var:
        if np.any(simulated_var[key] == np.nan):
            print(f"simulated_var contains {novalue} values in '{key}'. Those were replaced by {np.nan}")
    for key in auxiliary_var:
        if np.any(auxiliary_var[key] == np.nan):
            print(f"auxiliary_var contains {novalue} values in '{key}'. Those were replaced by {np.nan}")
    
    # Check types
    for var_i in range(len(names_var[0])):
        var_name_sim = names_var[0][var_i]
        
        if var_name_sim in simulated_var:
            sim_array = simulated_var[var_name_sim]

            expected_type = types_var[0][var_i]

            # Check type "continuous"
            if expected_type == "continuous":
                if not np.issubdtype(sim_array.dtype, np.number):
                    raise TypeError(f"Type mismatch for simulated_var '{var_name_sim}'. Expected numerical type for 'continuous', got {sim_array.dtype}.")
                           
            # Check type "categorical"
            elif expected_type == "categorical":
                if not np.issubdtype(sim_array.dtype, np.integer):
                    raise TypeError(f"Type mismatch for simulated_var '{var_name_sim}'. Expected integer type for 'categorical', got {sim_array.dtype}.")
                            
            # Invalid type
            else:
                raise ValueError(f"Invalid type '{expected_type}' specified. Expected 'continuous' or 'categorical'.")
                
    for aux_j in range(len(names_var[1])):
        var_name_aux = names_var[1][aux_j]
        
        if var_name_sim in simulated_var:
            aux_array = auxiliary_var[var_name_aux]

            expected_type = types_var[1][aux_j]
            
            # Check type "continuous"
            if expected_type == "continuous":
                if not np.issubdtype(aux_array.dtype, np.number):
                    raise TypeError(f"Type mismatch for auxiliary_var '{var_name_aux}'. Expected numerical type for 'continuous', got {aux_array.dtype}.")
            
            # Check type "categorical"
            elif expected_type == "categorical":
                if not np.issubdtype(aux_array.dtype, np.integer):
                    raise TypeError(f"Type mismatch for auxiliary_var '{var_name_aux}'. Expected integer type for 'categorical', got {aux_array.dtype}.")
            
            # Invalid type
            else:
                raise ValueError(f"Invalid type '{expected_type}' specified. Expected 'continuous' or 'categorical'.")
    
    return simulated_var, auxiliary_var

def create_auxiliary_and_simulated_var(csv_file_path):
    data_df = pd.read_csv(csv_file_path, sep=';')

    auxiliary_var = {}
    simulated_var = {}

    names_var = [[], []]
    types_var = [[], []]

    # Iterate through each row of the DataFrame
    for _, row in data_df.iterrows():
        var_name = str(row['var_name'])
        categ_conti = row['categ_conti']
        sim_aux = row['sim_aux']
        path = row['path']
        array_data = np.load(path)
        
        # Store the array in the appropriate dictionary
        if sim_aux == 'aux':
            if var_name != 'nan' :
                names_var[1].append(var_name)
                auxiliary_var [var_name] = array_data
            else:
                raise NameError(f"One auxiliary variable has no name, please consider naming all of your variables with different names")
            types_var[1].append(categ_conti)
        elif sim_aux == 'sim':
            if var_name != 'nan' :
                names_var[0].append(var_name)
                simulated_var[var_name] = array_data
            else:
                raise NameError(f"One simulated variable has no name, please consider naming all of your variables with different names")
               
            types_var[0].append(categ_conti)
    
    return simulated_var, auxiliary_var, names_var, types_var
    
def get_sim_grid_dimensions(simulated_var, simgrid_mask=None):
    ar  = simulated_var[next(iter(simulated_var))]
    nc, nr = ar.shape[0], ar.shape[1]
    return nc, nr
    
    
def create_directories(path2ti,path2cd,path2real, path2log, path2ind):
    """
    Create directories if they do not exist
    
    """
    if not os.path.exists(path2ti):
        os.makedirs(path2ti)
    if not os.path.exists(path2cd):
        os.makedirs(path2cd)
    if not os.path.exists(path2real):
        os.makedirs(path2real)
    if not os.path.exists(path2log):
        os.makedirs(path2log)
    if not os.path.exists(path2ind):
        os.makedirs(path2ind)
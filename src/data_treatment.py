# -*- coding:utf-8 -*-
__projet__ = "GeoclassificationMPS"
__nom_fichier__ = "interface"
__author__ = "MENGELLE Axel"
__date__ = "juillet 2024"

import numpy as np
from pickle import *

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
    # Replace None with novalue
    for key in simulated_var:
        simulated_var[key] = np.where(np.isnan(simulated_var[key]), novalue, simulated_var[key])
    for key in auxiliary_var:
        auxiliary_var[key] = np.where(np.isnan(auxiliary_var[key]), novalue, auxiliary_var[key])
    
    # Check for variable names
    num_vars_sim = len(simulated_var)
    num_vars_aux = len(auxiliary_var)
    if num_vars_sim != len(names_var[0]) or num_vars_aux != len(names_var[1]):
        message = "The number of variable names does not match the number of variables."
        if num_vars_sim != len(names_var[0]):
            message += f" In simulated_var, expected {num_vars_sim}, got {len(names_var[0])}."
        if num_vars_aux != len(names_var[1]):
            message += f" In auxiliary_var, expected {num_vars_aux}, got {len(names_var[1])}."
        raise ValueError(message)
    
    # Check dimensions XY
    for key in simulated_var:
        for key2 in auxiliary_var:
            if simulated_var[key].shape != auxiliary_var[key2].shape:
                raise ValueError(f"simulated_var and auxiliary_var do not have the same dimensions XY for '{key}' and '{key2}'.")
    
    # Check for NaN values in auxiliary_var
    for key in auxiliary_var:
        if np.isnan(auxiliary_var[key]).any():
            raise ValueError(f"auxiliary_var contains NaN values in '{key}', but it must be fully informed.")
    
    # Check for novalue values in simulated_var and auxiliary_var
    for key in simulated_var:
        if np.any(simulated_var[key] == novalue):
            print(f"simulated_var contains novalue values in '{key}'.")
    for key in auxiliary_var:
        if np.any(auxiliary_var[key] == novalue):
            print(f"auxiliary_var contains novalue values in '{key}'.")
    
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
    
def create_sim_and_aux(names_var, sim_var, aux_var):
    simulated_var = {}
    auxiliary_var = {}

    for i in range(len(names_var[0])):
        simulated_var[names_var[0][i]] = sim_var[i]
    for i in range(len(names_var[1])):
        auxiliary_var[names_var[1][i]] = aux_var[i]
    
    return simulated_var, auxiliary_var
    
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
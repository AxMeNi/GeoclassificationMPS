# -*- coding:utf-8 -*-
__projet__ = "GeoclassificationMPS"
__nom_fichier__ = "interface"
__author__ = "MENGELLE Axel"
__date__ = "juillet 2024"

import numpy as np
import pandas as pd

def check_variables(sim_var, auxdesc_var, auxcond_var, cond_var, names_var, types_var, novalue=-9999999):
    """
    Validate the types, consistency, and dimensions of variables, and process missing values.
    This function performs the following tasks:
    
    1. **Type Validation**: 
       - Ensures that the data type of each variable (simulated, auxiliary descriptive, auxiliary conditioning, and conditioning) matches its expected type ('continuous' or 'categorical').
    2. **Missing Value Handling**: 
       - Replaces any instances of a specified `novalue` in the variables with `np.nan`.
    3. **Shape Consistency**: 
       - Checks that all variables have consistent dimensions (XY) across their respective categories.
    4. **Name Validation**:
       - Confirms that each conditioning variable name matches a corresponding simulated variable name.
       - Ensures that each auxiliary descriptive variable has a corresponding auxiliary conditioning variable.

    Parameters:
    ----------
    sim_var : dict
        Dictionary containing simulated variables, where each key is a variable name and each value is a NumPy array.
    auxdesc_var : dict
        Dictionary containing auxiliary descriptive variables.
    auxcond_var : dict
        Dictionary containing auxiliary conditioning variables.
    cond_var : dict
        Dictionary containing conditioning variables.
    names_var : list of lists
        A list containing four sub-lists:
        - `names_var[0]`: Names of simulated variables.
        - `names_var[1]`: Names of auxiliary descriptive variables.
        - `names_var[2]`: Names of auxiliary conditioning variables.
        - `names_var[3]`: Names of conditioning variables.
    types_var : list of lists
        A list containing four sub-lists:
        - `types_var[0]`: Expected types ('continuous' or 'categorical') of simulated variables.
        - `types_var[1]`: Expected types of auxiliary descriptive variables.
        - `types_var[2]`: Expected types of auxiliary conditioning variables.
        - `types_var[3]`: Expected types of conditioning variables.
    novalue : scalar
        The value used to represent missing data in the variables, which will be replaced with `np.nan`.

    Returns:
    -------
    sim_var : dict
        A dictionary containing the validated and processed simulated variables.
    auxdesc_var : dict
        A dictionary containing the validated and processed auxiliary descriptive variables.
    auxcond_var : dict
        A dictionary containing the validated and processed auxiliary conditioning variables.
    cond_var : dict
        A dictionary containing the validated and processed conditioning variables.

    Raises:
    ------
    TypeError
        If the data type of a variable does not match its expected type.
    ValueError
        If an invalid type is specified, or if variables do not have consistent dimensions.
    NameError
        If a conditioning variable's name does not match any simulated variable, or if an auxiliary descriptive variable lacks a corresponding auxiliary conditioning variable.

    Notes:
    -----
    - This function assumes that each variable has a valid name and type.
    - It is important that the dimensions of all variables within their respective categories (e.g., simulated, auxiliary) are consistent.
    """
    
        
    # Check types
    for var_i in range(len(names_var[0])):
        varname_sim = names_var[0][var_i]
        sim_array = sim_var[varname_sim]
        expected_type = types_var[0][var_i]
        if expected_type == "continuous":
            if not np.issubdtype(sim_array.dtype, np.number):
                raise TypeError(f"Type mismatch for sim_var '{varname_sim}'. Expected numerical type for 'continuous', got {sim_array.dtype}.")    
        elif expected_type == "categorical":
            if not np.issubdtype(sim_array.dtype, np.integer):
                raise TypeError(f"Type mismatch for sim_var '{varname_sim}'. Expected integer type for 'categorical', got {sim_array.dtype}.")    
        else:
            raise ValueError(f"Invalid type for {varname_sim} '{expected_type}' specified. Expected 'continuous' or 'categorical'.")
                
    for var_i in range(len(names_var[1])):
        varname_auxdesc = names_var[1][var_i]
        auxdesc_array = auxdesc_var[varname_auxdesc]
        expected_type = types_var[1][var_i]
        if expected_type == "continuous":
            if not np.issubdtype(auxdesc_array.dtype, np.number):
                raise TypeError(f"Type mismatch for auxdesc_var '{varname_auxdesc}'. Expected numerical type for 'continuous', got {auxdesc_array.dtype}.")    
        elif expected_type == "categorical":
            if not np.issubdtype(auxdesc_array.dtype, np.integer):
                raise TypeError(f"Type mismatch for auxdesc_var '{varname_auxdesc}'. Expected integer type for 'categorical', got {auxdesc_array.dtype}.")    
        else:
            raise ValueError(f"Invalid type for {varname_auxdesc} '{expected_type}' specified. Expected 'continuous' or 'categorical'.")
    
    for var_i in range(len(names_var[2])):
        varname_auxcond = names_var[2][var_i]
        auxcond_array = auxcond_var[varname_auxcond]
        expected_type = types_var[2][var_i]
        if expected_type == "continuous":
            if not np.issubdtype(auxcond_array.dtype, np.number):
                raise TypeError(f"Type mismatch for auxcond_var '{varname_auxcond}'. Expected numerical type for 'continuous', got {auxcond_array.dtype}.")    
        elif expected_type == "categorical":
            if not np.issubdtype(auxcond_array.dtype, np.integer):
                raise TypeError(f"Type mismatch for auxcond_var '{varname_auxcond}'. Expected integer type for 'categorical', got {auxcond_array.dtype}.")    
        else:
            raise ValueError(f"Invalid type for {varname_auxcond} '{expected_type}' specified. Expected 'continuous' or 'categorical'.")
            
    for var_i in range(len(names_var[3])):
        varname_cond = names_var[3][var_i]
        cond_array = cond_var[varname_cond]
        expected_type = types_var[3][var_i]
        if expected_type == "continuous":
            if not np.issubdtype(cond_array.dtype, np.number):
                raise TypeError(f"Type mismatch for cond_var '{varname_cond}'. Expected numerical type for 'continuous', got {cond_array.dtype}.")    
        elif expected_type == "categorical":
            if not np.issubdtype(cond_array.dtype, np.integer):
                raise TypeError(f"Type mismatch for cond_var '{varname_cond}'. Expected integer type for 'categorical', got {cond_array.dtype}.")    
        else:
            raise ValueError(f"Invalid type for {varname_cond} '{expected_type}' specified. Expected 'continuous' or 'categorical'.")
    
    first_var = next(iter(sim_var))
    shape_var = sim_var[first_var].shape
                    
    # Replace novalue by "nan" and check dimensions XY
    for key in sim_var:
        if np.any(sim_var[key] == novalue):
            sim_var[key] = np.where(sim_var[key] == novalue, np.nan, sim_var[key])
        if sim_var[key].shape != shape_var:
                raise ValueError(f"Simulated variable does not have the same dimensions XY for '{first_var}' and '{key}'.")
    for key in auxdesc_var:
        if np.any(auxdesc_var[key] == novalue):
            auxdesc_var[key] = np.where(auxdesc_var[key] == novalue, np.nan, auxdesc_var[key])
        if auxdesc_var[key].shape != shape_var:
                raise ValueError(f"Auxiliary descriptive variable does not have the same dimensions XY for '{first_var}' and '{key}'.")
    for key in auxcond_var:
        if np.any(auxcond_var[key] == novalue):
            auxcond_var[key] = np.where(auxcond_var[key] == novalue, np.nan, auxcond_var[key])
        if auxcond_var[key].shape != shape_var:
                raise ValueError(f"Auxiliary conditioning variable does not have the same dimensions XY for '{first_var}' and '{key}'.")
    for key in cond_var:
        if np.any(cond_var[key] == novalue):
            cond_var[key] = np.where(cond_var[key] == novalue, np.nan, cond_var[key])
        if cond_var[key].shape != shape_var:
                raise ValueError(f"Conditioning variable does not have the same dimensions XY for '{first_var}' and '{key}'.")
    
    #Check that the names of the conditioning data are present in the list of the names of the simulated variables
    for name_condvar in names_var[3]:
        if name_condvar not in names_var[0]:
            raise NameError(f"The name of the conditioning variable '{name_condvar}' does not match any of the simulated variables. Conditioning variables must be named according to their corresponding simulated variables.")
    
    #Check that each auxdesc has one auxcond
    for name_auxdescvar in names_var[1]:
        if name_auxdescvar not in names_var[2] :
            raise NameError(f"The auxiliary descriptive variable '{name_auxdescvar}' has not matching auxcond_var. All auxiliary variables must be descriptive and conditioning.")
    
    return sim_var, auxdesc_var, auxcond_var, cond_var

def create_auxiliary_and_simulated_var(csv_file_path):
    """
    Load variables from a CSV file and categorize them into different categories: simulated, auxiliary descriptive, auxiliary conditioning, and conditioning variables.

    This function reads a CSV file containing metadata about various variables and loads the corresponding NumPy arrays from the specified paths. It then categorizes each variable based on its nature into one of the following categories: simulated variables, auxiliary descriptive variables, auxiliary conditioning variables, or conditioning variables.

    Parameters:
    ----------
    csv_file_path : str
        The path to the CSV file containing the metadata for each variable. The CSV file is expected to have the following columns:
        - 'var_name': Name of the variable.
        - 'categ_conti': Type of the variable (categorical or continuous).
        - 'nature': Nature of the variable (one of 'sim', 'auxdesc', 'auxcond', 'cond').
        - 'path': Path to the `.npy` file where the variable data is stored.

    Returns:
    -------
    sim_var : dict
        Dictionary with simulated variables, where each key is the variable name and each value is the corresponding NumPy array.
    auxdesc_var : dict
        Dictionary with auxiliary descriptive variables, where each key is the variable name and each value is the corresponding NumPy array.
    auxcond_var : dict
        Dictionary with auxiliary conditioning variables, where each key is the variable name and each value is the corresponding NumPy array.
    cond_var : dict
        Dictionary with conditioning variables, where each key is the variable name and each value is the corresponding NumPy array.

    names_var : list of lists
        A list of four sublists, where each sublist contains the names of the variables for each category:
        - `names_var[0]`: List of names of simulated variables.
        - `names_var[1]`: List of names of auxiliary descriptive variables.
        - `names_var[2]`: List of names of auxiliary conditioning variables.
        - `names_var[3]`: List of names of conditioning variables.

    types_var : list of lists
        A list of four sublists, where each sublist contains the types (categorical or continuous) of the variables for each category:
        - `types_var[0]`: List of types of simulated variables.
        - `types_var[1]`: List of types of auxiliary descriptive variables.
        - `types_var[2]`: List of types of auxiliary conditioning variables.
        - `types_var[3]`: List of types of conditioning variables.

    Raises:
    ------
    NameError
        If any variable is missing a name ('nan') in the CSV file.
    NameError
        If two variables of the same nature have the same name.
    ValueError
        If any variable has an invalid nature (not 'sim', 'auxdesc', 'auxcond', or 'cond').

    Notes:
    -----
    - The function assumes that each variable has a unique and valid name.
    - The nature of the variable determines its category, which can be one of the following:
        - "sim" for a simulated variable.
        - "auxdesc" for an auxiliary variable describing the simulated variable(s) in the TI.
        - "auxcond" for an auxiliary variable conditioning the variability of the simulated variable(s) in the simulation grid.
        - "cond" for a conditioning variable.
    """
    data_df = pd.read_csv(csv_file_path, sep=';')
    
    sim_var = {}
    auxdesc_var = {}
    auxcond_var = {}
    cond_var = {}

    names_var = [[], [], [], []]
    types_var = [[], [], [], []]

    for i, row in data_df.iterrows():
        var_name = str(row['var_name'])
        categ_conti = row['categ_conti']
        nature = row['nature']
        path = row['path']
        array_data = np.load(path)
        
        #SIMULATED
        if nature == 'sim':
            if var_name != 'nan' :
                if var_name not in names_var[0]:
                    names_var[0].append(var_name)
                    sim_var[var_name] = array_data
                else:
                    raise NameError(f"Line {i+1} of the CSV file : Two simulated variables have the same name, please consider naming all of your variables with different names.")
            else:
                raise NameError(f"Line {i+1} of the CSV file : One simulated variable has no name, please consider naming all of your variables with different names.")
            types_var[0].append(categ_conti)
        
        #AUXDESC
        elif nature == 'auxdesc':
            if var_name != 'nan' :
                if var_name not in names_var[1]:
                    names_var[1].append(var_name)
                    auxdesc_var[var_name] = array_data
                else:
                    raise NameError(f"Line {i+1} of the CSV file : Two auxiliary descriptive variables have the same name, please consider naming all of your variables with different names.")
            else:
                raise NameError(f"Line {i+1} of the CSV file : One auxiliary descriptive variable has no name, please consider naming all of your variables with different names.")
            types_var[1].append(categ_conti)
        
        #AUXCOND
        elif nature == 'auxcond':
            if var_name != 'nan' :
                if var_name not in names_var[2]:
                    names_var[2].append(var_name)
                    auxcond_var[var_name] = array_data
                else:
                    raise NameError(f"Line {i+1} of the CSV file : Two auxiliary conditioning variables have the same name, please consider naming all of your variables with different names.")
            else:
                raise NameError(f"Line {i+1} of the CSV file : One auxiliary conditioning variable has no name, please consider naming all of your variables with different names.")
            types_var[2].append(categ_conti)
            
        #COND
        elif nature == 'cond':
            if var_name != 'nan' :
                if var_name not in names_var[3]:
                    names_var[3].append(var_name)
                    cond_var[var_name] = array_data
                else:
                    raise NameError(f"Line {i+1} of the CSV file : Two condtioning variables have the same name, please consider naming all of your variables with different names.")
            else:
                raise NameError(f"Line {i+1} of the CSV file : One conditioning variable has no name, please consider naming all of your variables with different names.")
            types_var[3].append(categ_conti)
        
        else: 
            raise ValueError(f"Line {i+1} of the CSV file : One variable has an invalid nature ({nature}), please consider chosing between the following natures: \
                            \n    - \"sim\" for a simulated variable;\
                            \n    - \"auxdesc\" for an auxiliary variable describing the simulated variable(s) in the TI;\
                            \n    - \"auxcond\" for an auxiliary variable conditioning the variability of the simulated variable(s) in the simulation grid;\
                            \n    - \"cond\" for a conditioning variable.")
            
    return sim_var, auxdesc_var, auxcond_var, cond_var, names_var, types_var
    
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
# -*- coding:utf-8 -*-
__projet__ = "GeoclassificationMPS"
__nom_fichier__ = "interface"
__author__ = "MENGELLE Axel"
__date__ = "juillet 2024"

import numpy as np
import pandas as pd



def check_ti_methods(ti_methods):
    """
    Validate the list of TI (Training Image) methods to ensure it contains valid and compatible methods for processing.

    This function checks if the provided list of TI methods includes at least one of the four required methods. It also validates that if "ReducedTiSg" is chosen, it is the only method in the list, as it cannot be used in combination with other methods.

    Parameters:
    ----------
    ti_methods : list of str
        A list containing the names of the methods to be validated. The acceptable methods are:
        - "DependentCircles": Indicates a specific method for dependent circle analysis.
        - "DependentSquares": Indicates a specific method for dependent square analysis.
        - "IndependentSquares": Indicates a method for independent square analysis.
        - "ReducedTiSg": A method for reduced TI-SG analysis; this method must be used alone.

    Raises:
    ------
    ValueError
        If `ti_methods` does not contain at least one of the required methods.
    ValueError
        If "ReducedTiSg" is chosen along with any other method, as it must be selected alone.

    Returns:
    -------
    None
    """
    required_methods = ["DependentCircles", "DependentSquares", "IndependentSquares", "ReducedTiSg"]
    
    if not any(method in ti_methods for method in required_methods):
        raise ValueError('The list ti_methods must contain at least one of the four following methods: "DependentCircles", "DependentSquares", "IndependentSquares", "ReducedTiSg"')
        exit()
    if ("ReducedTiSg" in ti_methods) and (len(ti_methods) > 1):
        raise ValueError("Cannot chose ReducedTiSg with other methods, ReducedTiSg must be chosen solo.")
    return


def create_variables(csv_file_path):
    """
    Load variables from a CSV file and categorize them into different categories: simulated, auxiliary TI, auxiliary SG, and conditioning variables.

    This function reads a CSV file containing metadata about various variables and loads the corresponding NumPy arrays from the specified paths. It then categorizes each variable based on its nature into one of the following categories: simulated variables, auxiliary TI variables, auxiliary SG variables, or conditioning variables.

    Parameters:
    ----------
    csv_file_path : str
        The path to the CSV file containing the metadata for each variable. The CSV file is expected to have the following columns:
        - 'var_name': Name of the variable.
        - 'categ_conti': Type of the variable (categorical or continuous).
        - 'nature': Nature of the variable (one of 'sim', 'auxTI', 'auxSG', 'condIm').
        - 'grid': The grid in which the variable is : the 'TI' or the 'SG'
        - 'path': Path to the `.npy` file where the variable data is stored.

    Returns:
    -------
    sim_var : dict
        Dictionary with simulated variables, where each key is the variable name and each value is the corresponding NumPy array.
    auxTI_var : dict
        Dictionary with auxiliary TI variables, where each key is the variable name and each value is the corresponding NumPy array.
    auxSG_var : dict
        Dictionary with auxiliary SG variables, where each key is the variable name and each value is the corresponding NumPy array.
    condIm_var : dict
        Dictionary with conditioning image variables, where each key is the variable name and each value is the corresponding NumPy array.
    outputFlag : dict
        dictionary with boolean, where each key is the varaible name and each value is the boolean precising if the varaible should be retrieved in the output.

    names_var : list of lists
        A list of four sublists, where each sublist contains the names of the variables for each category:
        - `names_var[0]`: List of names of simulated variables.
        - `names_var[1]`: List of names of auxiliary TI variables.
        - `names_var[2]`: List of names of auxiliary SG variables.
        - `names_var[3]`: List of names of conditioning image variables.

    types_var : list of lists
        A list of four sublists, where each sublist contains the types (categorical or continuous) of the variables for each category:
        - `types_var[0]`: List of types of simulated variables.
        - `types_var[1]`: List of types of auxiliary TI variables.
        - `types_var[2]`: List of types of auxiliary SG variables.
        - `types_var[3]`: List of types of conditioning image variables.

    Raises:
    ------
    NameError
        If any variable is missing a name ('nan') in the CSV file.
    NameError
        If two variables of the same nature have the same name.
    ValueError
        If any variable has an invalid nature (not 'sim', 'auxTI', 'auxSG', or 'condIm').

    Notes:
    -----
    - The function assumes that each variable has a unique and valid name.
    - The nature of the variable determines its category, which can be one of the following:
        - "sim" for a simulated variable.
        - "auxTI" for an auxiliary variable describing the simulated variable(s) in the TI.
        - "auxSG" for an auxiliary variable conditioning the variability of the simulated variable(s) in the simulation grid.
        - "condIm" for a conditioning image variable.
    """
    
    data_df = pd.read_csv(csv_file_path, sep=';')
    
    sim_var = {}
    auxTI_var = {}
    auxSG_var = {}
    condIm_var = {}
    outputFlag = {} #To precise which variables will be retrieved in the output

    names_var = [[], [], [], []]
    types_var = [[], [], [], []]
    

    for i, row in data_df.iterrows():
        var_name = str(row['var_name'])
        categ_conti = row['categ_conti']
        nature = row['nature']
        path = row['path']
        grid = row['grid']
        array_data = np.load(path)
        
        #SIMULATED
        if nature == 'sim':
            if var_name != 'nan' :
                if var_name not in names_var[0]:
                    names_var[0].append(var_name)
                    sim_var[var_name] = array_data
                    types_var[0].append(categ_conti)
                    outputFlag[var_name] = True
                else:
                    raise NameError(f"Line {i+1} of the CSV file : Two simulated variables have the same name, please consider naming all of your variables with different names.")
            else:
                raise NameError(f"Line {i+1} of the CSV file : One simulated variable has no name, please consider naming all of your variables with different names.")
        
        #AUXTI
        elif nature == 'auxTI':
            if var_name != 'nan' :
                if var_name not in names_var[1]:
                    names_var[1].append(var_name)
                    auxTI_var[var_name] = array_data
                    types_var[1].append(categ_conti)
                    outputFlag[var_name] = False
                else:
                    raise NameError(f"Line {i+1} of the CSV file : Two auxiliary TI variables have the same name, please consider naming all of your variables with different names.")
            else:
                raise NameError(f"Line {i+1} of the CSV file : One auxiliary TI variable has no name, please consider naming all of your variables with different names.")   
        
        #AUXSG
        elif nature == 'auxSG':
            if var_name != 'nan' :
                if var_name not in names_var[2]:
                    names_var[2].append(var_name)
                    auxSG_var[var_name] = array_data
                    types_var[2].append(categ_conti)
                else:
                    raise NameError(f"Line {i+1} of the CSV file : Two auxiliary SG variables have the same name, please consider naming all of your variables with different names.")
            else:
                raise NameError(f"Line {i+1} of the CSV file : One auxiliary SG variable has no name, please consider naming all of your variables with different names.")    
            
        #CONDIM
        elif nature == 'condIm':
            if var_name != 'nan' :
                if var_name not in names_var[3]:
                    names_var[3].append(var_name)
                    condIm_var[var_name] = array_data
                    types_var[3].append(categ_conti)
                else:
                    raise NameError(f"Line {i+1} of the CSV file : Two condtioning image variables have the same name, please consider naming all of your variables with different names.")
            else:
                raise NameError(f"Line {i+1} of the CSV file : One conditioning image variable has no name, please consider naming all of your variables with different names.")
        
        else: 
            raise ValueError(f"Line {i+1} of the CSV file : One variable has an invalid nature ({nature}), please consider chosing between the following natures: \
                            \n    - \"sim\" for a simulated variable;\
                            \n    - \"auxTI\" for an auxiliary variable describing the simulated variable(s) in the TI;\
                            \n    - \"auxSG\" for an auxiliary variable conditioning the variability of the simulated variable(s) in the simulation grid;\
                            \n    - \"condIm\" for a conditioning image variable.")
                        
    return sim_var, auxTI_var, auxSG_var, condIm_var, names_var, types_var, outputFlag
 

def check_variables(sim_var, auxTI_var, auxSG_var, condIm_var, names_var, types_var, novalue=-9999999):
    """
    Validate the types, consistency, and dimensions of variables, and process missing values.
    This function performs the following tasks:
    
    1. **Type Validation**: 
       - Ensures that the data type of each variable (simulated, auxiliary TI, auxiliary SG, and conditioning image) matches its expected type ('continuous' or 'categorical').
    2. **Missing Value Handling**: 
       - Replaces any instances of a specified `novalue` in the variables with `np.nan`.
    3. **Shape Consistency**: 
       - Checks that all variables have consistent dimensions (XY) across their respective categories.
    4. **Name Validation**:
       - Confirms that each conditioning variable name matches a corresponding simulated variable name.
       - Ensures that each auxiliary TI variable has a corresponding auxiliary SG variable.

    Parameters:
    ----------
    sim_var : dict
        Dictionary containing simulated variables, where each key is a variable name and each value is a NumPy array.
    auxTI_var : dict
        Dictionary containing auxiliary TI variables.
    auxSG_var : dict
        Dictionary containing auxiliary SG variables.
    cond_var : dict
        Dictionary containing conditioning iamge variables.
    names_var : list of lists
        A list containing four sub-lists:
        - `names_var[0]`: Names of simulated variables.
        - `names_var[1]`: Names of auxTI variables.
        - `names_var[2]`: Names of auxSG variables.
        - `names_var[3]`: Names of conditioning image variables.
    types_var : list of lists
        A list containing four sub-lists:
        - `types_var[0]`: Expected types ('continuous' or 'categorical') of simulated variables.
        - `types_var[1]`: Expected types of auxTI variables.
        - `types_var[2]`: Expected types of auxSG variables.
        - `types_var[3]`: Expected types of conditioning image variables.
    novalue : scalar
        The value used to represent missing data in the variables, which will be replaced with `np.nan`.

    Returns:
    -------
    sim_var : dict
        A dictionary containing the validated and processed simulated variables.
    auxTI_var : dict
        A dictionary containing the validated and processed auxiliary TI variables.
    auxSG_var : dict
        A dictionary containing the validated and processed auxiliary SG variables.
    condIm_var : dict
        A dictionary containing the validated and processed conditioning image variables.

    Raises:
    ------
    KeyError
        If there is no simulated variable.
    TypeError
        If the data type of a variable does not match its expected type.
    ValueError
        If an invalid type is specified, or if variables do not have consistent dimensions or if one of the auxiliary variable provided don't have the same range of value in the TI and in the SG.
    NameError
        If a conditioning variable's name does not match any simulated variable, or if an auxSG variable lacks a corresponding auxTI variable.

    Notes:
    -----
    - This function assumes that each variable has a valid name and type.
    - It is important that the dimensions of all variables within their respective categories (e.g., simulated, auxiliary) are consistent.
    """
    
    # Check sim_var and aux_var exists
    if len(sim_var) == 0:
        raise KeyError(f"No simulated variable was provided. Please select at least one simulated variable.")
    
    if len(auxSG_var) == 0:
        raise KeyError(f"No auxiliary variable was provided. Please select at least one auxiliary variable.")
    
    # Check types
    for var_i in range(len(names_var[0])):
        varname_sim = names_var[0][var_i]
        sim_array = sim_var[varname_sim]
        expected_type = types_var[0][var_i]
        if expected_type == "continuous":
            if not np.issubdtype(sim_array.dtype, np.number):
                raise TypeError(f"Type mismatch for sim var '{varname_sim}'. Expected numerical type for 'continuous', got {sim_array.dtype}.")    
        elif expected_type == "categorical":
            if not np.issubdtype(sim_array.dtype, np.integer):
                raise TypeError(f"Type mismatch for sim var '{varname_sim}'. Expected integer type for 'categorical', got {sim_array.dtype}.")    
        else:
            raise ValueError(f"Invalid type for {varname_sim} '{expected_type}' specified. Expected 'continuous' or 'categorical'.")
                
    for var_i in range(len(names_var[1])):
        varname_auxTI = names_var[1][var_i]
        auxTI_array = auxTI_var[varname_auxTI]
        expected_type = types_var[1][var_i]
        if expected_type == "continuous":
            if not np.issubdtype(auxTI_array.dtype, np.number):
                raise TypeError(f"Type mismatch for auxTI var '{varname_auxTI}'. Expected numerical type for 'continuous', got {auxTI_array.dtype}.")    
        elif expected_type == "categorical":
            if not np.issubdtype(auxTI_array.dtype, np.integer):
                raise TypeError(f"Type mismatch for auxTI var '{varname_auxTI}'. Expected integer type for 'categorical', got {auxTI_array.dtype}.")    
        else:
            raise ValueError(f"Invalid type for {varname_auxTI} '{expected_type}' specified. Expected 'continuous' or 'categorical'.")
    
    for var_i in range(len(names_var[2])):
        varname_auxSG = names_var[2][var_i]
        auxSG_array = auxSG_var[varname_auxSG]
        expected_type = types_var[2][var_i]
        if expected_type == "continuous":
            if not np.issubdtype(auxSG_array.dtype, np.number):
                raise TypeError(f"Type mismatch for auxSG var '{varname_auxSG}'. Expected numerical type for 'continuous', got {auxSG_array.dtype}.")    
        elif expected_type == "categorical":
            if not np.issubdtype(auxSG_array.dtype, np.integer):
                raise TypeError(f"Type mismatch for auxSG var '{varname_auxSG}'. Expected integer type for 'categorical', got {auxSG_array.dtype}.")    
        else:
            raise ValueError(f"Invalid type for {varname_auxSG} '{expected_type}' specified. Expected 'continuous' or 'categorical'.")
            
    for var_i in range(len(names_var[3])):
        varname_condIm = names_var[3][var_i]
        condIm_array = condIm_var[varname_condIm]
        expected_type = types_var[3][var_i]
        if expected_type == "continuous":
            if not np.issubdtype(condIm_array.dtype, np.number):
                raise TypeError(f"Type mismatch for condIm var '{varname_condIm}'. Expected numerical type for 'continuous', got {condIm_array.dtype}.")    
        elif expected_type == "categorical":
            if not np.issubdtype(condIm_array.dtype, np.integer):
                raise TypeError(f"Type mismatch for condIm var '{varname_condIm}'. Expected integer type for 'categorical', got {condIm_array.dtype}.")    
        else:
            raise ValueError(f"Invalid type for {varname_condIm} '{expected_type}' specified. Expected 'continuous' or 'categorical'.")
    
    first_sim = next(iter(sim_var))
    shape_TI = sim_var[first_sim].shape
                    
    # Replace novalue by "nan" and check dimensions XY
    for key in sim_var:
        if np.any(sim_var[key] == novalue):
            sim_var[key] = np.where(sim_var[key] == novalue, np.nan, sim_var[key])
        if sim_var[key].shape != shape_TI:
                raise ValueError(f"Simulated variable does not have the same dimensions XY for '{first_sim}' and '{key}'.")
    for key in auxTI_var:
        if np.any(auxTI_var[key] == novalue):
            auxTI_var[key] = np.where(auxTI_var[key] == novalue, np.nan, auxTI_var[key])
        if auxTI_var[key].shape != shape_TI:
                raise ValueError(f"Auxiliary TI variable does not have the same dimensions XY for '{first_var}' and '{key}'.")
    
    first_aux = next(iter(auxSG_var))
    shape_SG = auxSG_var[first_aux].shape
    
    for key in auxSG_var:
        if np.any(auxSG_var[key] == novalue):
            auxSG_var[key] = np.where(auxSG_var[key] == novalue, np.nan, auxSG_var[key])
        if auxSG_var[key].shape != shape_SG:
                raise ValueError(f"Auxilliary SG variable does not have the same dimensions XY for '{first_aux}' and '{key}'.")
    for key in condIm_var:
        if np.any(condIm_var[key] == novalue):
            condIm_var[key] = np.where(condIm_var[key] == novalue, np.nan, condIm_var[key])
        if condIm_var[key].shape != shape_SG:
                raise ValueError(f"Conditioning image variable does not have the same dimensions XY for '{first_aux}' and '{key}'.")
    
    #Check that the names of the conditioning data are present in the list of the names of the simulated variables
    for name_condImvar in names_var[3]:
        if name_condImvar not in names_var[0]:
            raise NameError(f"The name of the conditioning image variable '{name_condImvar}' does not match any of the simulated variables. Conditioning variables must be named according to their corresponding simulated variables.")
    
    #Check that each auxTI has one auxSG
    for name_auxTIvar in names_var[1]:
        if name_auxTIvar not in names_var[2] :
            raise NameError(f"The auxiliary TI variable '{name_auxTIvar}' has not matching auxSG_var. All auxiliary variables must be TI and conditioning.")
    
    #Check that the range of the values of each auxTI is the same as its corresponding auxSG
    for name_auxvar in names_var[1]:
        if np.nanmin(auxTI_var[name_auxvar]) != np.nanmin(auxSG_var[name_auxvar]) or np.nanmax(auxTI_var[name_auxvar]) != np.nanmax(auxSG_var[name_auxvar]):
            raise ValueError(f"The auxiliary variable {name_auxvar} has not the same min and max value in the TI and in the SG. An auxiliary variable must have the same range of value in the TI and in the SG.")
    
    return sim_var, auxTI_var, auxSG_var, condIm_var

 
def count_variables(names_var):
    """
    Count the total number of unique variables based on the names provided in names_var.

    Parameters:
    ----------
    names_var : list of lists
        A list of four sublists, where each sublist contains the names of the variables for each category:
        - `names_var[0]`: List of names of simulated variables.
        - `names_var[1]`: List of names of auxiliary TI variables.
        - `names_var[2]`: List of names of auxiliary SG variables.
        - `names_var[3]`: List of names of conditioning image variables.

    Returns:
    -------
    int
        The total number of unique variables across all categories.
    """
    unique_variables = set()
    
    for category in names_var:
        unique_variables.update(category)
    
    return len(unique_variables)


def get_sim_grid_dimensions(auxTI_var):
    ar  = auxTI_var[next(iter(auxTI_var))]
    if ar is not None:
        nc, nr = ar.shape[0], ar.shape[1]
    else:
        raise ValueError(f"No auxiliary variable was provided, please provide at least one auxiliary variable to constrain the shape of the simulation grid")
    return nc, nr


def get_unique_names_and_types(names_var, types_var):
    """
    Create a list of all unique variable names and a corresponding list of their types.

    Parameters:
    ----------
    names_var : list of lists
        A list of lists, where each sublist contains the names of the variables for each category.
    types_var : list of lists
        A list of lists, where each sublist contains the types (categorical or continuous) of the variables for each category.

    Returns:
    -------
    unique_names : list
        A list of unique variable names.
    unique_types : list
        A list of types corresponding to the unique variable names.
    """
    unique_names = []
    unique_types = []

    for name_list, type_list in zip(names_var, types_var):
        for name, var_type in zip(name_list, type_list):
            if name not in unique_names:
                unique_names.append(name)
                unique_types.append(var_type)

    return unique_names, unique_types

      

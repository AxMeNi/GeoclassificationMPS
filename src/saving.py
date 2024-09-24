# -*- coding:utf-8 -*-
__projet__ = "GeoclassificationMPS"
__nom_fichier__ = "display_functions"
__author__ = "MENGELLE Axel"
__date__ = "septembre 2024"

import pandas as pd
from datetime import datetime
import os
import pickle


def save_deesse_output(deesse_output, output_dir, file_name):
    """
    Save the deesse_output to a specified folder.

    Parameters:
    -----------
    deesse_output : dict
        The output from the Deesse simulation that you want to save.
    
    output_dir : str
        The directory where you want to save the output.
    
    file_name : str
        The name of the file (without extension) to save the output as.

    Returns:
    --------
    None
    """

    # Ensure the output directory exists, if not create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Full path to the file
    file_path = os.path.join(output_dir, file_name + '.pkl')
    
    # Save the deesse_output to the file using pickle
    with open(file_path, 'wb') as file:
        pickle.dump(deesse_output, file)
    
    print(f"Deesse output successfully saved to {file_path}")

def save_simulation(deesse_output, params, comments="", output_directory="output/"):
    """
    Saves the Deesse output and updates an Excel file with simulation parameters.
    
    Args:
        deesse_output (dict): The output from Deesse simulation.
        params (dict): Dictionary of parameters used for the simulation.
        comments (str): Additional comments about the simulation.
        output_directory (str): Directory where the output files will be saved.
        
    Returns:
        None
    """
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Save Deesse output
    output_file_path = os.path.join(output_directory, f"deesse_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
    save_deesse_output(deesse_output, output_file_path)

    # Prepare data for Excel
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data = {
        'Date and Time': [now],
        'File Path': [output_file_path],
        'Comments': [comments]
    }

    # Add parameters to the data
    for key, value in params.items():
        data[key] = [value]

    # Define the path to the Excel file
    excel_file_path = os.path.join(output_directory, 'simulation_log.xlsx')

    # Check if the Excel file exists
    if os.path.exists(excel_file_path):
        # Load existing Excel file
        df_existing = pd.read_excel(excel_file_path)
    else:
        # Create a new DataFrame
        df_existing = pd.DataFrame()

    # Create a new DataFrame with the current simulation data
    df_new = pd.DataFrame(data)

    # Append the new data to the existing DataFrame
    df_updated = pd.concat([df_existing, df_new], ignore_index=True)

    # Save the updated DataFrame to Excel
    df_updated.to_excel(excel_file_path, index=False)
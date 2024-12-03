# -*- coding:utf-8 -*-
__projet__ = "GeoclassificationMPS"
__nom_fichier__ = "interface"
__author__ = "MENGELLE Axel"
__date__ = "novembre 2024"


import time
import pandas as pd



log_df = pd.DataFrame(columns=['Process', 'Start_Time', 'End_Time', 'Duration'])


def start_timer(process_name):
    """
    Start the timer for a given process and return the start time.

    This function records the current time at the moment the process starts
    and returns the process name along with the start time, which can later
    be used to calculate the duration of the process.

    Parameters:
    ----------
    process_name : str
        The name or description of the process being timed. This is used
        for tracking and logging purposes.

    Returns:
    -------
    (process_name, start_time), tuple
        A tuple containing the process name and the start time (as a float),
        which can be passed to `end_timer_and_log` for duration calculation.
    """
    
    start_time = time.time()
    return process_name, start_time


def end_timer_and_log(start_tuple, log_df):
    """
    End the timer, calculate the duration, and log the process details in a DataFrame.

    This function takes the process name and start time from `start_tuple`, calculates
    the duration of the process, and appends the details into a provided DataFrame
    (`log_df`). The updated `log_df` is returned for further use.

    Parameters:
    ----------
    start_tuple : tuple
        A tuple containing the process name (str) and the start time (float) from the `start_timer` function.
    log_df : pandas.DataFrame
        The DataFrame that holds the log entries. The function will append a new log entry to this DataFrame.

    Returns:
    -------
    log_df : pandas.DataFrame
        The updated DataFrame containing the new log entry.
    """
    
    end_time = time.time()
    process_name, start_time = start_tuple
    duration = end_time - start_time
    log_entry = pd.DataFrame([{
        'Process': process_name,
        'Start_Time': time.ctime(start_time),
        'End_Time': time.ctime(end_time),
        'Duration': duration
    }])
    log_df = pd.concat([log_df, log_entry], ignore_index=True)
    return log_df


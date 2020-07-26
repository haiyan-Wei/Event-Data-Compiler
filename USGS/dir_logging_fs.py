import os
import sys
import logging
import shutil
import inspect
import pandas as pd


def makefolders(folders):
    for folder in (folders):
        if not os.path.exists(folder):
            os.makedirs(folder)


def logInputs(variables):

    logging.info('\n\n========= Inputs =========\n')

    for var in variables:
        callers_local_vars = inspect.currentframe().f_back.f_locals.items()
        logging.info([var_name for var_name, var_val in callers_local_vars if var_val is var][0] + ': ' + str(var))
    logging.info('\n\n')


def logINFO(logname, script_goal, t0):
    descrp = 'This is a log file created with the following script. '
    descrp += 'Please turn off the option of "wrap text" to view.\n'
    logging.basicConfig(filename=logname, level=logging.INFO, format='%(message)s', filemode='w')
    script_loc = sys.argv[0]
    logging.info(f'Description: {descrp}\nScript location: {script_loc}\nObjectives:\n{script_goal}')
    logging.info(f'\n\nStarted at: {t0}\n\n')


def reduce_df_size(df):

    """ this function reduces the memory usage by changing data types.
        It does not change the values in data frame """

    optimized_df = df.copy()
    df_int = df.select_dtypes(include=['int64'])
    df_float = df.select_dtypes(include=['float'])
    converted_int = df_int.apply(pd.to_numeric, downcast='unsigned')
    converted_float = df_float.apply(pd.to_numeric,downcast='float')
    optimized_df[converted_int.columns] = converted_int
    optimized_df[converted_float.columns] = converted_float

    return optimized_df


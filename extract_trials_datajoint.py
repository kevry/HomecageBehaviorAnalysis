# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 14:59:04 2023

@author: Kevin Delgado

Query all necessary data from datajoint about animal
"""

import datajoint as dj
import datetime
import os
import pandas as pd
import stat
import time
from tqdm import tqdm

# get schema variables and spawn missing classes
experiment_schema = dj.Schema('homecage_experiment')

@experiment_schema
class SessionTrial(dj.Manual):
    pass
@experiment_schema
class Session(dj.Manual):
    pass

def get_mat_file_path(training_module_id, trial_datetime):
    """ get full path to trial mat file using training module id and trial datetime """
    
    # Note: This can be hardcoded for now ... training module data folder
    MAIN_MAT_FILE_FOLDER = r'Z:\Projects\Homecage\DLCVideos\trainingmodule_matfiles'
    
    if isinstance(trial_datetime, str):
        TRIAL_DATETIME_DT = datetime.datetime.strptime(trial_datetime[:-4], '%Y-%m-%d %H:%M:%S')
    else:
        TRIAL_DATETIME_DT = trial_datetime
    
    # video timestamps and datajoint trial datetimes are A BIT off. Adding +- 4 seconds of room
    seconds_range = 4
    MIN_DT, MAX_DT = TRIAL_DATETIME_DT - datetime.timedelta(seconds=seconds_range), TRIAL_DATETIME_DT + datetime.timedelta(seconds=seconds_range)
    
    TRIAL_DATE_STR = TRIAL_DATETIME_DT.strftime('%Y%m%d')
    TRIAL_HOUR = TRIAL_DATETIME_DT.strftime('%H')

    TM_DATE_FOLDER = os.path.join(MAIN_MAT_FILE_FOLDER, "TM_{}".format(str(training_module_id)), TRIAL_DATE_STR)
    if not os.path.exists(TM_DATE_FOLDER):
        return False, None
        
    # get all video folders from tm_date_folder
    VIDEO_FOLDER_NAME = None
    for folder in os.listdir(TM_DATE_FOLDER):
        _, _, video_hour, *_ = folder.split("_")
        if video_hour[:2] == TRIAL_HOUR:
            VIDEO_FOLDER_NAME = folder
            break
    if VIDEO_FOLDER_NAME == None:
        return False, None
    
    # full path to video folder
    VIDEO_FOLDER_PATH = os.path.join(TM_DATE_FOLDER, VIDEO_FOLDER_NAME)

    # loop through list of .mat files
    mat_file_found = False
    for file in [file for file in os.listdir(VIDEO_FOLDER_PATH) if '.mat' in file]:
        _, trial_dt = file[:-4].split("_")
        # convert file name to datetime
        FILE_DATETIME_DT = datetime.datetime.strptime(trial_dt, '%Y%m%d%H%M%S000')
        # check if mat file trial datetime is within range
        if MIN_DT <= FILE_DATETIME_DT <= MAX_DT:
            mat_file_found = True
            mat_file_path = os.path.join(VIDEO_FOLDER_PATH, file)
            break
            
    if mat_file_found == False:
        return False, None
    
    return True, mat_file_path
    


def extract_trials_datajoint(animalRFID, animal_folder, save_missing_trials=True, overwrite=False):
    """ get all trials with respective .mat files from DataJoint """
    print("Extracting trials with respective mat file from database.")

    
    found_csv_path = os.path.join(animal_folder, "FOUND_TRIALS.csv")
    if overwrite == True:
        if os.path.exists(found_csv_path):
            print("\tOverwritting FOUND_TRIALS.csv for {}".format(animalRFID))
    else:
        if os.path.exists(found_csv_path):
            print("\tFound FOUND_TRIALS.csv for {}".format(animalRFID))
            return found_csv_path
        
    # query all trials associated with animal and training_module_id it belongs to
    trial_list = (SessionTrial & 'subject = "{}"'.format(animalRFID)).fetch(as_dict = True, order_by='trial_datetime')
    if len(trial_list) == 0:
        print("\tNo trials found for {} in database".format(animalRFID))
        return None
    
    training_module_id = (Session & 'session_datetime = "{}"'.format(trial_list[0]['session_datetime'])).fetch(as_dict=True)[0]['training_module_id']
    
    print("\t-- {} --".format(animalRFID))
    print("\tTraining Module ID:", str(training_module_id))
    print("\tInitial trial datetime:", trial_list[0]["trial_datetime"])
    print("\tNumber of trials queried:", len(trial_list))

    trials_found = []
    trials_missing = []
       
    for trial_data in tqdm(trial_list, desc="\tFinding mat files"):
        status, mat_file_path = get_mat_file_path(training_module_id, trial_data["trial_datetime"])
        if status:
            trials_found.append([trial_data["trial_datetime"], mat_file_path])
        else:
            trials_missing.append([trial_data["trial_datetime"], mat_file_path])
    
    print("\tNumber of trials found with +-4sec method: {}/{}".format(str(len(trials_found)), str(len(trial_list))))
    
    found_csv_path = os.path.join(animal_folder, "FOUND_TRIALS.csv")
    df = pd.DataFrame(trials_found, columns=["trial_datetime", "mat_file_path"])
    df.to_csv(found_csv_path, index=False)
    os.chmod(found_csv_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    
    if save_missing_trials:
        missing_csv_path = os.path.join(animal_folder, "MISSING_TRIALS.csv")
        df = pd.DataFrame(trials_missing, columns=["trial_datetime", "mat_file_path"])
        df.to_csv(missing_csv_path, index=False)
        os.chmod(missing_csv_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        print("\tCreated FOUND_TRIALS.csv and MISSING_TRIALS.csv for {}!".format(animalRFID))
    else:
        print("\tCreated FOUND_TRIALS.csv for {}!".format(animalRFID))
        
    return found_csv_path



if __name__ == "__main__":
    pass
        
        
        
        
        
        
        
        
        
        
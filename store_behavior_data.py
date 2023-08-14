# -*- coding: utf-8 -*-
"""
Created on Mon Aug 7 17:25:53 2023
@author: Kevin Delgado
Split up animals that went through behavioral analysis by rig number to save data
"""

import argparse
from chenlabpylib import chenlab_filepaths, send_slack_notification
import datetime
import glob
import json
import numpy as np
import os
import pandas as pd
from scipy.io import loadmat, savemat
import stat
import sys
from tqdm import tqdm
import traceback
import yaml


def get_args():
    """ gets arguments from command line """
    parser = argparse.ArgumentParser(
        description="Parsing argument for animal ID",
        epilog="python file.py --config_file_path configs/config_test.yaml"
    )
    # required argument
    parser.add_argument("--json_file_name", '-jfp', required=False, help='Name of json file with animal list')
    parser.add_argument("--config_file_name", '-cfg', required=True, help='Name of config file with configurations')
    args = parser.parse_args()
    return args.json_file_name, args.config_file_name


if __name__ == "__main__":
    # get configuration file path
    json_file_name, config_file_name = get_args()
    
    with open(os.path.join(os.getcwd(), "configs", config_file_name), "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            sys.exit(exc)
    
    processing_folder = cfg["processing_folder"]    
    version = cfg["motion_mapper_version"]
    auto_encoder_model_path = cfg["motion_mapper_file_paths"]["auto_encoder_model_path"]
    umap_model_path = cfg["motion_mapper_file_paths"]["umap_model_path"]
    
    animal_folder_list = glob.glob(os.path.join(processing_folder, "*"))
    
    if sys.platform == 'linux': # assume running on SCC
        # load in JSON file
        f = open(os.path.join(os.getcwd(), "scc", "jsons", json_file_name))
        animal_rig_dict = json.load(f)
        f.close()
            
        task_id = int(os.environ["SGE_TASK_ID"])
        training_module_id = list(animal_rig_dict.keys())[task_id]
        animalRFIDlist = animal_rig_dict[training_module_id]
        print("Storing data for training module:", training_module_id)
        
    else: # running on local Windows lab computers
        animalRFIDlist = [str(animalRFID)for animalRFID in cfg["animal_list"]] # make sure all RFIDs are strings
        print("Storing data for all animals in config.")
        
    # %% Store animal data    
    for anm_idx, animal in enumerate(animalRFIDlist):
        animal_folder = os.path.join(processing_folder, animal)
    
        subject_name = os.path.basename(animal_folder)
        print("{}/{}:".format(str(anm_idx+1), str(len(animal_folder_list))), subject_name)
        
        found_trials_csv_file = os.path.join(animal_folder, "FOUND_TRIALS.csv")
        post_analyzed_dlc_file = os.path.join(animal_folder, "POST_ANALYZED_DLC.npz")
        encoded_pose_data_file = os.path.join(animal_folder, "ENCODED_POSE_DATA.mat")
        wavelets_file = os.path.join(animal_folder, "WAVELETS.mat")
        umap2d_file = os.path.join(animal_folder, "UMAP2D.mat")
        watershedregions_file = os.path.join(animal_folder, "WATERSHEDREGIONS.mat")
    
        try:
            # make sure all files exist
            for path in [found_trials_csv_file, post_analyzed_dlc_file, encoded_pose_data_file, wavelets_file, umap2d_file, watershedregions_file]:
                assert os.path.exists(path) == True
                
            # get all mat files form found_trials.csv
            trial_data = pd.read_csv(found_trials_csv_file).values
            full_mat_file_list = [chenlab_filepaths(path=trial[1]) for trial in trial_data]
            mat_files_not_used = full_mat_file_list.copy()
                    
            # load in POST_ANALYZED_DLC_DATA.npz
            data = np.load(chenlab_filepaths(path=post_analyzed_dlc_file))
            subject_data = data["data"]
            mat_file_list = data["mat_files"]
            per_trial_length = data["per_trial_length"]
            start_end_indexes = data["start_end_indexes"]
            
            post_processed_dlc_data_batched = np.split(subject_data, np.cumsum(per_trial_length)[:-1])
            
            # load in motionmapper inference data
            encoded_pose_data = loadmat(encoded_pose_data_file)["data"]
            wavelets = loadmat(wavelets_file)["data"]
            umap2d = loadmat(umap2d_file)["data"]
            watershedregions = loadmat(watershedregions_file)["data"]
            
            assert len(encoded_pose_data) == len(wavelets) == len(umap2d) == len(watershedregions)
            
            encoded_pose_data_batched = np.split(encoded_pose_data, np.cumsum(per_trial_length)[:-1])
            wavelets_batched = np.split(wavelets, np.cumsum(per_trial_length)[:-1])
            embedded2ddata_batched = np.split(umap2d, np.cumsum(per_trial_length)[:-1])
            watershedRegions_batched = np.split(watershedregions, np.cumsum(per_trial_length)[:-1])
            
            assert len(post_processed_dlc_data_batched) == len(mat_file_list) == len(per_trial_length) == len(start_end_indexes)
            assert len(encoded_pose_data_batched) == len(wavelets_batched) == len(embedded2ddata_batched) == len(watershedRegions_batched)
            assert len(post_processed_dlc_data_batched) == len(watershedRegions_batched)
            
            print("\tNumber of trial .mat files:", len(mat_file_list))
            for i, mat_file in enumerate(tqdm(mat_file_list, desc="\tSaving data to mat files")):
                mat_file = chenlab_filepaths(path=mat_file)
                
                mat_files_not_used = [file for file in mat_files_not_used if file != mat_file]
                
                matdata = loadmat(mat_file)
                
                matdata.pop("motion_mapper_analyzed_date", None)
                
                # save post analyzed DLC data
                matdata["post_processed_success"] = True
                matdata["egocentricwTM"] = post_processed_dlc_data_batched[i]
                matdata["egocentric_start_end_index"] = start_end_indexes[i]
                
                # save motionmapper inference data
                matdata["encoded_egocentricwTM"] = post_processed_dlc_data_batched[i]
                matdata["wavelets"] = wavelets_batched[i]
                matdata["embedded2D"] = embedded2ddata_batched[i]
                matdata["watershedregions"] = watershedRegions_batched[i]
                matdata["umap_model_path"] = umap_model_path
                matdata["autoencoder_model_path"] = auto_encoder_model_path
                matdata["motion_mapper_analyzed_ver"] = version
                matdata["behavioral_analysis_date"] = datetime.datetime.today().strftime('%m/%d/%Y, %H:%M:%S')
                savemat(mat_file, matdata)
                try: # attempt to chmod 777, continue even if fails
                    os.chmod(mat_file, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                except:
                    pass
                
            # for all mat files not used, set post processed success = False in mat files 
            print("\tNumber of trial .mat files not used:", len(mat_files_not_used))
            for mat_file in tqdm(mat_files_not_used, "\tMarking mat files not used"):
                mat_file = chenlab_filepaths(path=mat_file)
                matdata = loadmat(mat_file)
                matdata.pop("motion_mapper_analyzed_date", None)
                matdata["post_processed_success"] = False
                savemat(mat_file, matdata)
                try: # attempt to chmod 777, continue even if fails
                    os.chmod(mat_file, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                except:
                    pass
            
            send_slack_notification(message="BEHAVIOR DATA SAVED for {}".format(subject_name))
            
        except KeyError or FileNotFoundError or AssertionError:
            traceback.print_exc()
            send_slack_notification(message="ERROR SAVING BEHAVIOR DATA for {}".format(subject_name))
            continue

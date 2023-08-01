# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 15:01:17 2023

@author: Kevin Delgado
"""

import argparse
import time
import numpy as np
import os
import sys
import yaml

import ChenLabPyLib

def get_args():
    """ gets arguments from command line """
    parser = argparse.ArgumentParser(
        description="Parsing argument for config file path",
        epilog="python file.py --config_file_path configs/config_test.yaml"
    )
    # required argument
    parser.add_argument("--config_file_name", '-cfg', required=True, help='Name of configuration file')
    args = parser.parse_args()
    return args.config_file_name


# %%
if __name__ == "__main__":
    
    from extract_trials_datajoint import extract_trials_datajoint
    from post_processing_dlc.post_processing_dlc import PostAnalysisDLC
    from motionmapper.inference import MotionMapperInference
    
    # get configuration file path
    config_file_name = get_args()
    
    with open(os.path.join(os.getcwd(), 'configs', config_file_name), "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            sys.exit(exc)
    
    # create instance of MotionMapperInference object
    mminfer = MotionMapperInference(
        umap_model_path = ChenLabPyLib.chenlab_filepaths(path = cfg["motion_mapper_file_paths"]["umap_model_path"]), 
        auto_encoder_model_path = ChenLabPyLib.chenlab_filepaths(path = cfg["motion_mapper_file_paths"]["auto_encoder_model_path"]), 
        scaling_parameters_path = ChenLabPyLib.chenlab_filepaths(path = cfg["motion_mapper_file_paths"]["scaling_parameters_path"]),
        look_up_table_path = ChenLabPyLib.chenlab_filepaths(path = cfg["motion_mapper_file_paths"]["look_up_table_path"]),
        watershed_file_path = ChenLabPyLib.chenlab_filepaths(path = cfg["motion_mapper_file_paths"]["watershed_file_path"])
    )
    
    # create instance of Post-Process DLC object
    postdlc = PostAnalysisDLC(
        nose2tail_ae_path = ChenLabPyLib.chenlab_filepaths(path = cfg["post_processing_dlc_paths"]["nose2tail_ae_path"]), 
        feet_ae_path = ChenLabPyLib.chenlab_filepaths(path = cfg["post_processing_dlc_paths"]["feet_ae_path"]), 
        all_ae_path = ChenLabPyLib.chenlab_filepaths(path = cfg["post_processing_dlc_paths"]["all_ae_path"])
    )
    
    start_time = time.time()
    
    # %% Animal information
    
    animalRFIDlist = cfg["animal_list"]
    
    for animalRFID in animalRFIDlist:
        print("Analyzing", animalRFID)
        
        # animal folder
        animal_folder = os.path.join(ChenLabPyLib.chenlab_filepaths(path = cfg["processing_folder"]), animalRFID)
        os.makedirs(animal_folder, exist_ok=True)
        
        # Get all trials from respective animal on DataJoint
        found_trials_csv_path = extract_trials_datajoint(
            animalRFID=animalRFID, 
            animal_folder=animal_folder,
            datajoint_credentials=cfg["datajoint_credentials"],
            save_missing_trials=False, 
            overwrite=True
        )
        
        # %% Post-Processing DeepLabCut data for downstream behavior analysis
        post_analyzed_dlc_file_path = postdlc.run(
            csv_path=found_trials_csv_path, 
            animalRFID=animalRFID, 
            animal_folder=animal_folder,
            overwrite=True,
            save2trialmat=False
        )
    
        # %% Load in and extract post-processed DeepLabCut data
        raw_data = np.load(post_analyzed_dlc_file_path)
        projections = raw_data['data']
        per_trial_length = raw_data['per_trial_length']
        mat_files_used = raw_data['mat_files']
        
        projections_flatten = projections.reshape((-1, 36))
        
        print("Pose data reshaped:", projections_flatten.shape)
        print("Pose data dim:", projections.shape)
        
        del raw_data, projections
        
        # %% Run MotionMapper process
        mminfer.run(
            pose_data=projections_flatten, 
            per_trial_length=per_trial_length,
            mat_files_used=mat_files_used,
            animalRFID=animalRFID, 
            animal_folder=animal_folder,
            sigma=0.9,
            save_progress=True,
            save2trialmat=False
        )
        
        # %%
        print("Elapsed time:", time.time() - start_time)
        
        # %% Send Slack Notification when finished
        ChenLabPyLib.send_slack_notification(message="MotionMapper inference w/ {} finished".format(animalRFID))

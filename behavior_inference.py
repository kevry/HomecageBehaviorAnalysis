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
    
    
    from post_processing_dlc.post_processing_dlc import PostAnalysisDLC
    from motionmapper.inference import MotionMapperInference
    
    # get configuration file path
    config_file_name = get_args()
    
    with open(os.path.join(os.getcwd(), 'configs', config_file_name), "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            sys.exit(exc)
            
            
    # import datajoint and credentials
    import datajoint as dj
    dj.config['database.host'] = cfg["datajoint_credentials"]['host']
    dj.config['database.user'] = cfg["datajoint_credentials"]['user']
    dj.config['database.password'] = cfg["datajoint_credentials"]['password']
    from extract_trials_datajoint import extract_trials_datajoint
    
    # create instance of MotionMapperInference object
    mminfer = MotionMapperInference(
        umap_model_path = ChenLabPyLib.chenlab_filepaths(path = cfg["motion_mapper_file_paths"]["umap_model_path"]), 
        auto_encoder_model_path = ChenLabPyLib.chenlab_filepaths(path = cfg["motion_mapper_file_paths"]["auto_encoder_model_path"]), 
        scaling_parameters_path = ChenLabPyLib.chenlab_filepaths(path = cfg["motion_mapper_file_paths"]["scaling_parameters_path"]),
        look_up_table_path = ChenLabPyLib.chenlab_filepaths(path = cfg["motion_mapper_file_paths"]["look_up_table_path"]),
        watershed_file_path = ChenLabPyLib.chenlab_filepaths(path = cfg["motion_mapper_file_paths"]["watershed_file_path"]),
        version=cfg['motion_mapper_version']
    )
    
    # create instance of Post-Process DLC object
    postdlc = PostAnalysisDLC(
        nose2tail_ae_path = ChenLabPyLib.chenlab_filepaths(path = cfg["post_processing_dlc_paths"]["nose2tail_ae_path"]), 
        feet_ae_path = ChenLabPyLib.chenlab_filepaths(path = cfg["post_processing_dlc_paths"]["feet_ae_path"]), 
        all_ae_path = ChenLabPyLib.chenlab_filepaths(path = cfg["post_processing_dlc_paths"]["all_ae_path"])
    )
    
    start_time = time.time()
    
    # %% Animal information
    
    animalRFIDlist = [str(animalRFID)for animalRFID in cfg["animal_list"]] # make sure all RFIDs are strings
    
    for animalRFID in animalRFIDlist:
        print("Analyzing", animalRFID)
        
        # animal folder
        animal_folder = os.path.join(ChenLabPyLib.chenlab_filepaths(path = cfg["processing_folder"]), animalRFID)
        os.makedirs(animal_folder, exist_ok=True)
        
        # Get all trials from respective animal on DataJoint
        found_trials_csv_path = extract_trials_datajoint(
            animalRFID=animalRFID, 
            animal_folder=animal_folder,
            save_missing_trials=True, 
            overwrite=True
        )
        
        # # %% Post-Processing DeepLabCut data for downstream behavior analysis
        # post_analyzed_dlc_file_path = postdlc.run(
        #     csv_path=found_trials_csv_path, 
        #     animalRFID=animalRFID, 
        #     animal_folder=animal_folder,
        #     overwrite=cfg['post_processing_dlc_params']['overwrite'],
        #     save2trialmat=cfg['post_processing_dlc_params']['save2trialmat'],
        #     disable_progressbar=cfg['post_processing_dlc_params']['disable_progressbar']
        # )
    
        # # %% Load in and extract post-processed DeepLabCut data
        # raw_data = np.load(post_analyzed_dlc_file_path)
        # projections = raw_data['data']
        # per_trial_length = raw_data['per_trial_length']
        # mat_files_used = raw_data['mat_files']
        
        # projections_flatten = projections.reshape((-1, projections.shape[1]*projections.shape[2]))
        
        # print("Pose data reshaped:", projections_flatten.shape)
        # print("Pose data dim:", projections.shape)
        
        # del raw_data, projections
        
        # # %% Run MotionMapper process
        # mminfer.run(
        #     pose_data=projections_flatten, 
        #     per_trial_length=per_trial_length,
        #     mat_files_used=mat_files_used,
        #     animalRFID=animalRFID, 
        #     animal_folder=animal_folder,
        #     sigma=cfg['motion_mapper_inference_params']['sigma'],
        #     save_progress=cfg['motion_mapper_inference_params']['save_progress'],
        #     save2trialmat=cfg['motion_mapper_inference_params']['save2trialmat'],
        #     disable_progressbar=cfg['motion_mapper_inference_params']['disable_progressbar'],
        # )
        
        # # %%
        # print("Elapsed time:", time.time() - start_time)
        
        # # %% Send Slack Notification when finished
        # ChenLabPyLib.send_slack_notification(message="MotionMapper inference w/ {} finished".format(animalRFID))

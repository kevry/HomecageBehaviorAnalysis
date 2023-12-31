# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 15:01:17 2023
@author: Kevin Delgado
"""

import argparse
import json
import numpy as np
import os
import sys
import time
import yaml

import chenlabpylib

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


# %%
if __name__ == "__main__":
    
    # get configuration file path
    json_file_name, config_file_name = get_args()
    
    with open(os.path.join(os.getcwd(), "configs", config_file_name), "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            sys.exit(exc)
    
    
    if sys.platform == 'linux': # assume running on SCC
        # load in JSON file
        f = open(os.path.join(os.getcwd(), "scc", "jsons", json_file_name))
        data = json.load(f)
        f.close()
     
        task_id = int(os.environ["SGE_TASK_ID"])
        animalRFIDlist = [str(animalRFID)for animalRFID in data[task_id-1]] # make sure all RFIDs are strings
        
    else: # running on local Windows lab computers
        # import datajoint and credentials only if running on Windows
        import datajoint as dj
        dj.config['database.host'] = cfg["datajoint_credentials"]['host']
        dj.config['database.user'] = cfg["datajoint_credentials"]['user']
        dj.config['database.password'] = cfg["datajoint_credentials"]['password']
        from database.extract_trials_datajoint import extract_trials_datajoint
        
        animalRFIDlist = [str(animalRFID)for animalRFID in cfg["animal_list"]] # make sure all RFIDs are strings
       
    # ignore initializing motionmapperinference and postanalysisdlc if only running datajoint query    
    if cfg["only_run_datajoint"] == False: 
        
        from post_processing_dlc.post_processing_dlc import PostAnalysisDLC
        from motionmapper.inference import MotionMapperInference
        
        # create instance of MotionMapperInference object
        mminfer = MotionMapperInference(
            umap_model_path = chenlabpylib.chenlab_filepaths(path = cfg["motion_mapper_file_paths"]["umap_model_path"]), 
            auto_encoder_model_path = chenlabpylib.chenlab_filepaths(path = cfg["motion_mapper_file_paths"]["auto_encoder_model_path"]), 
            scaling_parameters_path = chenlabpylib.chenlab_filepaths(path = cfg["motion_mapper_file_paths"]["scaling_parameters_path"]),
            look_up_table_path = chenlabpylib.chenlab_filepaths(path = cfg["motion_mapper_file_paths"]["look_up_table_path"]),
            watershed_file_path = chenlabpylib.chenlab_filepaths(path = cfg["motion_mapper_file_paths"]["watershed_file_path"]),
            version=cfg['motion_mapper_version']
        )
        
        # create instance of Post-Process DLC object
        postdlc = PostAnalysisDLC(
            nose2tail_ae_path = chenlabpylib.chenlab_filepaths(path = cfg["post_processing_dlc_paths"]["nose2tail_ae_path"]), 
            feet_ae_path = chenlabpylib.chenlab_filepaths(path = cfg["post_processing_dlc_paths"]["feet_ae_path"]), 
            all_ae_path = chenlabpylib.chenlab_filepaths(path = cfg["post_processing_dlc_paths"]["all_ae_path"])
        )
        
    start_time = time.time()
    
    # %% Animal information

    for animalRFID in animalRFIDlist:
        print("Analyzing", animalRFID)
        
        # animal folder
        animal_folder = os.path.join(chenlabpylib.chenlab_filepaths(path = cfg["processing_folder"]), animalRFID)
          
        # only run when on Windows
        if sys.platform.startswith('win'):
            # Get all trials from respective animal on DataJoint
            os.makedirs(animal_folder, exist_ok=True)
            found_trials_csv_path = extract_trials_datajoint(
                animalRFID=animalRFID, 
                animal_folder=animal_folder,
                save_missing_trials=True, 
                overwrite=True
            )
        else: # assume .csv with queried data is already collected
            found_trials_csv_path = os.path.join(animal_folder, "FOUND_TRIALS.csv")
        
        if not os.path.exists(found_trials_csv_path):
            print("FOUND_TRIALS.csv does not exist for {}. Skipping to next".format(animalRFID))
            continue
        
        # skip rest of behavioral analysis if True
        if cfg["only_run_datajoint"] == True:
            #chenlabpylib.send_slack_notification(message="QUERY DATA FROM DB w/ {} finished".format(animalRFID))
            continue
    
        # %% Post-Processing DeepLabCut data for downstream behavior analysis
        post_analyzed_dlc_file_path = postdlc.run(
            csv_path=found_trials_csv_path, 
            animalRFID=animalRFID, 
            animal_folder=animal_folder,
            overwrite=cfg['post_processing_dlc_params']['overwrite'],
            save2trialmat=cfg['post_processing_dlc_params']['save2trialmat'],
            disable_progressbar=cfg['post_processing_dlc_params']['disable_progressbar']
        )
    
        # %% Load in and extract post-processed DeepLabCut data
        raw_data = np.load(post_analyzed_dlc_file_path)
        projections = raw_data['data']
        per_trial_length = raw_data['per_trial_length']
        mat_files_used = raw_data['mat_files']
        
        projections_flatten = projections.reshape((-1, projections.shape[1]*projections.shape[2]))
        
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
            overwrite=cfg['motion_mapper_inference_params']['overwrite'],
            save_progress=cfg['motion_mapper_inference_params']['save_progress'],
            save2trialmat=cfg['motion_mapper_inference_params']['save2trialmat'],
            sigma=cfg['motion_mapper_inference_params']['sigma'],
            disable_progressbar=cfg['motion_mapper_inference_params']['disable_progressbar'],
        )
        
        # %%
        print("Elapsed time:", time.time() - start_time)

        # %% Send Slack Notification when finished
        chenlabpylib.send_slack_notification(message="MOTIONMAPPER INFERENCE(SCC) w/ {} finished".format(animalRFID))

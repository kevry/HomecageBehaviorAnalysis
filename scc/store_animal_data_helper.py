# -*- coding: utf-8 -*-
"""
Created on Mon Aug 7 16:25:53 2023
@author: Kevin Delgado
Split up animals that went through behavioral analysis by rig number to save data
"""

import argparse
from chenlabpylib import chenlab_filepaths
import json
import pandas as pd
import os
import subprocess
import sys
import yaml


def get_args():
    """ gets arguments from command line """
    parser = argparse.ArgumentParser(
        description="Parsing argument for animal ID",
        epilog="python file.py --config_file_path configs/config_test.yaml"
    )
    # required argument
    parser.add_argument("--config_file_name", '-cfg', required=True, help='Name of config file with configurations')
    args = parser.parse_args()
    return args.config_file_name


if __name__ == "__main__":
    # get configuration file path
    config_file_name = get_args()
    
    with open(os.path.join(os.path.dirname(os.getcwd()), "configs", config_file_name), "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            sys.exit(exc)
            
    processing_folder = cfg["processing_folder"]

    # get animal list from cfg file
    animal_list = [str(animalRFID)for animalRFID in cfg["animal_list"]] 
    
    # get processing folder
    processing_folder = chenlab_filepaths(path=cfg["processing_folder"])
    
    if len(animal_list) <= 0:
        raise ValueError("No animals listed in configuration file.")
    
    if animal_list[0] == "all":
        print("Using all animals in {}".format(processing_folder))
        animal_list = os.listdir(processing_folder)
    
    # split animals based on rig number
    animal_rig_dict = {}
    for animalRFID in animal_list:
        animal_folder = os.path.join(processing_folder, animalRFID)
        found_trials_csv_file = os.path.join(animal_folder, "FOUND_TRIALS.csv")
        trial_data = pd.read_csv(found_trials_csv_file).values
        full_mat_file_list = [chenlab_filepaths(path=trial[1]) for trial in trial_data]
        training_module_id = os.path.dirname(os.path.dirname(os.path.dirname(full_mat_file_list[0])))
        if training_module_id not in animal_rig_dict.keys():
            animal_rig_dict[training_module_id] = []
        
        animal_rig_dict[training_module_id].append(animalRFID)
        
    num_of_jobs = len(animal_rig_dict.keys())
    num_of_animals = len(animal_list)
        
    # save list to JSON file
    json_obj = json.dumps(animal_rig_dict)
    
    # create json file name 
    json_file_name = "save_animal_data_n{}_b{}.json".format(str(num_of_animals), str(num_of_jobs))
    
    # create jsons directoru
    json_folder = os.path.join(os.getcwd(), "jsons")
    os.makedirs(json_folder, exist_ok=True)
    
    json_file_path = os.path.join(json_folder, json_file_name)
    with open(json_file_path, "w") as outfile:
        outfile.write(json_obj)
    print("{} created.".format(json_file_path))
    
    print("Number of animals in json:", num_of_animals)
    print("Number of jobs to run:", num_of_jobs)
    
    # create path to log folder
    log_folder = os.path.join(os.getcwd(), "log")
    os.makedirs(log_folder, exist_ok=True)
    
    # submit job
    subprocess.run(["qsub", 
        "-t", "1-{}".format(str(num_of_jobs)),
        "-tc", "5",
        "-o", log_folder,
        "-e", log_folder,
        "store_behavior_job.sh", 
        json_file_name, #json file path of animal list
        config_file_name #config file path
    ])


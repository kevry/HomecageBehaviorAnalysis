# Behavioral Analysis for the homecage data
Using the DeepLabCut pose data, running behavioral analysis with MotionMapper to extract behavior
1. https://github.com/DeepLabCut/DeepLabCut
2. https://github.com/bermanlabemory/motionmapperpy

## Installing dependencies
Make sure you have a virtual environment setup before installing all dependencies. You can do this with either Anaconda(conda) or traditional Python(pip). 

## Pipeline
The overall analysis is broken up into 3 steps:

1. Query trial data from DataJoint
2. Run Behavioral Analysis (PostProcessing DLC and MotionMapper inference)
3. Store analyzed data into its respective trial .mat files

## Configuration
All the parameters used throughout the process are contained in the config.yaml file. You can look at template_config.yaml for an example of how a config file should be and what each parameter does. Make a copy of the template_config.yaml file and modify this file.

Edit any parameters in the config file depending on your criteria.

## Running Analysis

### Step 1: Query data from DataJoint
The initial step of the process is gathering all data from DataJoint and saving its contents into a CSV file. Note, this step can only be done on local Windows machines in the lab since we aren't able to open SSH tunnels on any SCC compute nodes. 

Important parameters to set in your config file:
1. `processing_folder`
2. `datajoint_credentials` 
3. `only_run_datajoint`
4. `animal_list`. 

**How to run**:
1. Go to ``<repo_directory>``
2. ``python behavior_inference.py --config_file_name <name of your config file>.yaml``

The script will go through each animal in your <animal_list>, create an animal folder for the specific animal in the <processing_folder> and create a CSV file called _FOUND_TRIALS.csv_ with a list of file paths to all trial .mat files that belong to that animal.


### Step 2: PostProcessing DLC and MotionMapper Inference
The next step is running the behavioral analysis given the list of trials collected by the previous step. For improved speedup, we will use the SCC. 

Important parameters to set in your config file:
1. `processing_folder`
2. `post_processing_dlc_paths`
3. `post_processing_dlc_params`
4. `motion_mapper_version`
5. `motion_mapper_file_paths` 
6. `motion_mapper_inference_params`
7. `animal_list`

**How to run on the SCC**
1. Log in to SCC
2. Go to ``<repo_directory>/scc``
3. ``qsub run_behavior_job_array.sh <name of your config file>.yaml``

**How to run on Windows**
1. Go to ``<repo_directory>``
2. ``python behavior_inference.py --config_file_name <name of your config file>.yaml``


### Step 3: Storing data to trial mat files
The final step is saving all the data collected into each trial's respective .mat file. 

Important parameters to set in your config file:
1. `processing_folder`
2. `post_processing_dlc_paths`
3. `post_processing_dlc_params`
4. `motion_mapper_version`
5. `motion_mapper_file_paths` 
6. `motion_mapper_inference_params`
7. `animal_list`

**How to run on the SCC**
1. Log in to SCC
2. Go to ``<repo_directory>/scc``
3. ``qsub store_behavior_job_array.sh <name of your config file>.yaml``

**How to run on Windows**
1. Go to ``<repo_directory>``
2. ``python store_behavior_data.py --config_file_name <name of your config file>.yaml``

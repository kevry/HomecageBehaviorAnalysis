#!/bin/bash -l

#this merges output and error files into one file
#$ -j y

#this sets the project for the script to be run under
#$ -P jchenlab

#number of cores to select
#$ -pe omp 16

#specify the time limit
#$ -l h_rt=12:00:00

#activate conda environment
module load miniconda
conda activate homecagebehaviorENV

#configuration file path
config_file_name=$1

#run main python script
python store_animal_data_helper.py --config_file_name $config_file_name
exit


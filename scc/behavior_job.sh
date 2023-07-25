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

#handle hdf5 files on scc
export HDF5_USE_FILE_LOCKING='FALSE'

slptime=$(echo "scale=4 ; ($RANDOM/32768) * 10" | bc)
sleep $slptime

#run main python script
cd ..
python behavior_inference_scc.py --json_file_name $1 --config_file_path $2
# Homecage Behavior Analysis for whisker-based task
Using DeepLabCut pose data to quantify behavior with the MotionMapper algorithm

Download the contents of this repository by either manually downloading or using git clone:

`git clone https://github.com/kevry/HomecageBehaviorAnalysis.git`

## Installing dependencies
To run these scripts, you will need to have Python and install the needed libraries/dependencies. You can do this with either Anaconda(conda) or traditional Python(pip) depending on if your system already has Anaconda installed. 

To install using Anaconda:

`cd <repo>/environments`

`conda env create -f conda-env.yml`

`conda activate homecagebehaviorENV`

To install using pip:

`cd <repo>/environments`

`pip install virtualenv`(if you don't already have virtualenv installed)

`virtualenv homecagebehaviorENV`(to create your new Python environment)

`source homecagebehaviorENV/bin/activate.bat`(to enter the virtual environment)

`pip install -r requirements.txt`(to install the requirements in your current env)


## Running script
Here are the following arguments needed to the analysis:
1. The animal RFID

You can run the script on your command window by entering

`python behavior_inference.py --rfid AAOKWE23231MD`



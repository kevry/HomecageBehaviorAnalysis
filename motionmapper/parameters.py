# -*- coding: utf-8 -*-
"""
Written by (Gorden Berman lab): 
Kanishk Jain
kanishkbjain@gmail.com

Modified by (Jerry Chen Lab):
Kevin Delgado
thekevry@gmail.com
"""

from motionmapper_chenlab.mmfunctions import setRunParameters

#% Load the default parameters.
parameters = setRunParameters() 

# %%%%%%% PARAMETERS TO CHANGE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# These need to be revised everytime you are working with a new dataset. #

parameters.projectPath = None #% Full path to the project directory.

parameters.method = 'UMAP' #% We can choose between 'TSNE' or 'UMAP'

parameters.minF = 1        #% Minimum frequency for Morlet Wavelet Transform

parameters.maxF = 5       #% Maximum frequency for Morlet Wavelet Transform,
                           #% usually equals to the Nyquist frequency for your
                           #% measurements.

parameters.samplingFreq = 10    #% Sampling frequency (or FPS) of data.

parameters.numPeriods = 5       #% No. of dyadically spaced frequencies to
                                 #% calculate between minF and maxF.

parameters.pcaModes = 18 #% Number of low-d features.

parameters.numProcessors = -1     #% No. of processor to use when parallel
                                 #% processing for wavelet calculation (if not using GPU)  
                                 #% and for re-embedding. -1 to use all cores 
                                 #% available.

parameters.useGPU = -1           #% GPU to use for wavelet calculation, 
                                 #% set to -1 if GPU not present.

parameters.training_numPoints = 1000000   #% Number of points in mini-trainings.


# %%%%% NO NEED TO CHANGE THESE UNLESS MEMORY ERRORS OCCUR %%%%%%%%%%

parameters.trainingSetSize = 100000000   #% Total number of training set points to find. 
                                             #% Increase or decrease based on
                                             #% available RAM. For reference, 36k is a 
                                             #% good number with 64GB RAM.

parameters.embedding_batchSize = 30000  #% Lower this if you get a memory error when 
                                        #% re-embedding points on a learned map.
    
    
# %%%%%%% CUSTOM Parameters FOR CHENLAB CODE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

parameters.chenlab_tm_data = True
parameters.waveletDecomp = True
parameters.trainingPerc = 0.5


# %%%%%%% tSNE parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#% can be 'barnes_hut' or 'exact'. We'll use barnes_hut for this tutorial for speed.
parameters.tSNE_method = 'barnes_hut' 

# %2^H (H is the transition entropy)
parameters.perplexity = 32

# %number of neigbors to use when re-embedding
parameters.maxNeighbors = 200

# %local neighborhood definition in training set creation
parameters.kdNeighbors = 5

# %t-SNE training set perplexity
parameters.training_perplexity = 20


# %%%%%%%% UMAP Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Size of local neighborhood for UMAP.
parameters.n_neighbors = 100

# Negative sample rate while training.
parameters.train_negative_sample_rate = 5

# Negative sample rate while embedding new data.
parameters.embed_negative_sample_rate = 1

# Minimum distance between neighbors.
parameters.min_dist = 0.2

if parameters.method == 'TSNE':
    parameters.zValstr = 'zVals' 
else:
    parameters.zValstr = 'uVals'
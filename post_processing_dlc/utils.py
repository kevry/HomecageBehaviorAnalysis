# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 13:23:29 2023

@author: Kevin Delgado

Utilities section for post processing analysis of DeepLabCut data
"""

import sys
import numpy as np
from skimage import transform
from scipy.interpolate import InterpolatedUnivariateSpline


def extract_frame_info(tm_position, MAXHEIGHT = 360, MAXWIDTH = 640):
    """ using the training module anchor points detected for the trial,  ..."""
    
    # DeepLabCut training module model trained on 400x300(WxH) frame dimensions
    TRAINED_TM_HEIGHT, TRAINED_TM_WIDTH = 300, 400

    # get crop of frame using TM anchor points
    xmin, xmax = int(min(tm_position[:, 0])), int(max(tm_position[:, 0]))
    ymin, ymax = 0, int(max(tm_position[:, 1]))
    xwidth = min(xmax-xmin, TRAINED_TM_WIDTH)
    yheight = ymax-0
    obj_position = (xmin/MAXWIDTH, ymin/MAXHEIGHT, xwidth/MAXWIDTH, yheight/MAXHEIGHT)
    
    # check if padding was considerd in DLC analysis
    status, add_pad = consider_padding(obj_position, aspect_ratio=TRAINED_TM_WIDTH/TRAINED_TM_HEIGHT, MAXHEIGHT=360, MAXWIDTH=640)
    add2y = add_pad if status == -1 else 0
    
    # get original dimensions of frame before rescaling to 400x300
    original_frame_dims = obj_position[2]*MAXWIDTH, obj_position[3]*MAXHEIGHT + add2y
    scale2standard = (TRAINED_TM_WIDTH/original_frame_dims[0], TRAINED_TM_HEIGHT/original_frame_dims[1])
    
    # return original frame dim, scaling parameters used, anchor corner of TM crop, and padding value
    return original_frame_dims, scale2standard, obj_position, add2y


def consider_padding(obj_position, aspect_ratio, MAXHEIGHT, MAXWIDTH):
    """ check where padding was considered in frame rescaling """
    x, y, w, h = obj_position
    x, y, w, h = int(x*MAXWIDTH), int(y*MAXHEIGHT), int(w*MAXWIDTH), int(h*MAXHEIGHT)
    new_height = int(round(w/aspect_ratio))
    if new_height >= h and (y+new_height) < MAXHEIGHT:
        add_padding_h = new_height - h
        return -1, add_padding_h
    else:
        new_width = int(round(aspect_ratio * h))
        add_padding_w = new_width - w
        return 0, add_padding_w


def search_cutoff(bool_arr):
    """ search for cutoff in boolean array in reverse order, 
        return index of the first True value """
    cutoff_index = bool_arr.shape[0]
    for i in range(bool_arr.shape[0]-1, -1, -1):
        if bool_arr[i] == False:
            cutoff_index -= 1
        else:
            break
    return cutoff_index


def search_front_cutoff(bool_arr):
    """ search for cutoff in boolean array, 
        return index of the first True value"""
    cutoff_idx = 0
    for i in range(bool_arr.shape[0]):
        if bool_arr[i] == False:
            cutoff_idx += 1
        else:
            break
    cutoff_idx = cutoff_idx if cutoff_idx == 0 else cutoff_idx -1
    return cutoff_idx


def interpolation(interpolated_dlc_arr, markers_above_confidence, threshold_for_interp):
    """ apply linear interpolation on markers """

    # loop through each marker x,y coordinate
    for i in range(interpolated_dlc_arr.shape[1]):
        marker_confidences = markers_above_confidence[:,i].copy()
        
        # skip if all markers either known or unknown, skip part
        if np.all(np.invert(marker_confidences)) or np.all(marker_confidences):
            continue
        
        # linear interpolation only done on frames dropped between high confidence frames .. set cutoff ( +1)
        cutoff = search_cutoff(bool_arr = marker_confidences.copy()) + 1
        front_cutoff = search_front_cutoff(bool_arr = marker_confidences.copy())
        marker_confidences = marker_confidences[front_cutoff:cutoff]
        
        # if number of confidences after cut off is less than 4, skip part. Linear interpolation not good for only 4 points ...
        if marker_confidences.shape[0] < 4:
            continue
        
        x = np.argwhere(marker_confidences).flatten()
        x_new = np.argwhere(np.invert(marker_confidences)).flatten()
        
        # only apply linear interpolation to missing markers that are FPS/2 frames close to a high confidence frame
        x_new_valid = []
        for idx, missing_idx in enumerate(x_new):
            if np.any(np.abs(missing_idx - x) <= threshold_for_interp):
                x_new_valid.append(missing_idx)
        del x_new
        x_new_valid = np.array(x_new_valid)
        
        # skip if there are more unknown values than known
        if (x_new_valid.shape[0] - x.shape[0]) > round(marker_confidences.shape[0]*0.2):
            #print(i, x_new_valid.shape[0],  x.shape[0], round(marker_confidences.shape[0]*0.2), "HERE")
            continue
            
        for j in [0,1]:
            y = interpolated_dlc_arr[x,i,j]
            # linear interpolation
            try:
                s = InterpolatedUnivariateSpline(x, y, k=1, bbox=[x[0], x[-1]+1])
                y_new = s(x_new_valid)
            except:
                print(marker_confidences, x, y)
                print(x_new_valid, x_new_valid.shape[0], x.shape[0], round(marker_confidences.shape[0]*0.2))
                
            interpolated_dlc_arr[x_new_valid+front_cutoff,i,j] = y_new  

        # replace marker status to True after interpolation
        markers_above_confidence[x_new_valid+front_cutoff,i] = True
            
    return interpolated_dlc_arr, markers_above_confidence

    
def run_similarity_transformation(src, dest):
    """ similarity (scale, translate, rotate) transformation for data """
    simtransform = transform.SimilarityTransform()
    result = simtransform.estimate(src, dest)
    if result:
        return result, simtransform
    else:
        return result, None
    
    

def find_closest_neighbor(sub_dlc_arr, ae_res, subset_markers_conf):
    """ translate new found body parts based on closest neighbor """
    
    marker_indices_w_high_confidence = np.argwhere(subset_markers_conf == True).flatten()
    marker_indices_w_low_confidence = np.argwhere(subset_markers_conf == False).flatten()
    for i in range(len(marker_indices_w_low_confidence)):
        closest_marker_idx = marker_indices_w_high_confidence[np.argmin(marker_indices_w_high_confidence - marker_indices_w_low_confidence[i])]
        # distance between true high confidence marker and ae predicted high confidence marker
        true_marker_distance = sub_dlc_arr[closest_marker_idx] - ae_res[closest_marker_idx]
        # move new predicted marker with respect to closest true marker
        ae_res[marker_indices_w_low_confidence[i]] += true_marker_distance
        
    sub_dlc_arr[marker_indices_w_low_confidence] = ae_res[marker_indices_w_low_confidence]
    return sub_dlc_arr
      
    
    
def alignbyTM(dlc_arr_og_space, tm_position_og_space, ATTACH_TM):
    """ similarity transformation for markers based on aligning training module """
    
    # BASELINE is in 640x360 space
    BASELINE = np.array([
        [167.09735,  272.91092 ],
        [259.67645,  242.90894 ],
        [336.39093,  214.58502 ],
        [261.89355,   85.32816 ],
        [206.95055,  161.89272 ],
        [146.90479,  250.70068 ],
    ])
    
    # similarity transformation
    simtransform = transform.SimilarityTransform()
    result = simtransform.estimate(tm_position_og_space[:, :2], BASELINE)
    if result == False:
        raise ValueError("Error aligining training module")
        
    # transform dlc markers based on similarity transform parameters
    for i in range(dlc_arr_og_space.shape[0]):
        dlc_arr_og_space[i] = simtransform(dlc_arr_og_space[i])
    
    if ATTACH_TM: # return both tm coordinates and dlc data concatenated into one
        egoTMdata = []
        for i in range(dlc_arr_og_space.shape[0]):
            egoTMdata.append(np.array([np.concatenate([BASELINE, dlc_arr_og_space[i, :, :2]], axis=0)]))
        egoTMdata = np.concatenate(egoTMdata, axis=0)
        return egoTMdata
    else:
        return dlc_arr_og_space
    
    
    
def egocenterALL(h5):
    """ convert all TM anchor points and user-defined mouse markers into egocentric coordinates with respect to training module """
    center = (h5[:, 2, :] + h5[:, 3, :])/2
    egoh5 = h5.copy()
    for i in range(h5.shape[0]):
        egoh5[i] = egoh5[i] - center[i]

    mid_pt2 = (egoh5[:, 2] + egoh5[:, 3])/2 
    mid_pt1 = (egoh5[:, 0] + egoh5[:, 5])/2
    
    dir_arr = mid_pt2-mid_pt1 
    dir_arr = dir_arr / np.linalg.norm(dir_arr, axis=1)[:, np.newaxis] # get unit vector
    for t in range(egoh5.shape[0]):
        rot_mat = np.array([[dir_arr[t, 0], dir_arr[t, 1]], [-dir_arr[t, 1], dir_arr[t, 0]]])
        egoh5[t] = np.array(np.dot(egoh5[t], rot_mat.T))
    
    return egoh5   



def egocentermouse(h5):
    """ convert user-defined mouse markers into egocentric coordinates with respect to upperback """
    center = h5[:, 4, :]
    egoh5 = h5.copy()
    for i in range(h5.shape[0]):
        egoh5[i] = egoh5[i] - center[i]
    
    dir_arr = egoh5[:, 4]- egoh5[:, 3] 
    dir_arr = dir_arr / np.linalg.norm(dir_arr, axis=1)[:, np.newaxis] # get unit vector
    for t in range(egoh5.shape[0]):
        rot_mat = np.array([[dir_arr[t, 0], dir_arr[t, 1]], [-dir_arr[t, 1], dir_arr[t, 0]]])
        egoh5[t] = np.array(np.dot(egoh5[t], rot_mat.T))
    
    return egoh5   



def linux2windowspath(file_path):
	""" modify file path depending on the current OS in use.
	this code will sometimes run on Windows environment or SCC(linux) """
    
	# create hash table of letter network drives and SCC mounted paths
	map2scc = {
		'Z:': '/net/claustrum/mnt/data', 
		'Y:': '/net/claustrum/mnt/data1',
		'X:': '/net/claustrum2/mnt/data',
		'W:': '/net/clasutrum3/mnt/data', 
		'V:': '/net/claustrum4/mnt/storage/data',
	}

	map2win = {
		'/net/claustrum/mnt/data': 'Z:', 
		'/net/claustrum/mnt/data1': 'Y:',
		'/net/claustrum2/mnt/data': 'X:',
		'/net/clasutrum3/mnt/data': 'W:', 
		'/net/claustrum4/mnt/storage/data': 'V:',
	}

	# running on linux
	if sys.platform == 'linux':
		for key in map2scc:
			if key in file_path:
				file_path = file_path.replace(key, map2scc[key])
				break
		# reverse backslash		
		file_path = file_path.replace('\\', '/')
	else:
		# running on windows
		for key in map2win:
			if key in file_path:
				file_path = file_path.replace(key, map2win[key])
				break
		file_path = file_path.replace('/', '\\')

	return file_path



def fix_stretch(egocentric_dlc_arr):
    """ few extreme data points become outliers (far from TM anchor points), we can fix by just subtracting """
    TM_min_x, TM_max_x = egocentric_dlc_arr[:,:6,0].min() - 10, 350
    TM_min_y, TM_max_y = egocentric_dlc_arr[:,:6,1].min() - 10, egocentric_dlc_arr[:,:6,1].max() + 10
    
    if np.any(egocentric_dlc_arr[:,:,0] > TM_max_x):
        logic_arr = egocentric_dlc_arr[:,:,0] > TM_max_x
        egocentric_dlc_arr[logic_arr,0] = 348
    
    if np.any(egocentric_dlc_arr[:,:,1] > TM_max_y):
        logic_arr = egocentric_dlc_arr[:,:,1] > TM_max_y
        egocentric_dlc_arr[logic_arr,1] = TM_max_y - 2
        
    if np.any(egocentric_dlc_arr[:,:,0] < TM_min_x):
        logic_arr = egocentric_dlc_arr[:,:,0] < TM_min_x
        egocentric_dlc_arr[logic_arr,0] = TM_min_x + 2
        
    if np.any(egocentric_dlc_arr[:,:,1] < TM_min_y):
        logic_arr = egocentric_dlc_arr[:,:,1] < TM_min_y
        egocentric_dlc_arr[logic_arr,1] = TM_min_y + 2
        
    return egocentric_dlc_arr



def batch_for_trial(MARKERS_ABOVE_CONFIDENCE, MINIMUM_BATCH_SIZE=5):
    """ find batch of valid frames if it exists.
    not all frames in trial might be valid (animal not at start of trial or leaves early) """
    
    analyzed_frames_bool = MARKERS_ABOVE_CONFIDENCE.all(axis=1)
    
    group_of_batch_indexes = []
    index_for_batch = []
    for i, analyzed_frame in enumerate(analyzed_frames_bool):
        if analyzed_frame == False:
            if len(index_for_batch) >= MINIMUM_BATCH_SIZE:
                group_of_batch_indexes.append([index_for_batch[0], index_for_batch[-1]])
            index_for_batch = []
        else:
            index_for_batch.append(i)
    if len(index_for_batch) >= MINIMUM_BATCH_SIZE:
        group_of_batch_indexes.append([index_for_batch[0], index_for_batch[-1]])  
        
    if len(group_of_batch_indexes) == 0:
        return False, [-1,-1]
        
    # get largest batch 
    largest_batch= [-1,-1]
    for batch_index in group_of_batch_indexes:
        if (batch_index[-1] - batch_index[0]) > (largest_batch[-1] - largest_batch[0]):
            largest_batch = batch_index   
    
    return True, largest_batch










# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 15:19:56 2023
@author: Kevin Delgado
Calculate the watershed regions for each datapoint
"""

import hdf5storage
import numpy as np

def get_watershed_regions(embedded2ddata, watershed_file_path, behavior_labeled_look_up_table_inverted):
    """ calculate watershed regions of each data point """

    print("Finding watershedRegions for each data point")
    wshedfile = hdf5storage.loadmat(watershed_file_path)
    LL = wshedfile['LL']
    xx = wshedfile['xx'][0]
    
    # Code taken from BermanLabemory motionmapperpy repository
    # https://github.com/bermanlabemory/motionmapperpy
    watershedRegions = np.digitize(embedded2ddata, xx)
    watershedRegions = LL[watershedRegions[:, 1], watershedRegions[:, 0]]
    
    watershedRegionsSimplified = []
    for wshedLabel in watershedRegions:
        if wshedLabel == 0:
            wshedLabelGlobal = 0
        else:
            wshedLabelGlobal = int(behavior_labeled_look_up_table_inverted[str(wshedLabel)])
        watershedRegionsSimplified.append(wshedLabelGlobal)
    return np.array(watershedRegionsSimplified)
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 14:24:07 2023
@author: Kevin Delgado
plot scatterplot and heatmap of data
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from motionmapper.mmfunctions import findPointDensity, gencmap
import stat

def draw_plot(data, animalRFID, animal_folder, sigma = 0.1, c_limit=0.95):
    """ create scatter plot and heatmap for data """

    m = np.abs(data).max()
    fig, axes = plt.subplots(1, 2, figsize=(12,6))
    
    axes[0].scatter(data[:,0], data[:,1], marker='.', c=np.arange(data.shape[0]), s=1)
    axes[0].set_xlim([-m-10, m+10])
    axes[0].set_ylim([-m-10, m+10])
    axes[0].set_title(data.shape)
    
    _, xx, density = findPointDensity(data, sigma, 511, [-m-10, m+10])
    _ = axes[1].imshow(density, cmap=gencmap(), extent=(xx[0], xx[-1], xx[0], xx[-1]), 
              origin='lower', vmax=np.max(density)*c_limit)
    axes[1].set_title('Sigma: %0.02f'%(sigma))

    plot_file_path = os.path.join(animal_folder, "HEATMAP.png")
    fig.savefig(plot_file_path)
    os.chmod(plot_file_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    print("Created heatmap for {} data!".format(animalRFID))
    return

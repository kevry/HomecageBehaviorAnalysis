# -*- coding: utf-8 -*-
"""
Written by (Gorden Berman lab): 
Kanishk Jain
kanishkbjain@gmail.com

https://github.com/bermanlabemory/motionmapperpy
https://arxiv.org/pdf/1310.4249.pdf
"""

import copy
from easydict import EasyDict as edict
import matplotlib as mpl
import numpy as np
from scipy.fft import fft2, ifft2, fftshift
import warnings


def findWaveletsChenLab(projections, per_trial_length, pcaModes, omega0, numPeriods, samplingFreq, maxF, minF, numProcessors, useGPU):
    """
    findWavelets finds the wavelet transforms resulting from a time series.
    :param projections: N x d array of projection values.
    :param pcaModes: # of transforms to find.
    :param omega0: Dimensionless morlet wavelet parameter.
    :param numPeriods: number of wavelet frequencies to use.
    :param samplingFreq: sampling frequency (Hz).
    :param maxF: maximum frequency for wavelet transform (Hz).
    :param minF: minimum frequency for wavelet transform (Hz).
    :param numProcessors: number of processors to use in parallel code.
    :param useGPU: GPU to use.
    :return:
            amplitudes -> wavelet amplitudes (N x (pcaModes*numPeriods) )
            f -> frequencies used in wavelet transforms (Hz)
    """

    if useGPU>=0:
        try:
            import cupy as np
        except ModuleNotFoundError as E:
            warnings.warn("Trying to use GPU but cupy is not installed. Install cupy or set parameters.useGPU = -1. "
                  "https://docs.cupy.dev/en/stable/install.html")
            raise E

        np.cuda.Device(useGPU).use()
        #print('\t Using GPU #%i'%useGPU)
    else:
        import numpy as np
        import multiprocessing as mp
        if numProcessors<0:
            numProcessors = mp.cpu_count()
        #print('\t Using #%i CPUs.' % numProcessors)

    projections = np.array(projections)

    dt = 1.0 / samplingFreq
    minT = 1.0 / maxF
    maxT = 1.0 / minF
    Ts = minT * (2 ** ((np.arange(numPeriods) * np.log(maxT / minT)) / (np.log(2) * (numPeriods - 1))))
    f = (1.0 / Ts)[::-1]
    N = projections.shape[0]

    if useGPU>=0:
        amplitudes = np.zeros((numPeriods*pcaModes,N))
        for i in range(pcaModes):
            amplitudes[i*numPeriods:(i+1)*numPeriods] = fastWavelet_morlet_convolution_parallel_ChenLab(i, projections[:, i], per_trial_length, f, omega0, dt, useGPU)
    else:
        try:
            pool = mp.Pool(numProcessors)
            amplitudes = pool.starmap(fastWavelet_morlet_convolution_parallel_ChenLab,
                                      [(i, projections[:, i], per_trial_length, f, omega0, dt, useGPU) for i in range(pcaModes)])
            amplitudes = np.concatenate(amplitudes, 0)
            pool.close()
            pool.join()
        except Exception as E:
            pool.close()
            pool.join()
            raise E
    return amplitudes.T, f


def fastWavelet_morlet_convolution_parallel_ChenLab(modeno, x, per_trial_length, f, omega0, dt, useGPU):
    if useGPU>=0:
        import cupy as np
        np.cuda.Device(useGPU).use()
    else:
        import numpy as np

    amp_list = []
    x_list_split = np.split(x, np.cumsum(per_trial_length)[:-1])

    for x in x_list_split:
        N = len(x)
        L = len(f)
        amp = np.zeros((L, N))

        if not N // 2:
            x = np.concatenate((x, [0]), axis=0)
            N = len(x)
            wasodd = True
        else:
            wasodd = False

        #x = np.concatenate([np.zeros(int(N / 2)), x, np.zeros(int(N / 2))], axis=0)
        x = np.concatenate([np.ones(int(N / 2))*x[0], x, np.ones(int(N / 2))*x[-1]], axis=0)
        M = N
        N = len(x)
        scales = (omega0 + np.sqrt(2 + omega0 ** 2)) / (4 * np.pi * f)
        Omegavals = 2 * np.pi * np.arange(-N / 2, N / 2) / (N * dt)

        xHat = np.fft.fft(x)
        xHat = np.fft.fftshift(xHat)

        if wasodd:
            idx = np.arange((M / 2), (M / 2 + M - 2)).astype(int)
        else:
            idx = np.arange((M / 2), (M / 2 + M)).astype(int)

        for i in range(L):
            m = (np.pi ** (-0.25)) * np.exp(-0.5 * (-Omegavals * scales[i] - omega0) ** 2)
            q = np.fft.ifft(m * xHat) * np.sqrt(scales[i])

            q = q[idx]
            amp[i, :] = np.abs(q) * (np.pi ** -0.25) * np.exp(0.25 * (omega0 - np.sqrt(omega0 ** 2 + 2)) ** 2) / np.sqrt(
                2 * scales[i])
        # print('Mode %i done.'%(modeno))

        amp_list.append(amp)
    amp = np.concatenate(amp_list, axis=1)
    return amp


def gencmap():
    """
    Get behavioral map colormap as a matplotlib colormap instance.
    :return: Matplotlib colormap instance.
    """
    colors = np.zeros((64, 3))
    colors[:21, 0] = np.linspace(1, 0, 21)
    colors[20:43, 0] = np.linspace(0, 1, 23)
    colors[42:, 0] = 1.0

    colors[:21, 1] = np.linspace(1, 0, 21)
    colors[20:43, 1] = np.linspace(0, 1, 23)
    colors[42:, 1] = np.linspace(1, 0, 22)

    colors[:21, 2] = 1.0
    colors[20:43, 2] = np.linspace(1, 0, 23)
    colors[42:, 2] = 0.0
    return mpl.colors.ListedColormap(colors)


def getDensityBounds(density, thresh=1e-6):
    """
    Get the outline for density maps.
    :param density: m by n density image.
    :param thresh: Density threshold for boundaries. Default 1e-6.
    :return: (p by 2) points outlining density map.
    """
    x_w, y_w = np.where(density > thresh)
    x, inv_inds = np.unique(x_w, return_inverse=True)
    bounds = np.zeros((x.shape[0] * 2 + 1, 2))
    for i in range(x.shape[0]):
        bounds[i, 0] = x[i]
        bounds[i, 1] = np.min(y_w[x_w == bounds[i, 0]])
        bounds[x.shape[0] + i, 0] = x[-i - 1]
        bounds[x.shape[0] + i, 1] = np.max(y_w[x_w == bounds[x.shape[0] + i, 0]])
    bounds[-1] = bounds[0]
    bounds[:, [0, 1]] = bounds[:, [1, 0]]
    return bounds.astype(int)


def findPointDensity(zValues, sigma, numPoints, rangeVals):
    """
    findPointDensity finds a Kernel-estimated PDF from a set of 2D data points
    through convolving with a gaussian function.
    :param zValues: 2d points of shape (m by 2).
    :param sigma: standard deviation of smoothing gaussian.
    :param numPoints: Output density map dimension (n x n).
    :param rangeVals: 1 x 2 array giving the extrema of the observed range
    :return:
        bounds -> Outline of the density map (k x 2).
        xx -> 1 x numPoints array giving the x and y axis evaluation points.
        density -> numPoints x numPoints array giving the PDF values (n by n) density map.
    """
    xx = np.linspace(rangeVals[0], rangeVals[1], numPoints)
    yy = copy.copy(xx)
    [XX, YY] = np.meshgrid(xx, yy)
    G = np.exp(-0.5 * (np.square(XX) + np.square(YY)) / np.square(sigma))
    Z = np.histogramdd(zValues, bins=[xx, yy])[0]
    Z = Z / np.sum(Z)
    Z = np.pad(Z, ((0, 1), (0, 1)), mode='constant', constant_values=((0, 0), (0, 0)))
    density = fftshift(np.real(ifft2(np.multiply(fft2(G), fft2(Z))))).T
    density[density < 0] = 0
    bounds = getDensityBounds(density)
    return bounds, xx, density


def setRunParameters(parameters=None):
    """
    Get parameter dictionary for running motionmapperpy.
    :param parameters: Existing parameter dictionary, defaults will be filled for missing keys.
    :return: Parameter dictionary.
    """
    if isinstance(parameters, dict):
        parameters = edict(parameters)
    else:
        parameters = edict()


    """# %%%%%%%% General Parameters %%%%%%%%"""

    # %number of processors to use in parallel code
    numProcessors = 12

    useGPU = -1

    method = 'TSNE' # or 'UMAP'


    """%%%%%%%% Wavelet Parameters %%%%%%%%"""
    # %Whether to do wavelet decomposition, if False then use normalized projections for tSNE embedding.
    waveletDecomp = True

    # %number of wavelet frequencies to use
    numPeriods = 25

    # dimensionless Morlet wavelet parameter
    omega0 = 5

    # sampling frequency (Hz)
    samplingFreq = 100

    # minimum frequency for wavelet transform (Hz)
    minF = 1

    # maximum frequency for wavelet transform (Hz)
    maxF = 50


    """%%%%%%%% t-SNE Parameters %%%%%%%%"""
    # Global tSNE method - 'barnes_hut' or 'exact'
    tSNE_method = 'barnes_hut'

    # %2^H (H is the transition entropy)
    perplexity = 32

    # %embedding batchsize
    embedding_batchSize = 20000

    # %maximum number of iterations for the Nelder-Mead algorithm
    maxOptimIter = 100

    # %number of points in the training set
    trainingSetSize = 35000

    # %number of neigbors to use when re-embedding
    maxNeighbors = 200

    # %local neighborhood definition in training set creation
    kdNeighbors = 5

    # %t-SNE training set perplexity
    training_perplexity = 20

    # %number of points to evaluate in each training set file
    training_numPoints = 10000

    # %minimum training set template length
    minTemplateLength = 1

    """%%%%%%%% UMAP Parameters %%%%%%%%"""
    # Size of local neighborhood for UMAP.
    n_neighbors = 15

    # Negative sample rate while training.
    train_negative_sample_rate = 5

    # Negative sample rate while embedding new data.
    embed_negative_sample_rate = 1

    # Minimum distance between neighbors.
    min_dist = 0.1

    # UMAP output dimensions.
    umap_output_dims = 2

    # Number of training epochs.
    n_training_epochs = 1000

    # Embedding rescaling parameter.
    rescale_max = 100

    """%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""


    if not 'numProcessors' in parameters.keys():
        parameters.numProcessors = numProcessors

    if not 'numPeriods' in parameters.keys():
        parameters.numPeriods = numPeriods

    if not 'omega0' in parameters.keys():
        parameters.omega0 = omega0



    if not 'samplingFreq' in parameters.keys():
        parameters.samplingFreq = samplingFreq

    if not 'minF' in parameters.keys():
        parameters.minF = minF

    if not 'maxF' in parameters.keys():
        parameters.maxF = maxF


    if not 'tSNE_method' in parameters.keys():
        parameters.tSNE_method = tSNE_method

    if not 'perplexity' in parameters.keys():
        parameters.perplexity = perplexity

    if not 'embedding_batchSize' in parameters.keys():
        parameters.embedding_batchSize = embedding_batchSize

    if not 'maxOptimIter' in parameters.keys():
        parameters.maxOptimIter = maxOptimIter

    if not 'trainingSetSize' in parameters.keys():
        parameters.trainingSetSize = trainingSetSize

    if not 'maxNeighbors' in parameters.keys():
        parameters.maxNeighbors = maxNeighbors

    if not 'kdNeighbors' in parameters.keys():
        parameters.kdNeighbors = kdNeighbors

    if not 'training_perplexity' in parameters.keys():
        parameters.training_perplexity = training_perplexity

    if not 'training_numPoints' in parameters.keys():
        parameters.training_numPoints = training_numPoints

    if not 'minTemplateLength' in parameters.keys():
        parameters.minTemplateLength = minTemplateLength

    if not 'waveletDecomp' in parameters.keys():
        parameters.waveletDecomp = waveletDecomp

    if not 'useGPU' in parameters.keys():
        parameters.useGPU = useGPU

    if not 'n_neighbors' in parameters.keys():
        parameters.n_neighbors = n_neighbors

    if not 'train_negative_sample_rate' in parameters.keys():
        parameters.train_negative_sample_rate = train_negative_sample_rate

    if not 'embed_negative_sample_rate' in parameters.keys():
        parameters.embed_negative_sample_rate = embed_negative_sample_rate

    if not 'min_dist' in parameters.keys():
        parameters.min_dist = min_dist

    if not 'umap_output_dims' in parameters.keys():
        parameters.umap_output_dims = umap_output_dims

    if not 'n_training_epochs' in parameters.keys():
        parameters.n_training_epochs = n_training_epochs

    if not 'rescale_max' in parameters.keys():
        parameters.rescale_max = rescale_max

    if not 'method' in parameters.keys():
        parameters.method = method

    return parameters

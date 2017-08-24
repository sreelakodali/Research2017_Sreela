# Human Activity Recognition (HAR) Demo: Additional Methods for Processing Raw iPhone Acc/Gyro Data
# Author: Sreela Kodali, kodali@princeton.edu
# Updated last: August 13th, 2017
# 
# This file has the helper functions for livephoneDataReader.py
# livephone_utils.py is in the Models2Chip directory --> models2chip/dnn_models/scripts/livephone_utils.py

import numpy as np
import math

def filename(n):
	start = n.find('HAR') + 3
	end = n.find('.csv', start)
	return n[start:end]

def nRows(nSamples, o, w):
	ow = int(o * w)
	nr = math.floor((nSamples - ow) /(w - ow))
	nr = int(nr)
	return nr

def expandInput(pointer, dataset, sf, w):
	pointer = int(pointer)
	r = np.zeros((1, sf), 'float64')
	for j in range(0, w):
		j = int(j)
		p = pointer + j
		x = dataset[p, :]
		if j == 0:
			r = x
		else:
			r = np.concatenate([r, x])
	return r

def preprocessRawInput(dataset, o, w, sf):
	samples = np.shape(dataset)[0]
	nRows_reshaped = nRows(samples, o, w)
	nCols_reshaped = w * sf
	new = np.zeros((nRows_reshaped, nCols_reshaped),'float64')
	step = math.floor(w*(1 - o))

	k = 0
	for i in range(0, nRows_reshaped):
		new[i,:] = expandInput(k, dataset, sf, w)
		k = k + step

	return new
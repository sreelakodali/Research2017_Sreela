# Sreela Kodali - Reading in WISDM RAW

import numpy as np
import h5py
import csv

def FixNans(arr):
	for i in range(0, np.shape(arr)[1]):
		x = arr[:,i]
		mask  = np.isnan(x)
		x[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), x[~mask])
		arr[:,i] = x
	return arr

# WISDM

mainpath = '/Users/Sreela/Documents/School/Princeton/Year3/STARnet/Research2017_Sreela/DL_har_datasets/WISDM_Dataset/'
datapath = mainpath+'WISDM_ar_v1.1_raw.txt'

f=open(datapath,"r")
lines=f.readlines()
result=[]
for x in lines:
	result.append(x.split(",")[1])
f.close()
print(result)

inputs = FixNans(inputs)

targets[targets == 'Walking'] = 1
targets[targets == 'Jogging'] = 2
targets[targets == 'Upstairs'] = 3
targets[targets == 'Downstairs'] = 4
targets[targets == 'Sitting'] = 5
targets[targets == 'Standing'] = 6

print(np.unique(targets))

f = h5py.File('WISDMraw_test.h5', 'w')
dset1 = f.create_dataset('/all/inputs', data = inputs, dtype='float64')
dset2 = f.create_dataset('/all/targets', data = targets, dtype='int64')

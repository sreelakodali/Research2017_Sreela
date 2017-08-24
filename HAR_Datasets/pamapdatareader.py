# PAMAP2 Data Reader
# Reading in the PAMAP2 Dataset and creating a corresponding HDF5 File
# Author: Sreela Kodali, kodali@princeton.edu

# Command: python pamapdatareader.py
# produces a .hdf5 in the same directory as the .py file

import numpy as np
import h5py
import glob

#Helper function that replaces nans with linearly interpolated values
# Function takes in an input array and linearly interpolates each column,
# assuming each column is one sensor feature
def FixNans(arr):
	for i in range(0, np.shape(arr)[1]):
		x = arr[:,i]
		mask  = np.isnan(x)
		x[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), x[~mask])
		arr[:,i] = x
	return arr

# Download the original PAMAP2 Dataset https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring
# Download my version with reorganized subdirectories here: 
# Set the 'mainpath' to the main PAMAP2_Dataset directory
mainpath = '/Users/Sreela/Documents/School/Princeton/Year3/STARnet/Research2017_Sreela/HAR_Datasets/PAMAP2_Dataset/'

# In my modified version, I split the .dat 'Protocol' files into three subgroups in subdirectories for my clarity.
# Divided into test, validation, and training as described in the Hammerla Paper https://arxiv.org/pdf/1604.08880.pdf
# If you're using the original dataset, do the following:
# Create a subdirectory 'PAMAP2_Dataset/Test' and put the 'Protocol/subject106.dat' file
# Create a subdirectory 'PAMAP2_Dataset/Val' and put the 'Protocol/subject105.dat' file
# Create a subdirectory 'PAMAP2_Dataset/Training' and put the rest of the .dat 'Protocol files'
testpath = mainpath+'Test/subject106.dat'
valpath = mainpath+'Val/subject105.dat'
trainpath = mainpath+'Training/'



# VALIDATION DATA
v = np.loadtxt(valpath)
inputs_valPAMAP = v[:,2:] #columns starting with Heart Rate data, all the way to the end
inputs_valPAMAP = FixNans(inputs_valPAMAP)

rows = np.shape(inputs_valPAMAP)[0]
targets_valPAMAP = np.zeros((rows,))
targets_valPAMAP = v[:,1]
# Downsample the data from 100HZ to 33.3Hz like in the Hammerla paper
dsinputs_valPAMAP = inputs_valPAMAP[::3]
dstargets_valPAMAP = targets_valPAMAP[::3]

# TEST DATA
te = np.loadtxt(testpath)
inputs_testPAMAP = te[:,2:] #columns starting with Heart Rate data, all the way to the end
inputs_testPAMAP = FixNans(inputs_testPAMAP)

te_rows = np.shape(inputs_testPAMAP)[0]
targets_testPAMAP = np.zeros((te_rows,))
targets_testPAMAP = te[:,1]
# Downsample the data from 100HZ to 33.3Hz like in the Hammerla paper
dsinputs_testPAMAP = inputs_testPAMAP[::3]
dstargets_testPAMAP = targets_testPAMAP[::3]

# TRAINING DATA
trainfiles = glob.glob(trainpath + '*.dat')
i = 0
for name in trainfiles:
	tr = np.loadtxt(name)
	tr_rows = np.shape(tr)[0]
	a = tr[:,2:] #columns starting with Heart Rate data, all the way to the end
	b = np.zeros((tr_rows,))
	b = tr[:,1]
	a = a[::3] #downsample 
	b = b[::3] #downsample
	if (i == 0):
		dsinputs_trainPAMAP =  a
		dstargets_trainPAMAP = b
		i = 1
	else:
		dsinputs_trainPAMAP = np.concatenate((dsinputs_trainPAMAP, a))	
		dstargets_trainPAMAP = np.concatenate((dstargets_trainPAMAP, b))

# Replace nans with linearly interpolated values per column
dsinputs_trainPAMAP = FixNans(dsinputs_trainPAMAP)


# Print statements to see if all discrete classes present in each dataset
#print(np.unique(dstargets_valPAMAP))
#print(np.unique(dstargets_testPAMAP))
#print(np.unique(dstargets_trainPAMAP))


m = np.amax(dsinputs_trainPAMAP)
dsinputs_trainPAMAP = np.multiply(np.divide(dsinputs_trainPAMAP, m), 32)
dsinputs_testPAMAP = np.multiply(np.divide(dsinputs_testPAMAP, m), 32)
dsinputs_valPAMAP = np.multiply(np.divide(dsinputs_valPAMAP, m), 32)

# Remapping the discrete class number 0 1 2 3 4 5 6 7 12 13 16 17 24 --> 1 2 3 4 5 6 7 8 9 10 11 12 13
# so it's in a more usable form for the FC NN model
targets = [dstargets_valPAMAP, dstargets_testPAMAP, dstargets_trainPAMAP]
j = 0
for t in targets:
	t[t == 12] = 8
	t[t == 13] = 9
	t[t == 16] = 10
	t[t == 17] = 11
	t[t == 24] = 12
	t = t + 1
	targets[j] = t
	j = j + 1

# Print statements to ensure discrete classes mapped properly
#print(np.unique(dstargets_valPAMAP))
#print(np.unique(dstargets_testPAMAP))
#print(np.unique(dstargets_trainPAMAP))

# NOTE: If you want to replace nans with zeros instead of linearly interpolated values, 
# uncomment the following three lines and comment out 41, 53, and 82.
#dsinputs_valPAMAP = np.nan_to_num(dsinputs_valPAMAP)
#dsinputs_testPAMAP = np.nan_to_num(dsinputs_testPAMAP)
#dsinputs_trainPAMAP = np.nan_to_num(dsinputs_trainPAMAP)

#Checking if any nans present in the dataset. Should yield false if none.
print('Testing for NaN presence')
print(np.isnan(np.sum(dsinputs_valPAMAP)))
print(np.isnan(np.sum(dsinputs_testPAMAP)))
print(np.isnan(np.sum(dsinputs_trainPAMAP)))


# Create new hdf5 file with 3 subgroups: test, training, and validation
f = h5py.File('pamap2_rescale_nanLI.h5', 'w')
dset1 = f.create_dataset('/test/inputs', data = dsinputs_testPAMAP, dtype='float64')
dset2 = f.create_dataset('/test/targets', data = targets[1], dtype='int64')
dset3 = f.create_dataset('/training/inputs', data = dsinputs_trainPAMAP, dtype='float64')
dset4 = f.create_dataset('/training/targets', data = targets[2], dtype='int64')
dset5 = f.create_dataset('/validation/inputs', data = dsinputs_valPAMAP, dtype='float64')
dset6 = f.create_dataset('/validation/targets', data = targets[0], dtype='int64')

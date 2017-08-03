# Daphnet Gait Data Reader
# Reading in the Daphnet Gait Dataset and creating a corresponding HDF5 File
# Author: Sreela Kodali, kodali@princeton.edu

# Command: python dgdatareader.py
# produces a .hdf5 in the same directory as the .py file

import numpy as np
import h5py
import glob

# DG

# Download the original DG Dataset https://archive.ics.uci.edu/ml/datasets/Daphnet+Freezing+of+Gait
# Download my version with reorganized subdirectories here:
# Set the 'mainpath' to the main DG_dataset directory (I renamed the dataset_fog folder to DG_dataset)
mainpath = '/Users/Sreela/Documents/School/Princeton/Year3/STARnet/Research2017_Sreela/HAR_Datasets/DG_dataset/'

# In my modified version, I split the .txt 'Dataset' files into three subgroups in subdirectories for my clarity.
# Divided into test, validation, and training as described in the Hammerla Paper https://arxiv.org/pdf/1604.08880.pdf
# If you're using the original dataset, do the following:
# Create a subdirectory 'DG_dataset/Test' and put the 'Dataset/S02R0*.txt' files
# Create a subdirectory 'DG_dataset/Val' and put the 'Dataset/S09R01.txt' file
# Create a subdirectory 'DG_dataset/Training' and put the rest of the .txt 'Dataset' files
testpath = mainpath+'Test/'
valpath = mainpath+'Val/S09R01.txt'
trainpath = mainpath+'Training/'

# VALIDATION DATA
v = np.loadtxt(valpath)
inputs_valDG = v[:,1:10] #extracting sensing values
rows = np.shape(inputs_valDG)[0]

targets_valDG = np.zeros((rows,))
targets_valDG = v[:,10]
# Downsample the data from 64HZ to 32Hz like in the Hammerla paper
dsinputs_valDG = inputs_valDG[::2]
dstargets_valDG = targets_valDG[::2]

# TEST
testfiles = glob.glob(testpath + '*.txt')
i = 0
for name in testfiles:
	te = np.loadtxt(name)
	te_rows = np.shape(te)[0]
	a = te[:,1:10] #extracting sensing values
	b = np.zeros((te_rows,))
	b = te[:,10]
	a = a[::2] #Downsample
	b = b[::2] #Downsample
	if (i == 0):
		dsinputs_testDG =  a
		dstargets_testDG = b
		i = i + 1
	else:
		dsinputs_testDG = np.concatenate((dsinputs_testDG, a))
		dstargets_testDG = np.concatenate((dstargets_testDG, b))

# TRAINING
trainfiles = glob.glob(trainpath + '*.txt')
i = 0
for name in trainfiles:
	tr = np.loadtxt(name)
	tr_rows = np.shape(tr)[0]
	a = tr[:,1:10] #extracting sensing values
	b = np.zeros((tr_rows,))
	b = tr[:,10]
	a = a[::2] #Downsample
	b = b[::2] #Downsample
	if (i == 0):
		dsinputs_trainDG =  a
		dstargets_trainDG = b
		i = 1
	else:
		dsinputs_trainDG = np.concatenate((dsinputs_trainDG, a))
		dstargets_trainDG = np.concatenate((dstargets_trainDG, b))

# Remap the target class values from 0 1 2 --> 1 2 3 to adhere to the standards for the FC NN model
dstargets_valDG = dstargets_valDG +1
dstargets_testDG = dstargets_testDG +1
dstargets_trainDG = dstargets_trainDG +1

# Create new hdf5 file with 3 subgroups: test, training, and validation
f = h5py.File('DoG.h5', 'w')
dset1 = f.create_dataset('/test/inputs', data = dsinputs_testDG, dtype='float64')
dset2 = f.create_dataset('/test/targets', data = dstargets_testDG, dtype='int64')
dset3 = f.create_dataset('/training/inputs', data = dsinputs_trainDG, dtype='float64')
dset4 = f.create_dataset('/training/targets', data = dstargets_trainDG, dtype='int64')
dset5 = f.create_dataset('/validation/inputs', data = dsinputs_valDG, dtype='float64')
dset6 = f.create_dataset('/validation/targets', data = dstargets_valDG, dtype='int64')



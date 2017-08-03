# UCISmartphone Preprocessed Data Reader
# Reading in the UCI Smartphone Preprocessed Data and creating a corresponding HDF5 File
# Author: Sreela Kodali, kodali@princeton.edu

# Command: python UCISmartphoneProcessedreader.py
# produces a .hdf5 in the same directory as the .py file

import numpy as np
import h5py
import glob

# Download the original UCI Smartphone Dataset https://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions
# Rename the main folder 'UCISmartphone_Dataset' and set the 'mainpath' to the RawData subdirectory in it
mainpath = '/Users/Sreela/Documents/School/Princeton/Year3/STARnet/Research2017_Sreela/DL_har_datasets/UCISmartphone_Dataset/'
trainpath = mainpath + 'Train/'
testpath = mainpath + 'Test/'

#TRAINING
train_inputs = np.loadtxt(trainpath + 'X_train.txt')
train_targets = np.loadtxt(trainpath + 'y_train.txt')

#TEST
test_inputs = np.loadtxt(testpath + 'X_test.txt')
test_targets = np.loadtxt(testpath + 'y_test.txt')

# Ensure all the different classes are in both datasets
print(np.unique(train_targets))
print(np.unique(test_targets))

# Create new hdf5 file with 2 subgroups: test and training
f = h5py.File('UCISmartphoneProcessed.h5', 'w')
dset1 = f.create_dataset('/test/inputs', data = test_inputs, dtype='float64')
dset2 = f.create_dataset('/test/targets', data = test_targets, dtype='int64')
dset3 = f.create_dataset('/training/inputs', data = train_inputs, dtype='float64')
dset4 = f.create_dataset('/training/targets', data = train_targets, dtype='int64')

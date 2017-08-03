# UCISmartphone Raw Data Reader
# Reading in the UCI Smartphone Raw Data and creating a corresponding HDF5 File
# Author: Sreela Kodali, kodali@princeton.edu

# Command: python UCISmartphoneRawreader.py
# produces a .hdf5 in the same directory as the .py file


import numpy as np
import h5py
import glob


# Helper function that interprets the labels.txt file and populates the
# target output array with the correct class values accordingly
def makeTarget(arr, e, output):
	if (arr[0] == e): 
		start = int(arr[3]-1)
		end = int(arr[4]-1)
		output[start:end+1] = arr[2]

# Download the original UCI Smartphone Dataset https://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions
# Rename the main folder 'UCISmartphone_Dataset' and set the 'mainpath' to the RawData subdirectory in it
mainpath = '/Users/Sreela/Documents/School/Princeton/Year3/STARnet/Research2017_Sreela/HAR_Datasets/UCISmartphone_Dataset/RawData/'

# Knowing the raw dataset has 61 experiment sessions, create 61 regular
# string expressions in reference to each experiment
nExp = 61
expName = ["" for x in range(nExp)]
for i in range(0, nExp):
	if (i+1 < 10):
		expName[i] = '*_exp0'+str(i+1)+'*.txt'
	else:
		expName[i] = '*_exp'+str(i+1)+'*.txt'

# Iterate through the experiments and concatenate
# the acc data and gyro data for each experiment
expnumber = 1
labels = np.loadtxt(mainpath+'labels.txt')
for n in expName:
	files = glob.glob(str(mainpath + n))
	nFiles = 0
	for f in files:
		print(f)
		a = np.loadtxt(f)
		if (nFiles == 0):
			exp = a
			nFiles = 1
		else:
			exp = np.concatenate((exp, a), axis=1)

	# Create a corresponding target output array
	t = np.zeros((np.shape(exp)[0],1))
	for j in range(0, np.shape(labels)[0]):
		makeTarget(labels[j,:], expnumber, t)
	t = t + 1
	# Check if the experiment had instances from all the classes
	print(np.unique(t))
	
	if (expnumber == 1):
		inputs = exp
		targets = t
		print(expnumber)
		print(np.shape(inputs))
	else:
		inputs = np.concatenate([inputs, exp])
		targets = np.concatenate([targets, t])
		print(expnumber)
		print(np.shape(inputs))
	expnumber = expnumber + 1

# Concatenate the inputs and targets to separate into testing and training
final = np.concatenate((inputs, targets), axis=1)

# 70% of the data is to be used to training, 30% for testing. Of the 61 experiments
# 43 ought to be used to for training and the rest for testing.
# Experiment 43 concludes at length 770774, so :770774 includes 70% of the data
x = 770774
y = np.shape(final)[1]-1 # last column of the dataset

#Set the inputs
train_inputs = final[0:x,:y]
test_inputs = final[x:,:y]
#Set the targets
train_targets = np.zeros((np.shape(train_inputs)[0],))
test_targets = np.zeros((np.shape(test_inputs)[0],))
train_targets = final[0:x,y]
test_targets = final[x:,y]

# Ensure all the different classes are in both datasets
print("uniqueness of sets")
print(np.unique(train_targets))
print(np.unique(test_targets))

# Create new hdf5 file with 2 subgroups: test and training
f = h5py.File('UCISmartphoneRaw.h5', 'w')
dset1 = f.create_dataset('/test/inputs', data = test_inputs, dtype='float64')
dset2 = f.create_dataset('/test/targets', data = test_targets, dtype='int64')
dset3 = f.create_dataset('/training/inputs', data = train_inputs, dtype='float64')
dset4 = f.create_dataset('/training/targets', data = train_targets, dtype='int64')

# Sreela Kodali - Reading in iphone data!

import numpy as np
import h5py
import glob
import os

from numpy import genfromtxt
# UCI Smartphone
mainpath = '/Users/Sreela/Downloads/HAR_*.csv'
newest = max(glob.iglob(mainpath), key=os.path.getctime)
print(newest)

start = newest.find('HAR') + 3
end = newest.find('.csv', start)
f = newest[start:end]
print(f)

data = genfromtxt(newest, delimiter=',')


#Test
#test_inputs = np.loadtxt(testpath + 'X_test.txt')
#test_targets = np.loadtxt(testpath + 'y_test.txt')

#print(np.unique(train_targets))
#print(np.unique(test_targets))

f = h5py.File('HAR'+f+'.h5', 'w')
dset1 = f.create_dataset('/test/inputs', data = data, dtype='float64')
#dset2 = f.create_dataset('/test/targets', data = test_targets, dtype='int64')
#dset3 = f.create_dataset('/training/inputs', data = train_inputs, dtype='float64')
#dset4 = f.create_dataset('/training/targets', data = train_targets, dtype='int64')

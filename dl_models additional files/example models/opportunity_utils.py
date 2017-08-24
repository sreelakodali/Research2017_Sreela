# Sreela Kodali - July 18th, 2017
# Fully-connected NN Model for UCI Opportunity Dataset
# Based off of Hammerla et all 2016 paper
import numpy as np
import h5py
import math

def loadHDF5(self, f, label, inputbool):
	data = f[label]
	if inputbool: dtype = 'float32'
	else: dtype = 'float32'

	result = np.zeros(np.shape(data), dtype)
	data.read_direct(result)

	return result

def nRows(self, nSamples):
	ow = int(self.overlap * self.window_size)
	nr = math.floor((nSamples - ow) /(self.window_size - ow))
	nr = int(nr)
	return nr

def nCols(self, inputbool):
	nc = self.nb_classes
	if inputbool:
  		nc = self.window_size * self.sensor_features
  	return nc

def reshape(self, rows, inputbool):
	cols = nCols(self, inputbool)
	if inputbool:
		newShape = np.zeros((rows,cols),'float32')
	else:
       		newShape = np.zeros((rows,cols),'int32')
	return newShape

def expandInput(self, pointer, dataset):
	pointer = int(pointer)
	r = np.zeros((1, self.sensor_features), 'float32')
	for j in range(0, self.window_size):
		j = int(j)
		p = pointer + j
		x = dataset[p, :]
		if j == 0:
			r = x
		else:
			r = np.concatenate([r, x])
	return r

def compressTarget(self, pointer, dataset):
	pointer = int(pointer)

	#copy of output values of that window into r
	r = np.zeros((self.window_size,), 'int32')
	for j in range(0, self.window_size):
		j = int(j)
		p = pointer + j
		r[j] = dataset[p]
	counts = np.bincount(r)
	y = np.argmax(counts)
	output = np.zeros((1,self.nb_classes), 'int32')
	output[0][y-1] = 1
	return output

def precision_recall_nSample(self, predict, target, cl):
	tp = 0
	fp = 0
	fn = 0
	n = 0
	output = np.zeros((1,3))
	rows = np.shape(predict)[0]
	for i in range (0, rows):
		a = predict[i][cl]
		b = target [i][cl]
		if (b == 1):
			n = n + 1
		if (a and b):
			tp = tp + 1
		if ((a == 1) and (b == 0)):
			fp = fp + 1
		if ((a == 0) and (b == 1)):
			fn = fn + 1
	print('tp: '+str(tp))
	print('fp: '+str(fp))
	print('fn: '+str(fn))
	#print('n: '+str(n))
	if ((tp + fn) == 0): recall = 0.0
	else: recall = float(tp) / (tp + fn)
	if ((tp + fp)== 0): precision = 0.0
	else: precision = float(tp) / (tp + fp)
		
	#print('precision,' + str(cl+1) + ': ' + str(precision))
	#print('recall,' + str(cl+1) + ': ' + str(recall))
	print('n,' + str(cl+1) + ': ' + str(n))

	output[0][0] = precision
	output[0][1] = recall
	output[0][2] = n

	return output


def F1score(self, pr_array, sampleTotal):
	mTotal = 0
	wTotal = 0
	output = np.zeros((2,), 'float32')
	for i in range (0, self.nb_classes):
		p = pr_array[i][0]
		r = pr_array[i][1]
		n = pr_array[i][2]

		if (p == 0 or r == 0): v = 0
		else: v = (p * r)/(p + r)
		f = float(n)/sampleTotal

		mTotal = mTotal + v
		wTotal = wTotal + f*v
	
	output[0] = (2*mTotal)/(self.nb_classes) #meanF1
	output[1] = 2*wTotal #weightedF1

	return output




# if p is a one-hot vector
    # tp, total --> p element by element && y_test, total # of ones
    # tp, class --> for a specific column/class, p&&y_test --> total # of ones
    # fp, class --> if predict is 1 and y_test is 0
    # fn, class --> if predict is 0 and y_test is 1

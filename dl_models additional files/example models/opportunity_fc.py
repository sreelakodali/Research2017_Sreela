# Fully-connected NN Model for Human Activity Recognition (HAR)
# Note: the filename is a misnomer - this works on multiple HAR datasets
# Compatible with UCI Opportunity, Daphnet Gait, PAMAP2, UCI Smartphone Raw, UCI Smartphone Processed
# Based of of Hammerla et al 2016 paper

# Author: Sreela Kodali, kodali@princeton.edu
# Last Updated: July 31st, 2017
# Command: python run_models.py opportunity_fc

""" Instructions:
(1) To run the model, make sure it is imported and referred to in #the table in run_models.py
(2) Make sure the run.sh file refers to it accordingly
(3) In the folder .../dl-models/, create a new directory 'HAR'
(4) Create two subdirectories for 'HAR', 'Data' and 'Results'
(5) In the .../dl-models/HAR/Data directory, you can put .hdf5 files for different HAR datasets.
 All datafiles must adhere to the following format:
filename.hdf5
 |_test
   |_ inputs
   |_targets 
 |_training
   |_ inputs
   |_targets 
 ~Optional~
 |_validation
   |_ inputs
   |_targets 
 where
 .../inputs ->  a x b array, a samples, b sensor features; preferably float64 
 .../targets ->  preferably int64 integers, target values MUST be consecutive integers and are numbered starting from 1 (not 0)

 Note: If there are validation sets, make sure to uncomment lines xy for preprocessing and set the last two #parameters in line z for set_data() as x_val and y_val in lieu of x_test and y_test

 Each HAR dataset has a different filename, window sizes, overlap fractions, hidden dimension layer size, no. of sensor features,  and no. of classes.
 If adding a new dataset, add an additional elif() statement to the set of statements on line xyz with the aforementioned parameters.
 self.datafilename = 'name' from name.hdf5
 self.window_size = no. of dataset samples you want in a window;
  for context, a window is the input to the NN so self.window_size is the number of dataset samples that will be concatenated and inputted into the network
 self.overlap = fraction indicating how much one window overlaps with the previously sampled window;
  positive values indicate overlap, negative values indicate space between windows; the no. of samples overlapping = self.window_size * self.overlap
 self.hidden_dim = no. of neurons you to set per hidden layer
 self.sensor_features = no. of discrete parameters from a single dataset sample
 self.nb_classes = no. of classes of the dataset 

Default topology:
Input > hidden_dim > hidden_dim > hidden_dim > hidden dim > nb_classes """

#No. of hidden layers can be altered in the def build_model()

import numpy as np
import os
import h5py
import math
import csv

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.regularizers import l2
from keras.utils import np_utils
from keras import backend
from keras import optimizers

import operator as op
import sys

from dl_models.models.opportunity_utils import *
from dl_models.models.base import *

class opportunityFC(ModelBase):
  def __init__(self):
    super(opportunityFC,self).__init__('opportunity','fc')

    self.batch_size = 64
    self.lr = 0.0001
    self.momentum = 0.9

    #Setting the dataset
    # 0 - opportunity  1 - DG   2 - pamap2  3 - UCISmartphoneRaw  4 - UCISmartphoneProcessed
    
    self.datasetname = 3

    #daphnet gait
    if(self.datasetname == 1):
      self.datafilename = 'DoG_rescale'
      self.window_size = 16
      self.overlap = 0.5
      self.hidden_dim = 112
      self.sensor_features = 9
      self.nb_classes = 3
    #pamap2
    elif(self.datasetname == 2):
      self.datafilename = 'pamap2_rescale_nanLI'#'pamap2_nanLI'
      self.window_size = 18 #170
      self.overlap = -0.1 #-1/5.12
      self.hidden_dim = 55 #512
      self.sensor_features = 52
      self.nb_classes = 13
    #UCI Smartphone Raw
    elif(self.datasetname == 3):
      self.datafilename = 'UCISmartphoneRawiPhone'
      self.window_size = 64
      self.overlap = .5
      self.hidden_dim = 72
      self.sensor_features = 6
      self.nb_classes = 13
    #UCI Smartphone Processed
    elif(self.datasetname == 4):
      self.datafilename = 'UCISmartphoneProcessed'
      self.window_size = 1
      self.overlap = 0.0
      self.hidden_dim = 70
      self.sensor_features = 561
      self.nb_classes = 12
    #opportunity
    else:
      self.datafilename = 'opportunity'
      self.window_size = 12
      self.overlap = 0.5
      self.hidden_dim = 56
      self.sensor_features = 77
      self.nb_classes = 18

    self.main_dir = os.path.abspath('HAR')
    self.data_dir = self.main_dir+'/data/'
    self.results_dir = self.main_dir+'/results/'
    self.input_shape = (self.window_size * self.sensor_features,)

    # NOTE: Adjust for different no. of layers
    self.param_layer_ids       = ['Dense', 'Bias', 'Dense', 'Bias', 'Dense', 'Bias', 'Dense','Bias', 'Dense', 'Bias']
    self.default_prune_factors = ['0.001', '0.001','0.001', '0.001','0.001', '0.001', '0.001', '0.001', '0.001', '0.001']
  
  def preprocess_categorical(self, dataset, inputbool):
  	#for inputs, takes in 2D array dataset --> reorganizes for windows
  	#for targets, consolidates every 30 into 1

    samples = np.shape(dataset)[0]
    nRows_reshaped = nRows(self, samples)
    new = reshape(self, nRows_reshaped, inputbool)
    step = math.floor(self.window_size*(1 - self.overlap))

    k = 0
    for i in range(0, nRows_reshaped):
      if inputbool:
        new[i,:] = expandInput(self, k, dataset)
      else:
        new[i,:] = compressTarget(self, k, dataset)
      k = k + step
    return new

  
  def load_dataset(self, ):
  	print('Loading Dataset:')
        print(self.datafilename)

  	#load data from .hdf5 file
  	f = h5py.File(self.data_dir+self.datafilename+'.h5','r')
  	x_train = loadHDF5(self, f, '/training/inputs', 1)
  	y_train = loadHDF5(self, f, '/training/targets', 0)
  	x_test = loadHDF5(self, f, '/test/inputs', 1)
  	y_test = loadHDF5(self, f, '/test/targets', 0)
  	#x_val = loadHDF5(self, f, '/validation/inputs', 1)
  	#y_val = loadHDF5(self, f, '/validation/targets', 0)
  	f.close()

  	#preprocess for windows
  	x_train = self.preprocess_categorical(x_train, 1)
  	y_train = self.preprocess_categorical(y_train, 0)
  	x_test = self.preprocess_categorical(x_test, 1)
  	y_test = self.preprocess_categorical(y_test, 0)
  	#x_val = self.preprocess_categorical(x_val, 1)
  	#y_val = self.preprocess_categorical(y_val, 0)
        #print('mean x_test: ' + str(np.mean(x_test)))
        #print('mean y_test: ' + str(np.mean(y_test)))
        #print('stdev x_test: ' + str(np.std(x_test)))
        #print('stdev y_test: ' + str(np.std(y_test)))
  
  	self.set_data(x_train, y_train, x_test, y_test, x_test, y_test)

  
  def build_model(self,):
  	model = Sequential()
  	
  	#input layer
  	model.add(Dense(self.hidden_dim, input_shape=self.input_shape))
  	model.add(Activation('relu'))

  	# hidden layer 1
        model.add(Dense(self.hidden_dim))
        model.add(Activation('relu'))

        # hidden layer 2
        model.add(Dense(self.hidden_dim))
        model.add(Activation('relu'))

        # hidden layer 3
        model.add(Dense(self.hidden_dim))
        model.add(Activation('relu'))

        # hidden layer 4
        #model.add(Dense(self.hidden_dim))
        #model.add(Activation('relu'))

        # hidden layer 5
        #model.add(Dense(self.hidden_dim))
        #model.add(Activation('relu'))

    #last layer
        model.add(Dense(self.nb_classes))
        model.add(Activation('softmax'))

        self.set_model(model, self.param_layer_ids, self.default_prune_factors)
    
#  def fit_model(self, ):
#    self.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=self.num_epochs, verbose=1, validation_data=((self.x_val, self.y_val))


  def eval_model(self, ):
    # precision = tp / (tp + fp)
    # recall = tp / (tp + fn)
    # find precision and recall per class

    p = self.model.predict(self.x_test)
    p = np.rint(p)
    prn_values = np.zeros((self.nb_classes, 3), 'float64')
    for c in range(0, self.nb_classes):
      prn_values[c,:] = precision_recall_nSample(self, p, self.y_test, c)
    
    filename = str(self.datafilename)+'_'+str(self.window_size)+'w_'+str(abs(int(self.overlap*100)))+'o_'+str(self.hidden_dim)+'hd_5L.txt'
    f = open(self.results_dir+filename,"w+")
    np.savetxt(self.results_dir+filename, prn_values)
    f.close()

    N = np.shape(self.y_test)[0]
    #print('N:'+str(N))
    result = F1score(self, prn_values, N)
    f1_mean = result[0]
    f1_weight = result[1]

    trainPerformance = self.model.evaluate(self.x_train, self.y_train, verbose=0)
    testPerformance = self.model.evaluate(self.x_test, self.y_test, verbose=1)

    #print('Train loss:', trainPerformance[0])
    #print('Test loss:', testPerformance[0])
    print('Train accuracy:', trainPerformance[1])
    print('Test accuracy:', testPerformance[1])
    print('Mean F1 score: ' + str(f1_mean))
    print('Weighted F1 score: ' + str(f1_weight))
    
    #return f1_mean


if __name__ == '__main__':
  model = opportunityFC()
  model.load_dataset()
  model.build_model()
  model.compile_model()
  model.fit_model()
  model.eval_model()

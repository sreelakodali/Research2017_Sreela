# Human Activity Recognition (HAR) Demo: Processing Raw iPhone Acc/Gyro Data
# Author: Sreela Kodali, kodali@princeton.edu
# Updated last: August 13th, 2017
# 
# The program finds the most recent HAR_*.csv input data generated from the HAR iOS Data Logger app,
# preprocesses the data into windows, and saves it as an .npz format that is accessible to the
# Models2Chip code. The program then calls gen_acts_hex() from Models2Chip to create hex 
# files for all the preprocessed input data.
# livephoneDataReader.py is in the Models2Chip directory --> models2chip/dnn_models/scripts/livephoneDataReader.py
# This program is executed every time the "Process" button in the HAR GUI is clicked.
#
# Example call:
# python -u livephoneDataReader.py -d UCISmartphoneRaw -t 72 72 72 72 -l TC


import argparse
import numpy as np
import math
import h5py
import glob
import os
import gen_acts_hex
import run_demo_model
from livephone_utils import * 
from sim_config import *


def cli():
  # Command Line Interface
  parser = argparse.ArgumentParser()
  parser.add_argument('-d','--dataset',help='which dataset to use')
  parser.add_argument('-e','--example',type=int,nargs="+",help='zero-indexed example or inclusive range of examples to generate .hex for (none defaults to all examples)')
  parser.add_argument('-t','--topology',type=int,nargs="+",help='which topology/model to use (hidden layers separated by a space)')
  parser.add_argument('-l','--lane',choices=['SM','TC'],default=DEFAULT_LANE,help='which numerical representation to use')
  parser.add_argument('-8b','--megamem_8b',action='store_true',help='weights stored as 8-bit values (16-bit if False)')
  parser.add_argument('-bw','--megamem_bw',type=int,default=default_megamem_bw,help='megabyte memory bandwidth')
  parser.add_argument('-ncyc','--megamem_ncyc',type=int,default=default_megamem_ncyc,help='megabyte memory number of cycles')
  parser.add_argument('-b','--bypass',nargs='+',default=default_bypass,help='bypass configurations (e.g., fine/coarse 0.1)')
  parser.add_argument('-s','--save',help='save results as .txt',action='store_true')
  parser.add_argument('-v','--verbose',help='increase output verbosity',action='store_true')
  args = parser.parse_args()

  # ensure proper usage
  assert(args.dataset), 'Dataset required'
  assert(args.topology), 'Topology required'

  return args


if __name__=='__main__':
  args = cli()

  # Set rawDatapath to where the .csv file from the ios app is deposited
  rawDatapath = '/Users/Sreela/Downloads/'

  # Find the most recent HAR raw .csv file
  mainpath = rawDatapath + 'HAR_*.csv'
  newest = max(glob.iglob(mainpath), key=os.path.getctime)

  # Extract the file name
  f = filename(newest)

  # Extract raw data from .csv file
  rawData = np.genfromtxt(newest, delimiter=',')

  # Preprocessing parameters
  # overlap: fraction in which adjacent windows overlap --> between 0.0 and 1.0
  # window_size: number of raw samples in a window sample
  # sensor_features: number of features in each raw sample
  overlap = .5 
  window_size = 64
  sensor_features = 6

  # Preprocess the data into windows
  testData = preprocessRawInput(rawData, overlap, window_size, sensor_features)

  # Extract number of samples - 1 in testData
  e = np.shape(testData)[0] - 1

  # Set MtoCpath to where models2chip is located
  MtoCpath = '/Users/Sreela/Documents/School/Princeton/Year3/STARnet/Research2017_Sreela/models2chipCopy/'

  # Save preprocessed data as .npz
  npzpath = MtoCpath + 'datasets/UCISmartphoneRaw'
  np.savez(os.path.join(npzpath, 'HAR'+f), x_test=testData)

  # Set the example range for gen_acts_hex()
  example = [0, e]

  # Pass the number of test samples 's' to the HAR GUI
  s = e + 1
  print(s)

  # Generate the activation hex files for the new input data
  # --verbose is set as 'False' to avoid passing additional text to the GUI
  gen_acts_hex.main(args.dataset, example, args.lane, False)

  # Uncomment the following line to run the software inference model -- keep it commented if using the GUI and/or chip
  #run_demo_model.main(args.dataset, args.topology, example, args.lane, args.megamem_8b, args.save, args.verbose)

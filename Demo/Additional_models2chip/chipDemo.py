# Human Activity Recognition (HAR) Demo: Running Test Samples on the Chip
# Author: Sreela Kodali, kodali@princeton.edu
# Updated last: August 14th, 2017
# 
# The program calls run_demochip_prediction.py over an inputted example range.
# chipDemo.py is in the Models2Chip directory --> models2chip/si_test/scripts/chipDemo.py
# This program is executed every time the "Predict" button in the HAR GUI is clicked.
#
# Example call:
# python -u chipDemo.py -d UCISmartphoneRaw -e 0 10

import argparse
import glob
import os, os.path
import run_demochip_prediction
from dataset_utils import *


def cli():
  # Command Line Interface
  parser = argparse.ArgumentParser()
  parser.add_argument('-d','--dataset',help='which dataset to use')
  parser.add_argument('-e','--example',type=int,nargs="+",help='zero-indexed example or inclusive range of examples to generate .hex for (none defaults to all examples)')
  parser.add_argument('-t','--topology',type=int,nargs="+",help='which topology/model to use (hidden layers separated by a space)')
  parser.add_argument('-l','--lane',choices=['SM','TC'],default=DEFAULT_LANE,help='which numerical representation to use')
  parser.add_argument('-8b','--megamem_8b',action='store_true',help='weights stored as 8-bit values (16-bit if False)')
  parser.add_argument('-v','--verbose',help='increase output verbosity',action='store_true')
  args = parser.parse_args()

  # ensure proper usage
  assert(args.dataset), 'Dataset required'

  return args

if __name__=='__main__':
	args = cli()

  # Preset the topology for this particular model
	topology = [72, 72, 72, 72]

  # Preset the --topology with the constant set up above, but it can be changed if a -t is passed through when calling chipDemo.py
  # and if 'topology' is replaced in the line below with 'args.topology'
	run_demochip_prediction.main(args.dataset, topology, args.example, args.lane, args.megamem_8b, args.verbose)
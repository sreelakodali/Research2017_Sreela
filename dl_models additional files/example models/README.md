# HAR: Fully-connected NN Model
Note: the filenames are misnomers - this works on multiple HAR datasets
Compatible with UCI Opportunity, Daphnet Gait, PAMAP2, UCI Smartphone Raw, UCI Smartphone Processed

This 5-layered model is based off of Hammerla et al 2016 paper: https://arxiv.org/pdf/1604.08880.pdf

These files can be found in dl-models, in the **opportunityHAR branch**, or in this path:
/group/vlsiarch/kodali/dl-models

## Instructions
1) To run the model, make sure it is imported and referred to in #the table in run_models.py
2) Make sure the run.sh file refers to it accordingly
3) In the folder .../dl-models/, create a new directory 'HAR'
4) Create two subdirectories for 'HAR', 'Data' and 'Results'
5) In the .../dl-models/HAR/Data directory, you can put .hdf5 files for different HAR datasets.
 All datafiles must adhere to the following format:
 
**filename.hdf5**
- **test**
    - inputs
    - targets 
- **training**
    - inputs
    - targets 
- **Optional: validation**
    - inputs
    - targets 
    
 where
 
 .../inputs ->  a x b array, a samples, b sensor features; preferably float64 
 
 .../targets ->  preferably int64 integers, target values MUST be consecutive integers and are numbered starting from 1 (not 0)

Note: If there are validation sets, make sure to uncomment lines 161-162/170-171 for preprocessing and set the last two parameters in line 177 for set_data() as x_val and y_val in lieu of x_test and y_test

## Parameters for Datasets
Each HAR dataset has a different filename, window sizes, overlap fractions, hidden dimension layer size, no. of sensor features,  and no. of classes.
If adding a new dataset, add an additional elif() statement to the set of statements on line 114 with the aforementioned parameters.
* **self.datafilename** = 'name' from name.hdf5
* **self.window_size** = no. of dataset samples you want in a window;
  for context, a window is the input to the NN so self.window_size is the number of dataset samples that will be concatenated and inputted into the network
* **self.overlap** = fraction indicating how much one window overlaps with the previously sampled window;
  positive values indicate overlap, negative values indicate space between windows; the no. of samples overlapping = self.window_size * self.overlap
* **self.hidden_dim** = no. of neurons you to set per hidden layer
* **self.sensor_features** = no. of discrete parameters from a single dataset sample
* **self.nb_classes** = no. of classes of the dataset 

## Default topology:
Input > hidden_dim > hidden_dim > hidden_dim > hidden dim > nb_classes

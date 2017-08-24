# Fully-connected NN Model for Human Activity Recognition (HAR)
Note: the filename is a misnomer - this works on multiple HAR datasets
Compatible with UCI Opportunity, Daphnet Gait, PAMAP2, UCI Smartphone Raw, UCI Smartphone Processed
Based off of Hammerla et al 2016 paper


Each HAR dataset has a different filename, window sizes, overlap fractions, hidden dimension layer size, no. of sensor features,  and no. of classes.
If adding a new dataset, add an additional elif() statement to the set of statements on line xyz with the aforementioned parameters.
* self.datafilename = 'name' from name.hdf5
* self.window_size = no. of dataset samples you want in a window;
  for context, a window is the input to the NN so self.window_size is the number of dataset samples that will be concatenated and inputted into the network
* self.overlap = fraction indicating how much one window overlaps with the previously sampled window;
  positive values indicate overlap, negative values indicate space between windows; the no. of samples overlapping = self.window_size * self.overlap
* self.hidden_dim = no. of neurons you to set per hidden layer
* self.sensor_features = no. of discrete parameters from a single dataset sample
* self.nb_classes = no. of classes of the dataset 

###Default topology:
Input > hidden_dim > hidden_dim > hidden_dim > hidden dim > nb_classes

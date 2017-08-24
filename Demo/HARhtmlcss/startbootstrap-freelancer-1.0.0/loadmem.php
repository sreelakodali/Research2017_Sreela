<?php 
# Human Activity Recognition (HAR) Demo: Loading Weights into Chip Memory
# Author: Sreela Kodali, kodali@princeton.edu
# Updated last: August 13th, 2017
# This program calls the load_megamem.py script to load the weights of the DNN model into the chip's memory.
# loadmem.php is called when the 'Memory' button in the GUI is clicked.

# Set path to the models2chip directory
$path = "/Users/Sreela/Documents/School/Princeton/Year3/STARnet/Research2017_Sreela/models2chipCopy";

$command = "python -u ";
$suffix = "/si_test/scripts/load_megamem.py -d UCISmartphoneRaw -t 72 72 72 72 -l TC -v";
$command .= $path;
$command .= $suffix;
exec($command);
?>
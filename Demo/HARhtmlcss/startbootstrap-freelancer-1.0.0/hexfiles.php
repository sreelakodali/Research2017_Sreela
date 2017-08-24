<?php 
# Human Activity Recognition (HAR) Demo: Processing Raw iPhone Acc/Gyro Data
# Author: Sreela Kodali, kodali@princeton.edu
# Updated last: August 13th, 2017
# This program calls the livephoneDataReader.py script to find the most recent raw data and generate its hex files.
# hexfiles.php is called when the 'Process' button in the GUI is clicked and returns the number of test samples.

# Set path to the models2chip directory
$path = "/Users/Sreela/Documents/School/Princeton/Year3/STARnet/Research2017_Sreela/models2chipCopy";

$command = "python -u ";
$suffix = "/dnn_model/scripts/livephoneDataReader.py -d UCISmartphoneRaw -t 72 72 72 72 -l TC";
$command .= $path;
$command .= $suffix;
$output = exec($command);
echo $output;
?>
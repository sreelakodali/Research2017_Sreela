<?php
# Human Activity Recognition (HAR) Demo: Running Test Samples on the Chip
# Author: Sreela Kodali, kodali@princeton.edu
# Updated last: August 14th, 2017
# This program calls the chipDemo.py script to run the example activation hex files on the chip.
# chipPredict.php is called when the 'Predict' button in the GUI is clicked and returns predicted activity.

# Set path to the models2chip directory
$path = "/Users/Sreela/Documents/School/Princeton/Year3/STARnet/Research2017_Sreela/models2chipCopy";

# Get example number
$e = $_GET['clickCount'];

$base = "python -u ";
$suffix = "/si_test/scripts/chipDemo.py -d UCISmartphoneRaw -e ";
$base .= $path;
$base .= $suffix;

$j = "$e";
$j .= " ";
$j .= "$e";
$base .= $j;
$command = escapeshellcmd($base);
$output = shell_exec($command);
echo $output;
?>
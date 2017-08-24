# Human Activity Recognition Demo
### Author: Sreela Kodali, kodali@princeton.edu
This demo aims to use smartphone data and predict a user's physical activity - known as Human Activity Recognition (HAR).

# How Does It Work
1) Raw accelerometer and gyroscope data is collected from an iPhone and saved as a .csv file.
2) The .csv file is sent to a local laptop via bluetooh AirDrop*
3) On the laptop, scripts are run to preprocess the raw data and generate input activation files.
4) With USB-UART, weights of a trained HAR DNN model are loaded from the laptop into the memory of a deep learning accelerator.
5) A script is run on the laptop to run the input files on the deep learning accelerator. The predicted activity is displayed in a friendly interface.
* Note: if the local laptop is not an Apple device with the bluetooth AirDrop feature, the .csv file can also be sent via email/text. The iOS data collection application will prompt which method to send the raw data.

# Components
### 1. iPhone with iOS Data Collection App
Learn more about the app and get it here: https://github.com/sreelakodali/HAR_DataApp_iOS
### 2. Laptop with internet/bluetooth capabilities
Self explanatory. If the laptop is an Apple device, AirDrop can be used to send the iOS data directly to the Downlaods directory of the laptop.
### 3. Preprocessing/Accelerator python scripts
All these scripts are in models2chip in the 'Sreela_HAR' branch. They can also be found in my directory: /group/vlsiarch/kodali/models2chip. Some files are new additions while others are modifications of existing files. In your version of models2chip, make sure to place and update the files in the appropriate models2chip directories.
#### New Additional Files
1) **livephoneDataReader.py** - The program finds the most recent HAR_.csv input data generated from the HAR iOS Data Logger app,
 preprocesses the data into windows, and saves it as an .npz format that is accessible to the Models2Chip code. The program then calls gen_acts_hex() from Models2Chip to create hex files for all the preprocessed input data.
     * located —> dnn_models/scripts
2) **livephone_utils.py** - This file has the helper functions for livephoneDataReader.py
     * located —> dnn_models/scripts
3) **run_demo_model.py** - The program is a varied version of run_sw_model.py. It just outputs the predicted activity - it does not produce performance metrics since it does not know the actual labels for the test data. The main function can be called in livephoneDatareader.py
     * located —> dnn_models/scripts
4) **chipDemo.py** - The program calls run_demochip_prediction.py over an inputted example range. chipDemo.py is in the Models2Chip directory --> models2chip/si_test/scripts/chipDemo.py. This program is executed every time the "Predict" button in the HAR GUI is clicked.
     * located —> si_test/scripts
5) **run_demochip_prediction.py** - The program is a varied version of run_sm2_prediction.py. It just outputs the predicted activity - it does not produce performance metrics since it does not know the actual labels for the test data. The main function is called in chipDemo.py
     * located —> si_test/scripts
6) **dnn_Demo.py** - The program is a varied version of dnn.py. It eliminates the comparison with y-test/actual labels so there are no accuracy and performance metrics and most print statements are eliminated to ensure only the activities are passed through to the HAR GUI. Also, the number of bytes allocated to get the output values is rounded up to an increment of 8 bytes.
     * located —> si_test/scripts
#### Modified Files
1) **dnn_models/scripts/dataset_utils.py**
    * changed get_dataset_npz_path()
    * changed get_dataset_inputs_dir()
2) **si_test/scripts/dataset_utils.py**
    * changed get_dataset_npz_path()
    * changed get_dataset_inputs_dir()
3) **dnn_models/scripts/paths.py, si_test/scripts/path.py, and si_test/scripts/sm2_usb_devices.py**
    * update as per your configuration

### 4. GUI
The GUI can be found in the HARhtmlcss/startbootstrap-freelancer-1.0.0 directory. 
This GUI was created to have a centralized interface to run all the preprocessing/accelerator python scripts and see the predicted activity. The GUI was designed using HTML/CSS with bootstrap templates. The interface has 3 buttons:
1) **Process** - an Ajax request is associated with the button so once 'Process' is clicked, hexfiles.php executes livedatareader.py; the number of test samples is returned
2) **Memory** - an Ajax request is associated with the button so once 'Memory' is clicked, loadmem.php executes load_megamem.py
3) **Predict** - an Ajax request is associated with the button so once 'Predict' is clicked, chipPredict.py executes chipDemo.py n times for n test samples and returns the predicted activities; n test samples is extracted from when 'Process' was clicked

In the HARhtmlcss/startbootstrap-freelancer-1.0.0 directory, one will find index.html, css style formats, support files, and php scripts to help execute the aforementioned preprocessing/accelerator python scripts. In the three php scripts (hexfiles, loadmem, and chipPredict), make sure to change the directory to the models2chip directory.
To run the GUI, navigate to the HARhtmlcss/startbootstrap-freelancer-1.0.0 directory and set up a php server using the following command in Terminal:
```
  php -S localhost:8000
```
Open a browser and access the following link to view the GUI:
```
  http://localhost:8000/
```
### 5. Deep learning accelerator and USB cable
Nothing much to say here.



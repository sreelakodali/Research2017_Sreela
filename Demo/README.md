# Human Activity Recognition Demo
Author: Sreela Kodali, kodali@princeton.edu
This demo aims to use smartphone data and predict a user's physical activity - known as Human Activity Recognition (HAR).

# How Does It Work
1) Raw accelerometer and gyroscope data is collected from an iPhone and saved as a .csv file.
2) The .csv file is sent to a local laptop via bluetooh AirDrop*
3) On the laptop, scripts are run to preprocess the raw data and generate input activation hex files.
4) With USB-UART, weights of a trained HAR DNN model are loaded from the laptop into the memory of a deep learning accelerator.
5) A script is run on the laptop to run the input hex files on the deep learning accelerator. The predicted activity is displayed in a friendly interface.
* Note: if the local laptop is not an Apple device with the bluetooth AirDrop feature, the .csv file can also be sent via email/text. The iOS data collection application will prompt which method to send the raw data.

# Components
1) iPhone with Data Collection App
2) Laptop with internet/bluetooth capabilities
3) Preprocessing python scripts and python scripts to communicate with the deep learning accelerator
4) HTML/CSS/php GUI
5) Deep learning accelerator and USB cable


The full demo takes raw acc/gyro data from an iPhone, generates the input activation hex files, runs the hex files on the chip, and outputs the predicted physical activity.

Additional Files:
(1) In models2chip/
livephoneDataReader.py —> dnn_models/scripts
livephone_utils.py  —> dnn_models/scripts
run_demo_model.py  —> dnn_models/scripts
chipDemo.py  —> si_test/scripts
run_demochip_prediction.py  —> si_test/scripts

(2) HTML/CSS/PHP GUI
HARhtmlcss/startbootstrap-freelancer-1.0.0/*

(3) iOS Data Collection App

Modified Files in models2chip:
(1) dnn_models/scripts/dataset_utils.py
changed get_dataset_npz_path()
changed get_dataset_inputs_dir()

(2) si_test/scripts/dataset_utils.py
changed get_dataset_npz_path()
changed get_dataset_inputs_dir()

(3) si_test/scripts/dnn.py
made sure bytes allocated were increments of 8 and no. of classes evaluated was correct
commented out most print statements (i.e. line 58, 116, 121, 124, 154, 157, 308, 310, 328, 362)
commented out lines related to comparing with labeled results (i.e. y_test = load_from_dataset_npz—> if is_one_hot();
label_correct = hw_label —> label_margin = classes…; all the results except hw_label)

Note: Make sure to update dnn_models/scripts/paths.py, si_test/scripts/path.py, and si_test/scripts/sm2_usb_devices.py
for your configuration.

To run the demo, set up a php server in the HARhtmlcss/startbootstrap-freelancer-1.0.0 directory using the following command:

  php -S localhost:8000

Open a browser and access the following link to view the GUI:

  http://localhost:8000/

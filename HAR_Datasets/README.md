# HAR: Dataset Extraction and Preprocessing Scripts

Each python script corresponds to a specific HAR dataset and transforms the raw data available online into a usable .hdf5 format with training, testing, and validation sets. Follow the instructions denoted in each script.

The datasets are:
1. Opportunity* - Activities of Daily Living
    * https://archive.ics.uci.edu/ml/datasets/opportunity+activity+recognition
2. PAMAP2 - Activities of Daily Living
    * https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring
3. Daphnet Freezing of Gait - Freezing Incidents for patients with Parkinson's Disease
    * https://archive.ics.uci.edu/ml/datasets/Daphnet+Freezing+of+Gait
4. Smartphone-based HAR - Basic Physical Activities
    * https://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions

**HAR_Dataset.zip** has the resulting .hdf5 files for each dataset. It can be found here: https://drive.google.com/open?id=0B1gxxIIZaFQWWkFiUC00Y01XRE0

* Note: The processing script for the Opportunity dataset can be found here: https://github.com/nhammerla/deepHAR
The code for Opportunity is based off of the 2016 N Hammerla et al HAR paper: https://arxiv.org/pdf/1604.08880.pdf

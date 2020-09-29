###AD_multimodal_domain_adaptation: Implementation of variational auto encoder and adversial training on domain adaptation 

ADNI_info:
ADN1_Annual_2_Yr_1.5T_12_15_2019.csv- all raw mri information
DXSUM_PDXCONV_ADNIALL.csv- demographics and annotations

configurations: 

paths: pathes and files names
modelConfig: layers' configuration and hyperparameters


data preparation:
 
Step 1: in case you want to train MRI only, run getalladni1mriLabel.py to generate mri_label.csv file. in case of you want to correlates PET with MRI scans, first run getpetCurrentLabel.py to generate pet_label.csv, then run getmriCurrentLabel.py, to generate mri_label.csv. 

Step 2: create data in h5py format, three-way classification: run createmri3Dh5pyData.py for creating mri_data.hdf5, run createpet3dh5pyData.py for creating pet_data.hdf5; binary_classification: refer to crate3Dh5pybinary_mri.py and create3Dh5pyninary_pet.py.

step3: run getRIDroMRImapping.py and getRIDroPETmapping.py: geenrate patient-specif collection of scans according to their ID, prepare for data split 

step4: run splitDataset.py: split data into training, validation and testing and save in .pkl files. 

Models:

utils: utility functions, such as loss functions, save trained models, plot classification metrics etc. 

main_multi.py: main script for calling training, validation and test









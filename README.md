###AD_multimodal_domain_adaptation: Implementation of variational auto encoder and adversial training on domain adaptation 

Data used conatains both ADNI MRI and PET. All MRI has been preprocessed with skull stripped and spatial normalized. You can also find a running version of the code and the preprocessed data on the IIAT server under Boris's folders. 

ADNI_info:
ADN1_Annual_2_Yr_1.5T_12_15_2019.csv- all raw mri information
DXSUM_PDXCONV_ADNIALL.csv- demographics and annotations

configurations: 

paths: pathes and files names
modelConfig: layers' configuration and hyperparameters


data preparation:
 
1) generate label csv files: 
you can refer to getpetCurrentLabel.py (also applicable for MRI) that is to access the folders where you store the MRI/PET .nii and read all the file names. Then it opens the ADNI1_Annual_2_Yr_1.5T_12_15_2019.csv (meta info for all the MRI data you've preprocessed) as a dataframe, and read line by line to extract RID, and identify corresponding .nii files (each RID could have multiple scans). For the selected .nii files, it further extracts the scan data and compare with the exam data in the dataframe. It will reject the files if the difference between the two is larger than 180 days. Finally, it creates a new dataframe to store the selected files info ('file name', 'diagnosis labels' etc), and name it as mri_labels.csv/pet_labels.csv
you may find getmriCurrentLable.py is slightly different. This script does the same thing except that it keeps all available .nii files for each RID detected (without considering the exam data). 
2) create h5py files (MRI/PET): you can refer to createmri3Dh5pyData.py. This script opens the mri_labels.csv/pet_labels.csv (created from last step) as a dataframe and reads line by line, and loads MRI/PET volumes from corresponding .nii files, and save the original data matrices into a h5py file, named mri_data.hdf5/pet_data.hdf5 (it also saves the file names, diagnosis labels and RIDs).
3) generate a unique index for each MRI/PET volume and maps them to RIDs: you can refer to getRIDroMRImapping.py. This script opens the MRI/PRT h5py files (created from last step) and assign a unique index to each MRI/PET volume and then map the indices to RIDs. Finally, it gives you a pickle file, named RIDtoMRIdict.pkl/RIDroPETdict.pkl，in which you will see each RID has a collection of indices (corresponding MRI/PET volumes).  
4) split data into training , validation and testing (based on RIDs): You can refer to splitDataset.py. This is to do a patient-level split rather an image-level split to guarantee no overlap between trainning, validation and test (We train on some patients and validate/test on unseen patients). you can specify the percentage of RIDs (each RID could have multiple scans) you'd like to train on, then the rest is split one half for validation and the other half for testing. This script returns you three .pkl files (store the training, validation and testing image indices, respectively).

Models: vae.py where the backbone defined i.e. a 3D autoencoder and a domain discriminator

utils: utility functions, such as loss functions, save trained models, plot classification metrics etc. 

main_multi.py: main script for calling training, validation and test





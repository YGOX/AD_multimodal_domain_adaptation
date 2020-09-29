'''
contains paths used in the project
'''
from configurations.modelConfig import params
import time
		

set_to_use = params['train']['set_to_use']    
    
paths = {
	'data' : {
		'hdf5_path'								:	'/app/boris/hdf5', #'/home/ml/sbasu11/Documents/ADNI Project/ADNI_data/CNN/Inputs/',
		#'/home/NOBACKUP/sbasu11/',
		'Input_to_Training_Model' 				: 	'/app/boris/multimodal_AD_BL/ADNI_info_'+set_to_use+'/',
		'ADNI_study_data_original_labels'		: 	'/app/boris/multimodal_AD_BL/ADNI_info_'+set_to_use+'/',
		'Raw_PET_location'						: 	'/app/boris/data/PET_crop/*.nii',
		'Raw_MRI_location'						: 	'/app/boris/data/MRI/*.nii.gz'},
	'output'	:	{
		'pet_base_folder'							:	'/app/boris/data/pet_outputs/',
		'mri_base_folder'							:	'/app/boris/data/mri_outputs/',
		'vae_base_folder'							:	'/app/boris/data/vae_outputs/'
	}
}

file_names = {
	'data'	:	{
		'pet_hdf5_file'							: 	'pet_data.hdf5', #'3DMRItoNextLabel.hdf5',
		'mri_hdf5_file'							: 	'mri_data.hdf5',
		'ad_nc_pet_hdf5_file'							: 	'ad_nc_pet_data.hdf5',
		'ad_nc_mri_hdf5_file'							: 	'ad_nc_mri_data.hdf5',
		'ADNI_study_data_original_labels'		:	'DXSUM_PDXCONV_ADNIALL.csv',
		'ADNI1_MRI_data_original_labels': 'ADNI1_Annual_2_Yr_1.5T_12_15_2019.csv',
		'PET_to_curr_label_mapping'				:	'pet_labels.csv',
		'MRI_to_curr_label_mapping'				:	'mri_labels.csv',
		'pet_ad_nc_mapping'				:	'pet_ad_nc_labels.csv',
		'mri_ad_nc_mapping'				:	'mri_ad_nc_labels.csv',
		'MRI_list'				:	'mri_files.txt',
		'RIDtoPET'								:	'RIDtoPETdict.pkl',
		'RIDtoMRI'								:	'RIDtoMRIdict.pkl',
		'ad_nc_pet_rid'								:	'ad_nc_pet_rid.pkl',
		'ad_nc_mri_rid'								:	'ad_nc_mri_rid.pkl',
		'Train_pet_indices'						: 	'train_pet_indices.pkl',
		'Valid_pet_indices'						: 	'valid_pet_indices.pkl',
		'Test_pet_indices'						: 	'test_pet_indices.pkl',
		'Train_ad_nc_pet_indices'						: 	'train_ad_nc_pet_indices.pkl',
		'Valid_ad_nc_pet_indices'						: 	'valid_ad_nc_pet_indices.pkl',
		'Test_ad_nc_pet_indices'						: 	'test_ad_nc_pet_indices.pkl',
		'Train_ad_nc_mri_indices'						: 	'train_ad_nc_mri_indices.pkl',
		'Valid_ad_nc_mri_indices'						: 	'valid_ad_nc_mri_indices.pkl',
		'Test_ad_nc_mri_indices'						: 	'test_ad_nc_mri_indices.pkl',
		'Train_mri_indices'						: 	'train_mri_indices.pkl',
		'Valid_mri_indices'						: 	'valid_mri_indices.pkl',
		'Test_mri_indices'						: 	'test_mri_indices.pkl'
	},
	'output'	:	{
		'parameters'							:	'parameters.json',
		'train_loss_classification'				:	'train_loss_classification.pkl',
		'valid_loss'							:	'valid_loss.pkl',
		'train_accuracy'						:	'train_accuracy.pkl',
		'valid_accuracy'						:	'valid_accuracy.pkl',
		'valid_bal_accuracy'					:	'valid_bal_accuracy.pkl',
		'train_f1_score'						:	'train_f1_score.pkl',
		'valid_f1_score'						:	'valid_f1_score.pkl',
		'best_val'						:	'valid_best.txt',
		'test_results'						:	'test_results.txt',
		'test_scores'							: 	'test_scores.hdf5'
	}
}

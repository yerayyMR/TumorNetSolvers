import os
from TumorNetSolvers.preprocessing.data_preprocessor import preparingDataset, create_json_file
# =============================================
# Overview:
# This script prepares a tumor growth simulation dataset, specifically designed for use with the nnU-Net framework.
# It takes input images and tumor simulation parameters, performs necessary preprocessing (such as shifting,
# cropping, and downsampling), and structures the data in the format required by nnU-Net.
# 
# Main steps include:
# - Loading tumor simulation parameters fom JSON file & deriving muD, muRho
# - Preprocessing image data: shifting tumor center of mass to the image center, cropping & downsampling for uniformity + reducing memory costs
# - Saving the processed images in the required directories: `imagesTr` for training images and `labelsTr` for ground truth labels.
# - Generating a `dataset.json` file that summarizes the dataset, ensuring compatibility with nnU-Net.
# 
# The output will be stored in the following directories:
# - `raw_data/dataset_id_anatomical_struct/imagesTr`: Processed image data (e.g., raw_data/700_Brain/imagesTr)
# - `raw_data/dataset_id_anatomical_struct/labelsTr`: Processed label data (ground truth tumor segmentation)
# 
# Required Inputs:
# - Tumor simulation parameters (in a JSON format)
# - Patient Dataset (tumor & atlas images)
#
# Outputs:
# - Cropped and downsampled images stored in the appropriate nnU-Net directories
# - A `dataset.json` file that follows nnU-Net dataset conventions.
#
#Resulting directory structure:
#mount_dir/data_and_outputs/
#├── raw_data/
#│   ├── Dataset700_Brain/
#│   │   ├── imagesTr/
#│   │   │   └── Brain_p001_0000.nii.gz
#│   │   ├── labelsTr/
#│   │   │   └── Brain_p001.nii.gz
#├── preprocessed_data/
#│   ├── param_dict.pth
#│   └── dataset.json
#├── results/
#└── dataset.json

# =============================================

id = 700  # Dataset ID (must be of form XXX)
anatomical_struct = "Brain"  # Anatomical structure (e.g., Brain)
mount_dir = "/mnt/Drive3/jonas_zeineb/data_and_outputs"  
crop_sz = 120  # Crop size for the images
downsample_sz = 64  # Downsample size for the images
start, stop = 0, 9  # Range of patients to process (e.g., from patient 0 to patient 9)

# Ensure required directories exist, else create them
os.makedirs(mount_dir, exist_ok=True)
sub_dirs = ['raw_data', 'preprocessed_data', 'results']
for sub_dir in sub_dirs:
    os.makedirs(os.path.join(mount_dir, sub_dir), exist_ok=True)

# Prepare the dataset by processing patients
param_dict = preparingDataset(id, mount_dir, anatomical_struct, start=start, stop=stop, crop_sz=crop_sz, downsample_sz=downsample_sz)

raw_dataset_path = os.path.join(mount_dir, 'raw_data', f'Dataset{id}_{anatomical_struct}')
create_json_file(param_dict, raw_dataset_path, comment="Dataset for tumor growth simulation")


"""
performing extra preprocessing steps (nnUnet)
Dataset Fingerprint Extraction, Experiment Planning, and Preprocessing Script.
Steps:
1. Define necessary paths.
2. Extract dataset fingerprint. 
3. Plan experiment.
4. Preprocess the dataset.
"""

from TumorNetSolvers.utils.paths import nnUNet_preprocessed, nnUNet_raw, nnUNet_results
from TumorNetSolvers.reg_nnUNet.utilities.dataset_name_id_conversion import find_candidate_datasets, maybe_convert_to_dataset_name
from TumorNetSolvers.reg_nnUNet.experiment_planning.dataset_fingerprint.fingerprint_extractor import DatasetFingerprintExtractor
from TumorNetSolvers.reg_nnUNet.experiment_planning.plan_and_preprocess_api import plan_experiment_dataset, preprocess_dataset, preprocess_dataset2

#%% Step 1: Verify correct paths are set
# You can print or log the paths for debugging purposes
print("Paths for nnUNet directories:")
print(f"Preprocessed data: {nnUNet_preprocessed}")
print(f"Raw data: {nnUNet_raw}")
print(f"Results data: {nnUNet_results}")

#%% Step 2: Dataset Fingerprint Extraction
# Assuming dataset ID is defined (example ID: 600)
dataset_id = 600

# Find candidate datasets and convert to dataset name
print("Finding candidate datasets...")
find_candidate_datasets(dataset_id)
dataset_name = maybe_convert_to_dataset_name(dataset_id)

# Extract dataset fingerprint
print(f"Extracting dataset fingerprint for {dataset_name}...")
fpe = DatasetFingerprintExtractor(dataset_name, 8, verbose=True)
fingerprint = fpe.run()
print("Fingerprint extraction complete.")

#%% Step 3: Experiment Planning
print("Planning experiment for dataset...")
plan, plan_identifier = plan_experiment_dataset(dataset_id)
print(f"Experiment plan identifier: {plan_identifier}")

#%% Step 4: Dataset Preprocessing
print(f"Preprocessing dataset {dataset_name}...")
# Preprocessing (use preprocess_dataset2 for generating masks too (only for comparing to baseline), else use preprocess_dataset )
preprocess_dataset2(dataset_id, num_processes=(8, 4, 8))

print("Preprocessing complete.")

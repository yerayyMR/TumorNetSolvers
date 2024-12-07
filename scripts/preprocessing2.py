# =============================================
# Overview:
# This script prepares and preprocesses the dataset for use with the  framework.
# Steps:
# 1. Data Preparation: Process raw patient datasets and tumor simulation parameters. (original preprocessing pipeline +  saving in nnU-Net directory structure)
# e.g.
# mount_dir/data_and_outputs/
#   ├── raw_data/
#   │   ├── Dataset700_Brain/
#   │   │   ├── imagesTr/
#   │   │   │   └── Brain_p001_0000.nii.gz
#   │   │   ├── labelsTr/
#   │   │   │   └── Brain_p001.nii.gz
#   ├── preprocessed_data/
#   │   ├── param_dict.pth
#   │   └── dataset.json
#   ├── results/
#   └── dataset.json
#
# 2. Dataset Fingerprint Extraction: Analyze dataset properties for experiment planning. (nnUnet preprocessing)
# 3. Experiment Planning: Create an experiment configuration for the dataset. (nnUnet preprocessing)
# 4. Dataset Preprocessing: additional preprocessing steps. (nnUnet preprocessing)
#
# Outputs:
# - Preprocessed data stored in the right directory structure
#   ready for model training and validation.
# =============================================

#%%

import os
import torch
from set_env import set_environment_variables
set_environment_variables()
nnUNet_preprocessed = os.getenv('nnUNet_preprocessed')
nnUNet_raw = os.getenv('nnUNet_raw')
nnUNet_results = os.getenv('nnUNet_results')
print(nnUNet_preprocessed,nnUNet_raw,nnUNet_results)

from TumorNetSolvers.preprocessing.data_preprocessor import preparingDataset, create_json_file

from TumorNetSolvers.reg_nnUNet.utilities.dataset_name_id_conversion import (
    find_candidate_datasets, maybe_convert_to_dataset_name
)
from TumorNetSolvers.reg_nnUNet.experiment_planning.dataset_fingerprint.fingerprint_extractor import DatasetFingerprintExtractor
from TumorNetSolvers.reg_nnUNet.experiment_planning.plan_and_preprocess_api import (
    plan_experiment_dataset, preprocess_dataset
)



# ============ Data Preparation ============
# Configuration
DATASET_ID = 500  # Must be in XXX format
ANATOMICAL_STRUCTURE = "Brain"
MOUNT_DIR = "/mnt/Drive3/jonas_zeineb/data_and_outputs"
CROP_SIZE = 120  # Image crop size
DOWNSAMPLE_SIZE = 64  # Downsample size
PATIENT_RANGE = (0, 3)  # Patients to process (start, stop)

# Ensure required directories exist
for sub_dir in ['raw_data', 'preprocessed_data', 'results']:
    print(os.path.join(MOUNT_DIR, sub_dir))
    os.makedirs(os.path.join(MOUNT_DIR, sub_dir), exist_ok=True)
    os.makedirs(os.path.join(MOUNT_DIR, sub_dir,f'Dataset{DATASET_ID}_{ANATOMICAL_STRUCTURE}'), exist_ok=True)
    print(os.path.join(MOUNT_DIR, sub_dir,f'Dataset{DATASET_ID}_{ANATOMICAL_STRUCTURE}'))
#%%
# Process patient datasets
print("Preparing dataset...")
param_dict = preparingDataset(
    DATASET_ID, MOUNT_DIR, ANATOMICAL_STRUCTURE,
    start=PATIENT_RANGE[0], stop=PATIENT_RANGE[1],
    crop_sz=CROP_SIZE, downsample_sz=DOWNSAMPLE_SIZE
)


#%%
PP_DATASET_PATH=os.path.join(nnUNet_preprocessed, f'Dataset{DATASET_ID}_{ANATOMICAL_STRUCTURE}')
pth=os.path.join(PP_DATASET_PATH, 'param_dict.pth')
torch.save(param_dict, pth)
# Create `dataset.json` for nnU-Net
RAW_DATASET_PATH = os.path.join(MOUNT_DIR, 'raw_data', f'Dataset{DATASET_ID}_{ANATOMICAL_STRUCTURE}')
create_json_file(param_dict, RAW_DATASET_PATH, comment="Dataset for tumor growth simulation")
print("Dataset preparation complete.")
#%%
# ============ Fingerprint Extraction ============


print("Extracting dataset fingerprint...")
print(nnUNet_preprocessed)
candidate_datasets = find_candidate_datasets(DATASET_ID)
dataset_name = maybe_convert_to_dataset_name(DATASET_ID)

fpe = DatasetFingerprintExtractor(dataset_name, num_processes=8, verbose=False)
fingerprint = fpe.run()
print("Fingerprint extraction complete.")

# ============ Experiment Planning ============

print("Planning experiment...")
plan, plan_identifier = plan_experiment_dataset(DATASET_ID)
print(f"Experiment planned with identifier: {plan_identifier}")

# ============ Dataset Preprocessing ============

print("Preprocessing dataset...")
#USE generate_masks=True if binary masks (GM-WM) are also needed (comparison with baseline),
#else use preprocess_dataset 
preprocess_dataset(DATASET_ID, num_processes=(8, 4, 8), generate_masks=False)
print("Dataset preprocessing complete.")

# ============ Summary ============

print(f"""
Process completed. Outputs are stored in:
- Raw data: {nnUNet_raw}
- Preprocessed data: {nnUNet_preprocessed}
- Results: {nnUNet_results}
""")

# %%

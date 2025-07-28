"""
Requirements:
- Dataset Name and corresponding `dataset.json` file.
- Training configuration ('2D' or '3D') and fold setup.
- Preprocessing plan generated via the `plan_preprocess_commands` pipeline.
- Correct paths defined in the project environment.

Inputs:
- `dataset_name`: The name of the dataset to be trained on.

Outputs:
- Trained model and associated files stored in the specified project directory.
"""
#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ['WANDB_DIR'] = '/home/home/yeray_jonas/tumornetsolvers/wandb'
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))  # scripts directory
src_path = os.path.abspath(os.path.join(current_dir, '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import torch
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
from TumorNetSolvers.training.updating_trainer import Trainer
from batchgenerators.utilities.file_and_folder_operations import load_json
from set_env import set_environment_variables
set_environment_variables()
nnUNet_preprocessed = os.getenv('nnUNet_preprocessed')

# ============ Configuration ============

# Define the dataset and training configuration
DATASET_NAME = 'Dataset900_Brain'  
TRAINING_CONFIGURATION = '3d_fullres'  # '2d', '3d_lowres', '3d_fullres', etc.
DEVICE = torch.device('cuda:0')

# Define project and training parameters
PROJECT_NAME = "NN-based-tumor-solvers"  # for wandb
MODEL_NAME = "ViT"  # other options are 'nnUnet', 'TumorSurrogate' and 'ViT'
SIGNATURE = "10k"  # Unique signature for logging and reproducibility

# Define experiments regarding insertion of parameters (mode and location)
'''EXPERIMENTS = [
    ['MLP', 'one_token'],
    ['Linear', 'one_token'],
    ['MLP', 'mul_token'],
    ['MLP', 'embed_concat'],
    ['Linear', 'embed_concat'],
    ['MLP', 'embed_add'],
    ['Linear', 'embed_add']]'''
EXPERIMENTS = [['MLP', 'mul_token']]

#[['c', 'a_downsampling']]
#[['c', 'b_downsampling']]

# ============ Load Training Plans and Dataset ============

PLANS_FILE = os.path.join(nnUNet_preprocessed, DATASET_NAME, 'nnUNetPlans.json')
DATASET_JSON_FILE = os.path.join(nnUNet_preprocessed, DATASET_NAME, 'dataset.json')

# Load plans and dataset metadata
print(f"Loading plans from: {PLANS_FILE}")
plan = load_json(PLANS_FILE)

print(f"Loading dataset JSON from: {DATASET_JSON_FILE}")
dataset_json = load_json(DATASET_JSON_FILE)

# ============ Initialize and Run Training ============

for experiment in EXPERIMENTS:
    # Initialize the nnU-Net trainer
    print("Initializing trainer...")
    trainer = Trainer(
        plans=plan,
        configuration=TRAINING_CONFIGURATION,
        device=DEVICE,
        signature=SIGNATURE,
        fold='train_val_test',
        model=MODEL_NAME,
        dataset_json=dataset_json,
        project_name=PROJECT_NAME,
        experiments = experiment,
        seed=12345
    )

    # Run the training process
    print("Starting training...")
    trainer.run_training()
    if MODEL_NAME == "ViT":
        print("Training complete." + f"[{MODEL_NAME}] Mode: {experiment[1]}, Method: {experiment[0]}")
    else:
        print("Training complete." + f"[{MODEL_NAME}] Loc: {experiment[1]}, Mode: {experiment[0]}")

# %%

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
os.environ["CUDA_VISIBLE_DEVICES"]="0" # Server specific, adjust accordingly
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ['WANDB_DIR'] = '/home/home/yeray_jonas/tumornetsolvers/wandb' # Adjust to the desired path where the wandb local files are saved (such files can be deleted afterwards)
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
set_environment_variables() # Modify path in this function accordingly
nnUNet_preprocessed = os.getenv('nnUNet_preprocessed')

# ============ Configuration ============

# WARNING: FOLD must be defined as a number on the first time the training is run for each set of preprocessed data
# (this will simply create the split file, then set to one of the string options)
FOLD = 'train_val_test'                 # The fold represents the splitting of data (str or int): 'train_val_test' (train and validation or only training to be implemented)
NUM_FOLDS = None                           # The number of folds for the first time the split is done (int or None): (total data is divided into an X amount of folds. One belongs to validation, another to test and the rest to training)

if isinstance(FOLD,int) and not isinstance(NUM_FOLDS,int):
    raise ValueError("A number of folds is required.")
elif isinstance(FOLD,int) and NUM_FOLDS < 3:
    raise ValueError("The number of folds must at least be three for division into train, val and test.")
    

# Define the dataset and training configuration
DATASET_NAME = 'Dataset900_Brain'  
TRAINING_CONFIGURATION = '3d_fullres'   # '2d', '3d_lowres', '3d_fullres', etc.
DEVICE = torch.device('cuda:0')
NUM_EPOCHS = 1000                       # max. number of epochs (int)
BATCH_CUSTOM = None                     # custom batch size for experiments (int or None) -- If None, default used according to GPU
ENDING = 'trial'                           # Specific ending to the naming of the folder where weights and logs will be saved (str or None) -- If None default naming based one experiment will be used
MAX_TRAIN = 72                          # Hours that the training is allowed to run for (int / float or None) -- If None time from wall clock will not enforce it to stop
LOAD_PATH = None                        # Path to load previous training if needed (str or None) -- If None not imported -- Filename MUST contain the epoch number that it was saved on
EVERY_HOURS = 6                         # Every how many hours an epoch shall be saved for comparison (int / float or None)
PATIENCE = 10                           # Num. of epochs for patience regariding early stopping (int or None) -- If None early stopping is not considered
COUNTER_EPOCH_SAVE = None               # Every how many epochs shall be saved for comparison (int) -- If None not used to be saved -- Currently as soon as new counter reached, previous is deleted
COUNTER_EPOCH_EMA = None                # Every how many epochs that the ema loss has improved (MSE average) shall an epoch be saved for comparison (int) -- If None not used to be saved --  Currently as soon as new counter reached, previous is deleted

# Define project and training parameters
PROJECT_NAME = "NN-based-tumor-solvers"  # for wandb
MODEL_NAME = "TumorSurrogate"  # other options are 'nnUnet', 'TumorSurrogate' and 'ViT'
SIGNATURE = "10k"  # Unique signature for logging and reproducibility

# Define experiments regarding insertion of parameters (mode and location) -- At least one must be defined and always in double list format [[], []]
'''MODEL_EXPERIMENTS = {

    'nnUnet': [
        ['c', 'a_downsampling'], ['a', 'a_downsampling'],
        ['c', 'b_downsampling'], ['a', 'b_downsampling'],
        ['c', 'inputs'], ['a', 'inputs'],
        ['c', 'b_bottleneck'], ['a', 'b_bottleneck'],
        ['c', 'a_bottleneck'], ['a', 'a_bottleneck'],
        ['c', 'a_upsampling'], ['a', 'a_upsampling'],
        ['c', 'b_upsampling'], ['a', 'b_upsampling'],
        ['a', 'a_upsampling_skip']
    ],
    'TumorSurrogate': [
        ['c', 'a_downsampling'], ['a', 'a_downsampling'],
        ['c', 'b_downsampling'], ['a', 'b_downsampling'],
        ['c', 'inputs'], ['a', 'inputs'],
        ['c', 'b_bottleneck'], ['a', 'b_bottleneck'],
        ['c', 'a_bottleneck'], ['a', 'a_bottleneck'],
        ['c', 'a_bottleneck_after'], ['a', 'a_bottleneck_after'], -- Modified version according to the paper
        ['c', 'a_upsampling_after'], ['a', 'a_upsampling_after'], -- Upsampling version according to the paper
    ],
    'ViT': [
        ['MLP', 'one_token'],
        ['Linear', 'one_token'],
        ['MLP', 'mul_token'],
        ['Linear', 'mul_token'],
        ['MLP', 'embed_concat'],
        ['Linear', 'embed_concat'],
        ['MLP', 'embed_add'],
        ['Linear', 'embed_add']
    ]
}'''


EXPERIMENTS = [['c', 'a_bottleneck_after']]


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
        fold=FOLD,
        num_folds=NUM_FOLDS,
        model=MODEL_NAME,
        dataset_json=dataset_json,
        project_name=PROJECT_NAME,
        experiment = experiment,
        seed=12345,
        num_epochs=NUM_EPOCHS,
        batch_custom=BATCH_CUSTOM,
        ending=ENDING,
        max_train=MAX_TRAIN,
        load_path=LOAD_PATH,
        every_hours=EVERY_HOURS,
        patience=PATIENCE,
        counter_epoch_save=COUNTER_EPOCH_SAVE,
        counter_epoch_ema=COUNTER_EPOCH_EMA
    )
    if isinstance(FOLD, int):
        print("Please initialize the fold as one of the provided strings.")
        break

    # Run the training process
    print("Starting training...")
    trainer.run_training()
    if MODEL_NAME == "ViT":
        print("Training complete." + f"[{MODEL_NAME}] Mode: {experiment[1]}, Method: {experiment[0]}")
    else:
        print("Training complete." + f"[{MODEL_NAME}] Loc: {experiment[1]}, Mode: {experiment[0]}")

# %%

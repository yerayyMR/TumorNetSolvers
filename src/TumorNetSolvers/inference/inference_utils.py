
import os
import sys
# For different experiments
current_dir = os.path.dirname(os.path.abspath(__file__))  # scripts directory
src_path = os.path.abspath(os.path.join(current_dir, '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from batchgenerators.utilities.file_and_folder_operations import load_json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from scripts.set_env import set_environment_variables
set_environment_variables()
# Access the environment variable for nnUNet_preprocessed
nnUNet_preprocessed = os.environ.get('nnUNet_preprocessed')
if nnUNet_preprocessed is None:
    raise EnvironmentError("nnUNet_preprocessed environment variable is not set.")


class CustomDataset(Dataset):
    """
    PyTorch Dataset for loading preprocessed `.npz` files for inference

    Each `.npz` file contains the following keys:
        - 'data': Input image data (required).
        - 'mask': Mask data (optional; if not available, returns None).
        - 'seg': GT simulation data (target) (required).

    Args:
        data_folder (str): Path to the folder containing the `.npz` files (preprocessed test samples)
        keys (list): List of keys (filenames without extensions) to load from `data_folder`.
    """

    def __init__(self, data_folder: str, keys: list):
        self.data_folder = data_folder
        self.keys = keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        data_path = os.path.join(self.data_folder, f"{key}.npz")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File {data_path} not found.")
        try:
            with np.load(data_path) as data:
                image = data['data']
                mask = data['mask'] if 'mask' in data else None
                target = data['seg']
        except Exception as e:
            raise ValueError(f"Error reading {data_path}: {e}")
        
        if mask is not None:
            return (torch.tensor(image, dtype=torch.float32),
                    torch.tensor(mask, dtype=torch.float32),
                    torch.tensor(target, dtype=torch.float32),
                    key)
        else:
            return (torch.tensor(image, dtype=torch.float32),
                    torch.tensor(target, dtype=torch.float32),
                    key)


def get_settings_and_file_paths(dataset_name: str):
    """
    Loads dataset configuration files and parameters for inference.

    Args:
        dataset_name (str): Name of the dataset (e.g., "Dataset700_Brain").

    Returns:
        tuple: (plan, dataset_json, test_keys, parameters)
            - plan (dict): nnUNet plan configuration loaded from `nnUNetPlans.json`.
            - dataset_json (dict): Dataset metadata loaded from `dataset.json`.
            - test_keys (list): List of test data keys from `splits_final.json`.
            - parameters (dict): Model parameters loaded from `param_dict.pth`.

    Raises:
        FileNotFoundError: If any of the required files are missing.
    """
    # Use the environment variable for nnUNet_preprocessed
    plans_file = os.path.join(nnUNet_preprocessed, dataset_name, 'nnUNetPlans.json')
    if not os.path.exists(plans_file):
        raise FileNotFoundError(f"Plan file {plans_file} not found.")
    plan = load_json(plans_file)

    dataset_json_file = os.path.join(nnUNet_preprocessed, dataset_name, 'dataset.json')
    if not os.path.exists(dataset_json_file):
        raise FileNotFoundError(f"Dataset JSON file {dataset_json_file} not found.")
    dataset_json = load_json(dataset_json_file)

    splits_file = os.path.join(nnUNet_preprocessed, dataset_name, 'splits_final.json')
    if not os.path.exists(splits_file):
        raise FileNotFoundError(f"Splits file {splits_file} not found.")
    splits_f = load_json(splits_file)
    test_keys = splits_f[0]["test"]

    param_file = os.path.join(nnUNet_preprocessed, dataset_name, 'param_dict.pth')
    if not os.path.exists(param_file):
        raise FileNotFoundError(f"Parameter file {param_file} not found.")
    parameters = torch.load(param_file)

    return plan, dataset_json, test_keys, parameters

def construct_chkpt(nnUnet_results, model, dataset_name, experiment, ending):

    if model == "ViT":
        chkpt= os.path.join(nnUnet_results, f"{dataset_name}/Trainer__nnUNetPlans__3d_fullres/fold_train_val_test/_10k_{model}/MODE_{experiment[1]}_METHOD_{experiment[0]}{'_'+ending if ending is not None else ''}")
    else:
        chkpt=os.path.join(nnUnet_results, f"{dataset_name}/Trainer__nnUNetPlans__3d_fullres/fold_train_val_test/_10k_{model}/LOC_{experiment[1]}_MODE_{experiment[0]}{'_'+ending if ending is not None else ''}")

    # Pattern to match checkpoint files like: checkpoint_MODEL_<number>_best_ema_loss.pth
    pattern = re.compile(rf"^checkpoint_{model}_(\d+)_best_ema_loss\.pth$")

    # Scan directory for matching files and extract numbers
    best_checkpoint = None
    max_number = -1

    for filename in os.listdir(chkpt):
        match = pattern.match(filename)
        if match:
            number = int(match.group(1))
            if number > max_number:
                max_number = number
                best_checkpoint = filename

    if best_checkpoint is None:
        raise FileNotFoundError(f"No matching checkpoint file found in {chkpt} for MODEL={model}")
    
    chkpt = os.path.join(chkpt, best_checkpoint)
    return chkpt

def construct_output_base(nnUNet_results, model, dataset_name, coefficients, experiment, ending, signature):

    if coefficients != None:
        if model == "ViT":
            if all(isinstance(item, list) and len(item) == 2 for item in coefficients): # Only rho and D
                output_base = os.path.join(nnUNet_results, dataset_name, f"{model}_{signature}", f"MODE_{experiment[1]}_METHOD_{experiment[0]}{'_' + ending if ending is not None else ''}_rho_{coefficients[0]}_D_{coefficients[1]}", 'preds')
            else:
                output_base = os.path.join(nnUNet_results, dataset_name, f"{model}_{signature}", f"MODE_{experiment[1]}_METHOD_{experiment[0]}{'_'+ending if ending is not None else ''}_coeffs_{coefficients[0]}", 'preds')
        else:
            if all(isinstance(item, list) and len(item) == 2 for item in coefficients): # Only rho and D
                output_base = os.path.join(nnUNet_results, dataset_name, f"{model}_{signature}", f"LOC_{experiment[1]}_MODE_{experiment[0]}{'_'+ending if ending is not None else ''}_rho_{coefficients[0]}_D_{coefficients[1]}", 'preds')
            else:
                output_base = os.path.join(nnUNet_results, dataset_name, f"{model}_{signature}", f"LOC_{experiment[1]}_MODE_{experiment[0]}{'_'+ending if ending is not None else ''}_coeffs_{coefficients[0]}", 'preds')
    else:
        if model == "ViT":
            output_base = os.path.join(nnUNet_results, dataset_name, f"{model}_{signature}", f"MODE_{experiment[1]}_METHOD_{experiment[0]}{'_'+ending if ending is not None else ''}", 'preds')
        else:
            output_base = os.path.join(nnUNet_results, dataset_name, f"{model}_{signature}", f"LOC_{experiment[1]}_MODE_{experiment[0]}{'_'+ending if ending is not None else ''}", 'preds')

    return output_base
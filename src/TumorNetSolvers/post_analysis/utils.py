import os
import sys
# For different experiments
current_dir = os.path.dirname(os.path.abspath(__file__))  # scripts directory
src_path = os.path.abspath(os.path.join(current_dir, '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)
#nnUNet_preprocessed = os.environ.get('nnUNet_preprocessed')
from TumorNetSolvers.inference.inference_utils import CustomDataset, get_settings_and_file_paths
from torch.utils.data import DataLoader
import numpy as np
import nibabel as nib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from scripts.set_env import set_environment_variables
set_environment_variables()
nnUNet_preprocessed = os.environ.get('nnUNet_preprocessed')

import torch
import json

def save_full_ground_truths(dataset_name: str, output_base: str):

    """
    Iterate through test set and save full 3D ground truth tumor masks as NIfTI files.
    """
    data_folder = os.path.join(nnUNet_preprocessed, dataset_name,"nnUNetPlans_3d_fullres")
    plan, dataset_json, test_keys, parameters = get_settings_and_file_paths(dataset_name)
    dataset = CustomDataset(data_folder, test_keys)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for batch in data_loader:
        if len(batch) == 4:
            _, _, target, keys = batch
        elif len(batch) == 3:
            _, target, keys = batch
        else:
            raise ValueError("Unexpected batch structure")

        patient_id = keys[0]
        filter= lambda x : (x>0)*x
        target = filter(target)
        target_np = target[0, 0].cpu().numpy()  # shape: (H, W, D)

        # Prepare output path
        save_path = os.path.join(output_base, patient_id)
        os.makedirs(save_path, exist_ok=True)

        # Save as full NIfTI
        nib.save(nib.Nifti1Image(target_np.astype(np.float32), affine=np.eye(4)), os.path.join(save_path, "ground_truth_full.nii.gz"))

def num_params(pth_path):

    def count_params(obj):
        if isinstance(obj, torch.Tensor):
            return obj.numel()
        elif isinstance(obj, dict):
            return sum(count_params(v) for v in obj.values())
        else:
            return 0  # ignore non-tensor, non-dict entries

    checkpoint = torch.load(pth_path, map_location='cpu')

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    total_params = count_params(state_dict)
    print(f"Total parameters: {total_params}")

def calc_audc(model_experiments, dataset, ending, signature):

    base_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'performance_summaries')

    def compute_audc_from_json(json_path):
        """
        Computes the Area Under the Dice Curve (AUDC) and standard deviation of Dice
        scores from a JSON file with sample_metrics containing Dice_x.y keys.

        Parameters:
            json_path (str): Path to the JSON file.

        Returns:
            audc (float): Area under the mean Dice curve.
            audc_std (float): Area under the std dev Dice curve.
            thresholds (list of float): Sorted thresholds.
            mean_dice_scores (list of float): Mean Dice scores per threshold.
            std_dice_scores (list of float): Std dev of Dice scores per threshold.
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        if "sample_metrics" not in data:
            raise ValueError("JSON missing 'sample_metrics' key.")

        sample_metrics = data["sample_metrics"]
        if not sample_metrics:
            raise ValueError("'sample_metrics' is empty.")

        # Collect all Dice keys from the first sample (assuming all samples have same keys)
        first_sample_key = next(iter(sample_metrics))
        dice_keys = sorted([k for k in sample_metrics[first_sample_key].keys() if k.startswith("Dice_")],
                        key=lambda x: float(x.split("_")[1]))

        if len(dice_keys) < 2:
            raise ValueError("Not enough Dice threshold points to compute AUDC.")

        thresholds = [float(k.split("_")[1]) for k in dice_keys]

        # Gather dice values for each threshold across all samples
        all_scores = {k: [] for k in dice_keys}
        for sample_id, metrics in sample_metrics.items():
            for key in dice_keys:
                if key in metrics:
                    all_scores[key].append(metrics[key])
                else:
                    raise ValueError(f"Sample '{sample_id}' missing key '{key}'.")

        # Compute mean and std per threshold
        mean_dice_scores = [np.mean(all_scores[k]) for k in dice_keys]
        std_dice_scores = [np.std(all_scores[k]) for k in dice_keys]

        # Compute AUDC as integral over the mean Dice curve
        audc = np.trapz(mean_dice_scores, thresholds)
        # Compute AUDC std as integral over the std Dice curve
        audc_std = np.trapz(std_dice_scores, thresholds)

        return audc, audc_std, thresholds, mean_dice_scores, std_dice_scores


    # Main logic
    all_audc_results = {}

    for model, experiments in model_experiments.items():
        model_results = {}

        for exp in experiments:
            # Build experiment folder name
            if model == "ViT":
                exp_folder = f"MODE_{exp[1]}_METHOD_{exp[0]}{'_'+ending if ending is not None else ''}"
            else:
                exp_folder = f"LOC_{exp[1]}_MODE_{exp[0]}{'_'+ending if ending is not None else ''}"

            # Build path to JSON file
            json_folder = os.path.join(base_path, dataset, f"{model}_{signature}", exp_folder)

            json_file = f"evaluation_results_{model}_10k.json"
            json_path = os.path.join(json_folder, json_file)

            if not os.path.exists(json_path):
                print(f"[Missing] {json_path}")
                continue

            try:
                audc, audc_std, thresholds, mean_dice, std_dice = compute_audc_from_json(json_path)
                model_results[exp_folder] = {
                    "AUDC": audc,
                    "AUDC_std": audc_std,
                    "thresholds": thresholds,
                    "mean_dice": mean_dice,
                    "std_dice": std_dice
                }
                print(f"[OK] {model}/{exp_folder} -> AUDC: {audc:.4f} ± {audc_std:.4f}")
            except Exception as e:
                print(f"[Error] {model}/{exp_folder}: {e}")

                all_audc_results[model] = model_results


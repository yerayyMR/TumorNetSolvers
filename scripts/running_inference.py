"""
This script performs inference using specified models on a given dataset.
Requirements:
- Preprocessed dataset directory with corresponding configuration files.
- Model names and parameters for inference.
- Correctly defined paths for input data and output results.

Inputs:
- `dataset_name`: Name of the dataset for inference.
- `models`: List of models to use for inference (e.g., ViT, nnUnet, etc.), can also be used for one model eg ['ViT'].

Outputs:
- Predictions saved in specified output directory/-ies.
"""
#%%
import os
from set_env import set_environment_variables
set_environment_variables()

from TumorNetSolvers.inference.inference_utils import get_settings_and_file_paths
from TumorNetSolvers.inference.run_inference_NEWEST import run_inference
import torch

def main():
    nnUNet_preprocessed = os.getenv('nnUNet_preprocessed')
    nnUNet_results= os.getenv('nnUNet_results')
    # ============ Configuration ============
    DATASET_NAME = 'Dataset500_Brain'  # Specify the dataset name
    MODELS = ['nnUnet']  # Models to use for inference e.g. ['ViT', 'nnUnet', 'TumorSurrogate']
    DEVICE = torch.device('cuda:0')  

    # Paths
    DATA_FOLDER = os.path.join(nnUNet_preprocessed, DATASET_NAME,"nnUNetPlans_3d_fullres")
    OUTPUT_BASE = os.path.join(nnUNet_results, DATASET_NAME, 'preds')
    SIGNATURE='10k'

    # ============ Run Inference ============
    print("Running inference...")
    run_inference(
        dataset_name=DATASET_NAME,
        models=MODELS,
        data_folder=DATA_FOLDER,
        output_base=OUTPUT_BASE,
        device=DEVICE,
        signature=SIGNATURE
    )
    print("Inference complete. Results saved to output directory.")

if __name__ == "__main__":
    main()

# %%

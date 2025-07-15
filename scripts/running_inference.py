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
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import sys
from set_env import set_environment_variables
set_environment_variables()

current_dir = os.path.dirname(os.path.abspath(__file__))  # scripts directory
src_path = os.path.abspath(os.path.join(current_dir, '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import torch
from TumorNetSolvers.inference.inference_utils import get_settings_and_file_paths
from TumorNetSolvers.inference.run_inference_NEWEST import run_inference


def main():
    nnUNet_preprocessed = os.getenv('nnUNet_preprocessed')
    nnUNet_results= os.getenv('nnUNet_results')
    # ============ Configuration ============
    DATASET_NAME = 'Dataset900_Brain'  # Specify the dataset name
    MODELS = ['nnUnet']  # Models to use for inference e.g. ['ViT', 'nnUnet', 'TumorSurrogate']
    DEVICE = torch.device('cuda:0')  

    EXPERIMENTS = [ ['a', 'b_upsampling'], ['c', 'b_upsampling'],
                    ['a', 'a_upsampling'], ['c', 'a_upsampling'],
                    ['a', 'a_upsampling_skip']]
    # ['a', 'b_bottleneck'], ['a', 'a_bottleneck'], ['c', 'a_bottleneck'] 
    # Paths
    for experiment in EXPERIMENTS:
        DATA_FOLDER = os.path.join(nnUNet_preprocessed, DATASET_NAME,"nnUNetPlans_3d_fullres")
        if MODELS[0] == "ViT":
            OUTPUT_BASE = os.path.join(nnUNet_results, DATASET_NAME, f'MODE_{experiment[1]}_METHOD_{experiment[0]}_og', 'preds')
        else:
            OUTPUT_BASE = os.path.join(nnUNet_results, DATASET_NAME, f'LOC_{experiment[1]}_MODE_{experiment[0]}', 'preds')
        SIGNATURE='10k'

        # ============ Run Inference ============
        print("Running inference...")
        run_inference(
            dataset_name=DATASET_NAME,
            models=MODELS,
            data_folder=DATA_FOLDER,
            output_base=OUTPUT_BASE,
            device=DEVICE,
            signature=SIGNATURE,
            experiment = experiment,
            chkpt=f"/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset900_Brain/Trainer__nnUNetPlans__3d_fullres/fold_train_val_test/_10k_{MODELS[0]}/LOC_{experiment[1]}_MODE_{experiment[0]}/checkpoint_{MODELS[0]}_best_ema_loss.pth"
        )
        print("Inference complete. Results saved to output directory.")

if __name__ == "__main__":
    main()

# %%

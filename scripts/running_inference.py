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
os.environ["CUDA_VISIBLE_DEVICES"]="0"
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
    #nnUNet_results= os.getenv('nnUNet_results')
    nnUNet_results = os.path.join('/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/', "results")#os.environ.get('nnUNet_results')
    # ============ Configuration ============
    DATASET_NAME = 'Dataset900_Brain'  # Specify the dataset name
    MODELS = ['nnUnet']  # Models to use for inference e.g. ['ViT', 'nnUnet', 'TumorSurrogate']
    DEVICE = torch.device('cuda:0')  

    #EXPERIMENTS = [['Linear', 'embed_concat']]
    #EXPERIMENTS = [['c', 'a_bottleneck_after']]
    EXPERIMENTS = [['c', 'a_upsampling']]
    '''EXPERIMENTS = [ ['c', 'a_downsampling'], ['a', 'a_downsampling'],
                    ['c', 'b_downsampling'], ['a', 'b_downsampling'],
                    ['c', 'inputs'], ['a', 'inputs'],
                    ['c', 'b_bottleneck'], ['a', 'b_bottleneck'],
                    ['c', 'a_bottleneck'], ['a', 'a_bottleneck']]'''
    # ['a', 'b_bottleneck'], ['a', 'a_bottleneck'], ['c', 'a_bottleneck']
    '''COEFFICIENTS = [
        [0, 0], [0, 0.25], [0, 0.5], [0, 0.75], [0, 1],
        [0.25, 0], [0.25, 0.25], [0.25, 0.5], [0.25, 0.75], [0.25, 1],
        [0.5, 0], [0.5, 0.25], [0.5, 0.5], [0.5, 0.75], [0.5, 1],
        [0.75, 0], [0.75, 0.25], [0.75, 0.5], [0.75, 0.75], [0.75, 1],
        [1, 0], [1, 0.25], [1, 0.5], [1, 0.75], [1, 1]
    ]'''
    COEFFICIENTS = [[0], [0.2], [0.4], [0.6], [0.8], [1]]
    # Paths
    for experiment in EXPERIMENTS:
        for coefficient in COEFFICIENTS:
            DATA_FOLDER = os.path.join(nnUNet_preprocessed, DATASET_NAME,"nnUNetPlans_3d_fullres")
            if MODELS[0] == "ViT":
                OUTPUT_BASE = os.path.join(nnUNet_results, DATASET_NAME, MODELS[0], f'MODE_{experiment[1]}_METHOD_{experiment[0]}_check', 'preds')
                chkpt=f"/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset900_Brain/Trainer__nnUNetPlans__3d_fullres/fold_train_val_test/_10k_{MODELS[0]}/MODE_{experiment[1]}_METHOD_{experiment[0]}_check/checkpoint_{MODELS[0]}_306_best_ema_loss.pth"
            elif MODELS[0] == "TumorSurrogate":
                OUTPUT_BASE = os.path.join(nnUNet_results, DATASET_NAME, MODELS[0], 'correctResNet', f'LOC_{experiment[1]}_MODE_{experiment[0]}_check', 'preds')
                chkpt=f"/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset900_Brain/Trainer__nnUNetPlans__3d_fullres/fold_train_val_test/_10k_{MODELS[0]}/correctResNet/LOC_{experiment[1]}_MODE_{experiment[0]}_best_FK_50_4/best_cases/checkpoint_{MODELS[0]}_368_best_ema_loss.pth"
            else:
                OUTPUT_BASE = os.path.join(nnUNet_results, DATASET_NAME, MODELS[0], f'LOC_{experiment[1]}_MODE_{experiment[0]}_best_FK_50_{coefficient[0]}', 'preds')
                chkpt=f"/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset900_Brain/Trainer__nnUNetPlans__3d_fullres/fold_train_val_test/_10k_{MODELS[0]}/LOC_{experiment[1]}_MODE_{experiment[0]}_best_FK_50_2/best_cases/checkpoint_{MODELS[0]}_862_best_ema_loss.pth"
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
                chkpt=chkpt,
                D=coefficient[0],
                rho=coefficient[0],
                all=coefficient[0]
            )
            print("Inference complete. Results saved to output directory.")

if __name__ == "__main__":
    main()

# %%

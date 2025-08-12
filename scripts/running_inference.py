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
os.environ["CUDA_VISIBLE_DEVICES"]="0" # Server specific, adjust accordingly
import sys
from set_env import set_environment_variables
set_environment_variables()

current_dir = os.path.dirname(os.path.abspath(__file__))  # scripts directory
src_path = os.path.abspath(os.path.join(current_dir, '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import torch
#from TumorNetSolvers.inference.inference_utils import get_settings_and_file_paths
from TumorNetSolvers.inference.run_inference_NEWEST import run_inference


def main():
    nnUNet_preprocessed = os.getenv('nnUNet_preprocessed')
    # ============ Configuration ============
    DATASET_NAME = 'Dataset900_Brain'       # Specify the dataset name
    MODEL = 'nnUnet'                        # Models to use for inference: 'ViT', 'nnUnet', 'TumorSurrogate'
    DEVICE = torch.device('cuda:0')  

    ENDING = "trial"                           # Specific ending to the naming of the folder where weights and logs will be saved (str or None) -- If None default naming based one experiment will be used
    CHECKPOINT = None                       # Directory to file containing the weights of the model to be used for inference (str or None) -- If None it will use the last epoch of the best_ema_loss in the corresponding default directory
    OUTPUT_BASE = None                      # Directory or list of directories to where the inference values are saved (str or list(str) or None) -- If None it will be set to default, check: src/TumorNetSolvers/inference/inference_utils.py

    DATA_FOLDER = os.path.join(nnUNet_preprocessed, DATASET_NAME,"nnUNetPlans_3d_fullres")
    SIGNATURE='10k'

    BATCH_SIZE = 25                         # Set desired batch size for inference

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


    EXPERIMENTS = [['c', 'a_upsampling']]

    # Define coefficients -- Same format as EXPERIMENTS:
    # - If two values in inner list, they correspond uniquely to rho and D correspondingly [[rho, D]]
    # - If only one value in the inner list, it affects to the 5 parameters [[all_params]]
    #
    # Only values between 0 and 1 (included) are allowed. The value is percentage/100.
    # Therefore, 0.2 makes the coefficient(s) to be 20% its original value. (e.g., original D = 10, applied a 0.2, the new D is D = 10 * 0.2 = 2)
    #
    # For each list those coefficients will be applied for each experiment previously defined.
    #
    # If COEFFICIENTS = None --> Original values are used (same as defining them to be = 1)
    #
    # Naming of files adjusted automatically, modify below if desired
    '''COEFFICIENTS = [
        [0, 0], [0, 0.25], [0, 0.5], [0, 0.75], [0, 1],
        [0.25, 0], [0.25, 0.25], [0.25, 0.5], [0.25, 0.75], [0.25, 1],
        [0.5, 0], [0.5, 0.25], [0.5, 0.5], [0.5, 0.75], [0.5, 1],
        [0.75, 0], [0.75, 0.25], [0.75, 0.5], [0.75, 0.75], [0.75, 1],
        [1, 0], [1, 0.25], [1, 0.5], [1, 0.75], [1, 1]
    ]
    COEFFICIENTS = [[0], [0.2], [0.4], [0.6], [0.8], [1]]'''

    COEFFICIENTS = [
        [0, 0], [0, 0.25], [0, 0.5], [0, 0.75], [0, 1],
        [0.25, 0], [0.25, 0.25], [0.25, 0.5], [0.25, 0.75], [0.25, 1],
        [0.5, 0], [0.5, 0.25], [0.5, 0.5], [0.5, 0.75], [0.5, 1],
        [0.75, 0], [0.75, 0.25], [0.75, 0.5], [0.75, 0.75], [0.75, 1],
        [1, 0], [1, 0.25], [1, 0.5], [1, 0.75], [1, 1]
    ]

    # Check number of coefficients corresponds to possibly definition of several output bases
    if OUTPUT_BASE != None and COEFFICIENTS != None:
        assert len(COEFFICIENTS) == len(OUTPUT_BASE), "Mismatch: COEFFICIENTS and OUTPUT_BASE must be of the same length"
    # Check OUTPUT_BASE follows the definition
    if COEFFICIENTS is not None:
        if OUTPUT_BASE is not None and not isinstance(OUTPUT_BASE, str):
            raise TypeError("OUTPUT_BASE must be a string or None when COEFFICIENTS is provided")


    # Paths
    for experiment in EXPERIMENTS:
        if COEFFICIENTS != None:
            for idx, coefficient in enumerate(COEFFICIENTS):
                # Different runs are done since each coefficient set will be saved in a different directory by default
                # ============ Run Inference ============
                print("Running inference...")
                run_inference(
                    dataset_name=DATASET_NAME,
                    model=MODEL,
                    data_folder=DATA_FOLDER,
                    output_base = OUTPUT_BASE[idx] if isinstance(OUTPUT_BASE, list) else OUTPUT_BASE,
                    device=DEVICE,
                    signature=SIGNATURE,
                    ending=ENDING,
                    experiment = experiment,
                    chkpt=CHECKPOINT,
                    coefficients=coefficient,
                    batch_size=BATCH_SIZE
                )
                print("Inference complete. Results saved to output directory.")
        else:

            # ============ Run Inference ============
            print("Running inference...")
            run_inference(
                    dataset_name=DATASET_NAME,
                    model=MODEL,
                    data_folder=DATA_FOLDER,
                    output_base=OUTPUT_BASE,
                    device=DEVICE,
                    signature=SIGNATURE,
                    ending=ENDING,
                    experiment = experiment,
                    chkpt=CHECKPOINT,
                    coefficients=COEFFICIENTS,
                    batch_size=BATCH_SIZE
                )
            print("Inference complete. Results saved to output directory.")
if __name__ == "__main__":
    main()

# %%

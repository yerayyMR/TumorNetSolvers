"""
This script performs performance evaluation and statistical analysis of prediction results.
Requirements:
- Predictions folder containing model output files.
- Ground truth folder with corresponding target files.
- JSON file to save evaluation results.
- Utility scripts for metrics computation and statistics analysis.

Steps:
1. Return eval metrics per sample and save results to a JSON file.
2. Analyze statistical properties and extreme values of evaluation metrics.

Inputs:
- `preds_folder`: Path to the directory containing model predictions.
- `gt_folder`: Path to the directory containing ground truth labels.
- `json_file`: JSON file containing evaluation results (for statistical analysis).

Outputs:
- `evaluation_results.json`: performance metrics.
- `output_summary.json`: statistical summary and extreme value analysis.
"""
# %% Performance Evaluation and Statistical Analysis
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))  # scripts directory
src_path = os.path.abspath(os.path.join(current_dir, '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from set_env import set_environment_variables
from TumorNetSolvers.evaluation.eval_preds_folder import compute_metrics
from TumorNetSolvers.evaluation.file_io import save_results_to_json
from TumorNetSolvers.evaluation.statistics import compute_statistics_with_extremes
set_environment_variables()

# Define environment variables
nnUNet_preprocessed = os.getenv('nnUNet_preprocessed')
nnUNet_results = os.getenv('nnUNet_results')
#nnUNet_results = os.path.join('/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/', "results")

# Configuration
DATASET_NAME = 'Dataset900_Brain' 
MODEL = 'ViT'  # Model for evaluation: 'ViT', 'nnUnet', 'TumorSurrogate'
SIGNATURE = '10k'
MASKED = False  # Binary flag for masked evaluation

ENDING = 'trial'
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


EXPERIMENTS = [['Linear', 'embed_concat']]


for experiment in EXPERIMENTS:
    # Define output directory for performance summaries
    if MODEL == "ViT":
        summary_dir = os.path.join("performance_summaries", DATASET_NAME , f"{MODEL}_{SIGNATURE}", f"MODE_{experiment[1]}_METHOD_{experiment[0]}{'_' + ENDING if ENDING is not None else ''}")
    else:
        summary_dir = os.path.join("performance_summaries", DATASET_NAME , f"{MODEL}_{SIGNATURE}", f"LOC_{experiment[1]}_MODE_{experiment[0]}{'_' + ENDING if ENDING is not None else ''}")
    os.makedirs(summary_dir, exist_ok=True)

    # Determine ground truth folder
    if MASKED:
        GT_FOLDER = os.path.join(nnUNet_results, DATASET_NAME, 'masked_gt')
        if not os.path.exists(GT_FOLDER) or len(os.listdir(GT_FOLDER)) == 0:
            raise FileNotFoundError(f"Masked ground truth folder is missing or empty: {GT_FOLDER}")
    else:
        GT_FOLDER = os.path.join(nnUNet_preprocessed, DATASET_NAME, 'nnUNetPlans_3d_fullres')

    # Determine predictions folder
    if MODEL == "ViT":
        if MASKED:
            PREDS_FOLDER = os.path.join(nnUNet_results, DATASET_NAME, f"{MODEL}_{SIGNATURE}", f"MODE_{experiment[1]}_METHOD_{experiment[0]}{'_' + ENDING if ENDING is not None else ''}", 'preds', 'masked')
        else:
            PREDS_FOLDER = os.path.join(nnUNet_results, DATASET_NAME, f"{MODEL}_{SIGNATURE}", f"MODE_{experiment[1]}_METHOD_{experiment[0]}{'_' + ENDING if ENDING is not None else ''}", 'preds', 'notMasked')
    else:
        if MASKED:
            PREDS_FOLDER = os.path.join(nnUNet_results, DATASET_NAME, f"{MODEL}_{SIGNATURE}", f"LOC_{experiment[1]}_MODE_{experiment[0]}{'_' + ENDING if ENDING is not None else ''}", 'preds', 'masked')
        else:
            PREDS_FOLDER = os.path.join(nnUNet_results, DATASET_NAME, f"{MODEL}_{SIGNATURE}", f"LOC_{experiment[1]}_MODE_{experiment[0]}{'_' + ENDING if ENDING is not None else ''}", 'preds', 'notMasked')

    # Validate folders
    if not os.path.exists(PREDS_FOLDER):
        raise FileNotFoundError(f"Predictions folder not found: {PREDS_FOLDER}")
    if not os.path.exists(GT_FOLDER):
        raise FileNotFoundError(f"Ground truth folder not found: {GT_FOLDER}")

    # Define output file paths
    EVALUATION_RESULTS_FILE = os.path.join(summary_dir, f"evaluation_results_{MODEL}_{SIGNATURE}{'_masked' if MASKED else ''}.json")
    OUTPUT_SUMMARY_FILE = os.path.join(summary_dir, f"output_summary_{MODEL}_{SIGNATURE}{'_masked' if MASKED else ''}.json")

    # Compute evaluation metrics
    print(f"Computing evaluation metrics for {MODEL} (masked={MASKED})...")
    results = compute_metrics(PREDS_FOLDER, GT_FOLDER)
    save_results_to_json(results, EVALUATION_RESULTS_FILE)
    print(f"Evaluation results saved to {EVALUATION_RESULTS_FILE}")

    # Compute statistical summary
    print("Analyzing statistical properties and extremes...")
    summary = compute_statistics_with_extremes(EVALUATION_RESULTS_FILE, OUTPUT_SUMMARY_FILE)
    print(f"Summary statistics saved to {OUTPUT_SUMMARY_FILE}")

# %%

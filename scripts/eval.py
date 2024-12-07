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
from set_env import set_environment_variables
from TumorNetSolvers.evaluation.eval_preds_folder import compute_metrics
from TumorNetSolvers.evaluation.file_io import save_results_to_json
from TumorNetSolvers.evaluation.statistics import compute_statistics_with_extremes
set_environment_variables()

# Define environment variables
nnUNet_preprocessed = os.getenv('nnUNet_preprocessed')
nnUNet_results = os.getenv('nnUNet_results')

# Configuration
DATASET_NAME = 'Dataset500_Brain' 
MODELS = ['nnUnet']  # List of models for evaluation
SIGNATURE = '10k'
MASKED = False  # Binary flag for masked evaluation

# Define output directory for performance summaries
summary_dir = os.path.join("performance_summaries")
os.makedirs(summary_dir, exist_ok=True)

# Determine ground truth folder
if MASKED:
    GT_FOLDER = os.path.join(nnUNet_results, DATASET_NAME, 'masked_gt')
    if not os.path.exists(GT_FOLDER) or len(os.listdir(GT_FOLDER)) == 0:
        raise FileNotFoundError(f"Masked ground truth folder is missing or empty: {GT_FOLDER}")
else:
    GT_FOLDER = os.path.join(nnUNet_preprocessed, DATASET_NAME, 'nnUNetPlans_3d_fullres')

# Evaluate each model
for MODEL in MODELS:
    # Determine predictions folder
    if MASKED:
        PREDS_FOLDER = os.path.join(nnUNet_results, DATASET_NAME, 'preds', f'_{MODEL}_{SIGNATURE}/masked')
    else:
        PREDS_FOLDER = os.path.join(nnUNet_results, DATASET_NAME, 'preds', f'_{MODEL}_{SIGNATURE}/notMasked')

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

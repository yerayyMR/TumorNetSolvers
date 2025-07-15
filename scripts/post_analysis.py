# %%
# Prints the metrics
import json
import os

# Define the path to the JSON files folder
json_folder = os.path.join(os.path.dirname(__file__), '..', 'performance_summaries', 'init2')

# List of JSON files you want to read
json_files = [
    'evaluation_results_nnUnet_10k.json',
    'output_summary_nnUnet_10k.json',
]

for json_file in json_files:
    file_path = os.path.join(json_folder, json_file)
    
    # Load JSON data
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    print(f"Data from {json_file}:")
    
    # Your existing code to print metrics and stats
    for metric, stats in data.items():
        print(f"Metric: {metric}")
        for key, value in stats.items():
            if isinstance(value, list):
                print(f"  {key}: {value[0]} with value {value[1]}")
            else:
                print(f"  {key}: {value}")
        print()
# %%
# Plots the dice score
import matplotlib.pyplot as plt
# Extract Dice metrics and thresholds
dice_thresholds = []
dice_means = []

for key in sorted(data.keys()):
    if key.startswith("Dice_"):
        thresh = float(key.split("_")[1])
        dice_thresholds.append(thresh)
        # Parse Mean ± SE string like "0.90555739 ± nan"
        mean_se_str = data[key]["Mean \u00b1 SE"]
        mean_val = float(mean_se_str.split("±")[0].strip())
        dice_means.append(mean_val)

plt.figure(figsize=(8,5))
plt.plot(dice_thresholds, dice_means, marker='o')
plt.title("Dice Score vs Threshold")
plt.xlabel("Threshold")
plt.ylabel("Dice Score (Mean)")
plt.grid(True)
plt.show()
metrics = ['aMSE', 'MAE', 'SSIM']
means = []
for metric in metrics:
    mean_se_str = data[metric]["Mean \u00b1 SE"]
    mean_val = float(mean_se_str.split("±")[0].strip())
    means.append(mean_val)

plt.figure(figsize=(6,4))
plt.bar(metrics, means, color=['skyblue', 'orange', 'green'])
plt.ylabel("Mean Value")
plt.title("Mean values of aMSE, MAE, SSIM")
plt.show()
# %%
# Tries to plot the generated images
#from TumorGrowthToolkit.FK import Solver as FKSolver
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import time
import scipy.ndimage
import nibabel as nib
input_tissue_path = '/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset700_Brain/init_1k/infer_params/_nnUnet_experiment/optimizeOutputPatients/BRAIN_p4/input_tissue.nii.gz'
output_ground_truth_path = '/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset700_Brain/init_1k/infer_params/_nnUnet_experiment/optimizeOutputPatients/BRAIN_p4/output_ground_truth.nii.gz'
prediction_masked_path = '/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset700_Brain/init_1k/infer_params/_nnUnet_experiment/optimizeOutputPatients/BRAIN_p4/prediction_masked.nii.gz'
prediction_thresholded_path = '/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset700_Brain/init_1k/infer_params/_nnUnet_experiment/optimizeOutputPatients/BRAIN_p4/prediction_thresholded.nii.gz'
data = nib.load(output_ground_truth_path).get_fdata()
print(f"Type: {type(data)}")
print(f"Shape: {data.shape}")
print(f"Dtype: {data.dtype}")
print(f"Min/Max: {data.min()} / {data.max()}")
z = data.shape[2] // 2  # middle slice
plt.imshow(data[:, :, z], cmap='gray')
plt.title(f"Slice {z}")
plt.colorbar()
plt.axis('off')
plt.show()
data = nib.load(prediction_masked_path).get_fdata()
plt.imshow(data[:, :, z], cmap='gray')
plt.title(f"Slice {z}")
plt.colorbar()
plt.axis('off')
plt.show()
# Create custom color maps
# Idea based on what Jonas did
'''cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap', ['black', 'white'], 256)
cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap2', ['black', 'green', 'yellow', 'red'], 256)
def plot_tumor_states(wm_data, initial_state, final_state, slice_index):
    plt.figure(figsize=(12, 6))

    # Plot initial state
    plt.subplot(1, 2, 1)
    plt.imshow(wm_data[:, :, slice_index], cmap=cmap1, vmin=0, vmax=1, alpha=1)
    plt.imshow(initial_state[:, :, slice_index], cmap=cmap2, vmin=0, vmax=1, alpha=0.65)
    plt.title("Initial Tumor State")

    # Plot final state
    plt.subplot(1, 2, 2)
    plt.imshow(wm_data[:, :, slice_index], cmap=cmap1, vmin=0, vmax=1, alpha=1)
    plt.imshow(final_state[:, :, slice_index], cmap=cmap2, vmin=0, vmax=1, alpha=0.65)
    plt.title("Final Tumor State")
    plt.show()

# Run the FK_solver and plot the results
start_time = time.time()
fk_solver = FKSolver(parameters)
result = fk_solver.solve()
end_time = time.time()  # Store the end time
execution_time = int(end_time - start_time)  # Calculate the difference

print(f"Execution Time: {execution_time} seconds")
if result['success']:
    print("Simulation successful!")
    plot_tumor_states(wm_data, result['initial_state'], result['final_state'], NzT)
    plot_time_series(wm_data,result['time_series'], NzT)
else:
    print("Error occurred:", result['error'])'''
# %%
import numpy as np
masked_dir = '/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset700_Brain/forward_0_coeff/preds/_nnUnet_10k/masked/BRAIN_p4.npy'
notMasked_dir = '/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset700_Brain/forward_0_coeff/preds/_nnUnet_10k/notMasked/BRAIN_p4.npy'
data = np.load(masked_dir)
data_notMasked = np.load(notMasked_dir)
print(f"Type: {type(data)}")
print(f"Shape: {data.shape}")
print(f"Dtype: {data.dtype}")
print(f"Min/Max: {data.min()} / {data.max()}")
import matplotlib.pyplot as plt

# Remove batch or channel dim
volume = data[0]  # Now shape is (64, 64, 64)
volume_notMasked = data_notMasked[0]

# Show a central slice
plt.figure()
plt.imshow(volume[32], cmap='gray')
plt.title("Masked slice 32 - forward_1k_0_coeff - p4")
plt.show()

plt.figure()
plt.imshow(volume_notMasked[32], cmap='gray')
plt.title("Not masked slice 32 - forward_1k_0_coeff - p4")
plt.show()
# %%

# Comparison of different patients and different cases of images --> After parameter update
import nibabel as nib
import matplotlib.pyplot as plt

# === File paths ===
paths = {
    'p1': [
        ('0_coeff', '/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset700_Brain/init_1k_0_coeff/infer_params/_nnUnet_experiment/optimizeOutputPatients/BRAIN_p1/prediction_masked.nii.gz'),
        ('0.5_coeff', '/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset700_Brain/init_1k_0.5_coeff/infer_params/_nnUnet_experiment/optimizeOutputPatients/BRAIN_p1/prediction_masked.nii.gz'),
        ('gt_coeff', '/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset700_Brain/init_1k/infer_params/_nnUnet_experiment/optimizeOutputPatients/BRAIN_p1/prediction_masked.nii.gz'),
        ('ground_truth_image', '/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset700_Brain/init_1k/infer_params/_nnUnet_experiment/optimizeOutputPatients/BRAIN_p1/output_ground_truth.nii.gz'),
    ],
    'p4': [
        ('0_coeff', '/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset700_Brain/init_1k_0_coeff/infer_params/_nnUnet_experiment/optimizeOutputPatients/BRAIN_p4/prediction_masked.nii.gz'),
        ('0.5_coeff', '/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset700_Brain/init_1k_0.5_coeff/infer_params/_nnUnet_experiment/optimizeOutputPatients/BRAIN_p4/prediction_masked.nii.gz'),
        ('gt_coeff', '/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset700_Brain/init_1k/infer_params/_nnUnet_experiment/optimizeOutputPatients/BRAIN_p4/prediction_masked.nii.gz'),
        ('ground_truth_image', '/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset700_Brain/init_1k/infer_params/_nnUnet_experiment/optimizeOutputPatients/BRAIN_p4/output_ground_truth.nii.gz'),
    ]
}

# === Plot setup ===
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Init Predictions', fontsize=20, weight='bold')

for row_idx, patient in enumerate(['p1', 'p4']):
    for col_idx, (label, path) in enumerate(paths[patient]):
        data = nib.load(path).get_fdata()
        z = data.shape[2] // 2  # middle slice
        ax = axes[row_idx, col_idx]
        im = ax.imshow(data[:, :, z], cmap='gray')
        ax.set_title(f"{patient.upper()} - {label}")
        ax.axis('off')
        fig.colorbar(im, ax=ax, shrink=0.6)

plt.tight_layout()
plt.show()

# === Forward Paths ===
paths_forward = {
    'p1': [
        ('0_coeff', '/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset700_Brain/forward_1k_0_coeff/infer_params/_nnUnet_experiment/optimizeOutputPatients/BRAIN_p1/prediction_masked.nii.gz'),
        ('0.5_coeff', '/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset700_Brain/forward_1k_0.5_coeff/infer_params/_nnUnet_experiment/optimizeOutputPatients/BRAIN_p1/prediction_masked.nii.gz'),
        ('gt_coeff', '/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset700_Brain/forward_1k/infer_params/_nnUnet_experiment/optimizeOutputPatients/BRAIN_p1/prediction_masked.nii.gz'),
        ('ground_truth_image', '/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset700_Brain/forward_1k/infer_params/_nnUnet_experiment/optimizeOutputPatients/BRAIN_p1/output_ground_truth.nii.gz'),
    ],
    'p4': [
        ('0_coeff', '/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset700_Brain/forward_1k_0_coeff/infer_params/_nnUnet_experiment/optimizeOutputPatients/BRAIN_p4/prediction_masked.nii.gz'),
        ('0.5_coeff', '/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset700_Brain/forward_1k_0.5_coeff/infer_params/_nnUnet_experiment/optimizeOutputPatients/BRAIN_p4/prediction_masked.nii.gz'),
        ('gt_coeff', '/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset700_Brain/forward_1k/infer_params/_nnUnet_experiment/optimizeOutputPatients/BRAIN_p4/prediction_masked.nii.gz'),
        ('ground_truth_image', '/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset700_Brain/forward_1k/infer_params/_nnUnet_experiment/optimizeOutputPatients/BRAIN_p4/output_ground_truth.nii.gz'),
    ]
}

# === Plot setup ===
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Forward Predictions', fontsize=20, weight='bold')

for row_idx, patient in enumerate(['p1', 'p4']):
    for col_idx, (label, path) in enumerate(paths_forward[patient]):
        data = nib.load(path).get_fdata()
        z = data.shape[2] // 2  # middle slice
        ax = axes[row_idx, col_idx]
        im = ax.imshow(data[:, :, z], cmap='gray')
        ax.set_title(f"{patient.upper()} - {label}")
        ax.axis('off')
        fig.colorbar(im, ax=ax, shrink=0.6)

plt.tight_layout()
plt.show()
# %%
# COmparison for Jonas based on the inference predictions

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# === File paths ===
paths = {
    'p1': [
        ('0_coeff', '/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset700_Brain/forward_1k_0_coeff/preds/_nnUnet_10k/masked/BRAIN_p1.npy'),
        ('0.5_coeff', '/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset700_Brain/forward_1k_0.5_coeff/preds/_nnUnet_10k/masked/BRAIN_p1.npy'),
        ('gt_coeff', '/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset700_Brain/forward_1k/preds/_nnUnet_10k/masked/BRAIN_p1.npy'),
        ('ground_truth_image', '/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset700_Brain/init_1k/infer_params/_nnUnet_experiment/optimizeOutputPatients/BRAIN_p1/output_ground_truth.nii.gz'),
    ],
    'p4': [
        ('0_coeff', '/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset700_Brain/forward_1k_0_coeff/preds/_nnUnet_10k/masked/BRAIN_p4.npy'),
        ('0.5_coeff', '/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset700_Brain/forward_1k_0.5_coeff/preds/_nnUnet_10k/masked/BRAIN_p4.npy'),
        ('gt_coeff', '/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset700_Brain/forward_1k/preds/_nnUnet_10k/masked/BRAIN_p4.npy'),
        ('ground_truth_image', '/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset700_Brain/init_1k/infer_params/_nnUnet_experiment/optimizeOutputPatients/BRAIN_p4/output_ground_truth.nii.gz'),
    ]
}

# === Plot setup ===
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Forward Predictions - Masked (.npy)', fontsize=20, weight='bold')

for row_idx, patient in enumerate(['p1', 'p4']):
    for col_idx, (label, path) in enumerate(paths[patient]):
        if path.endswith('.npy'):
            data = np.load(path)[0]  # remove batch dimension
        else:
            data = nib.load(path).get_fdata()

        z = data.shape[0] // 2  # show middle slice in axis 0
        ax = axes[row_idx, col_idx]
        im = ax.imshow(data[z], cmap='gray')
        ax.set_title(f"{patient.upper()} - {label}")
        ax.axis('off')
        fig.colorbar(im, ax=ax, shrink=0.6)

plt.tight_layout()
plt.show()

# === File paths ===
paths = {
    'p1': [
        ('0_coeff', '/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset700_Brain/init_1k_0_coeff/preds/_nnUnet_10k/masked/BRAIN_p1.npy'),
        ('0.5_coeff', '/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset700_Brain/init_1k_0.5_coeff/preds/_nnUnet_10k/masked/BRAIN_p1.npy'),
        ('gt_coeff', '/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset700_Brain/init_1k/preds/_nnUnet_10k/masked/BRAIN_p1.npy'),
        ('ground_truth_image', '/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset700_Brain/init_1k/infer_params/_nnUnet_experiment/optimizeOutputPatients/BRAIN_p1/output_ground_truth.nii.gz'),
    ],
    'p4': [
        ('0_coeff', '/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset700_Brain/init_1k_0_coeff/preds/_nnUnet_10k/masked/BRAIN_p4.npy'),
        ('0.5_coeff', '/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset700_Brain/init_1k_0.5_coeff/preds/_nnUnet_10k/masked/BRAIN_p4.npy'),
        ('gt_coeff', '/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset700_Brain/init_1k/preds/_nnUnet_10k/masked/BRAIN_p4.npy'),
        ('ground_truth_image', '/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset700_Brain/init_1k/infer_params/_nnUnet_experiment/optimizeOutputPatients/BRAIN_p4/output_ground_truth.nii.gz'),
    ]
}

# === Plot setup ===
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Init Predictions - Masked (.npy)', fontsize=20, weight='bold')

for row_idx, patient in enumerate(['p1', 'p4']):
    for col_idx, (label, path) in enumerate(paths[patient]):
        if path.endswith('.npy'):
            data = np.load(path)[0]  # remove batch dim
        else:
            data = nib.load(path).get_fdata()

        z = data.shape[0] // 2  # middle slice (axis 0)
        ax = axes[row_idx, col_idx]
        im = ax.imshow(data[z], cmap='gray')
        ax.set_title(f"{patient.upper()} - {label}")
        ax.axis('off')
        fig.colorbar(im, ax=ax, shrink=0.6)

plt.tight_layout()
plt.show()
# %%

# For different experiments


import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
EXPERIMENTS = [ ['c', 'a_downsampling'], ['a', 'a_downsampling'],
                    ['c', 'b_downsampling'], ['a', 'b_downsampling'],
                    ['c', 'inputs'], ['a', 'inputs'],
                    ['c', 'b_bottleneck'], ['a', 'b_bottleneck']]
for experiment in EXPERIMENTS:
    # === File paths ===
    paths = {
        'p1': [
            ('0_coeff', f'/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset700_Brain/LOC_{experiment[1]}_MODE_{experiment[0]}_0_coeff/preds/_nnUnet_10k/masked/BRAIN_p1.npy'),
            ('0.5_coeff', f'/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset700_Brain/LOC_{experiment[1]}_MODE_{experiment[0]}_05_coeff/preds/_nnUnet_10k/masked/BRAIN_p1.npy'),
            ('gt_coeff', f'/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset700_Brain/LOC_{experiment[1]}_MODE_{experiment[0]}/preds/_nnUnet_10k/masked/BRAIN_p1.npy'),
            ('ground_truth_image', '/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset700_Brain/init_1k/infer_params/_nnUnet_experiment/optimizeOutputPatients/BRAIN_p1/output_ground_truth.nii.gz'),
        ],
        'p4': [
            ('0_coeff', f'/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset700_Brain/LOC_{experiment[1]}_MODE_{experiment[0]}_0_coeff/preds/_nnUnet_10k/masked/BRAIN_p4.npy'),
            ('0.5_coeff', f'/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset700_Brain/LOC_{experiment[1]}_MODE_{experiment[0]}_05_coeff/preds/_nnUnet_10k/masked/BRAIN_p4.npy'),
            ('gt_coeff', f'/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset700_Brain/LOC_{experiment[1]}_MODE_{experiment[0]}/preds/_nnUnet_10k/masked/BRAIN_p4.npy'),
            ('ground_truth_image', '/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset700_Brain/init_1k/infer_params/_nnUnet_experiment/optimizeOutputPatients/BRAIN_p4/output_ground_truth.nii.gz'),
        ]
    }

    # === Plot setup ===
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'LOC_{experiment[1]}_MODE_{experiment[0]}', fontsize=20, weight='bold')

    for row_idx, patient in enumerate(['p1', 'p4']):
        for col_idx, (label, path) in enumerate(paths[patient]):
            if path.endswith('.npy'):
                data = np.load(path)[0]  # remove batch dimension
            else:
                data = nib.load(path).get_fdata()

            z = data.shape[0] // 2  # show middle slice in axis 0
            ax = axes[row_idx, col_idx]
            im = ax.imshow(data[z], cmap='gray')
            ax.set_title(f"{patient.upper()} - {label}")
            ax.axis('off')
            fig.colorbar(im, ax=ax, shrink=0.6)

    plt.tight_layout()
    plt.show()

# %%
import numpy as np
import os
import json
import matplotlib.pyplot as plt

# Histogram of metrics per experiment through all patients on test set
EXPERIMENTS = [ ['c', 'a_downsampling'], ['a', 'a_downsampling'],
                    ['c', 'b_downsampling'], ['a', 'b_downsampling'],
                    ['c', 'inputs'], ['a', 'inputs'],
                    ['c', 'b_bottleneck'], ['a', 'b_bottleneck'],
                    ['c', 'a_bottleneck'], ['a', 'a_bottleneck']]

MODEL = ['nnUnet']
DATASET_NAME = "Dataset900_Brain"
# List of JSON files you want to read
json_files = [
    f'evaluation_results_{MODEL[0]}_10k.json',
    f'output_summary_{MODEL[0]}_10k.json',
]
# This dictionary will collect all sample_metrics per experiment key
experiment_metrics = {}

for experiment in EXPERIMENTS:
    json_folder = os.path.join(os.path.dirname(__file__), '..', 'performance_summaries', DATASET_NAME , MODEL[0], f'LOC_{experiment[1]}_MODE_{experiment[0]}')
    all_data = {}

    for json_file in json_files:
        file_path = os.path.join(json_folder, json_file)

        with open(file_path, 'r') as f:
            data = json.load(f)

        saved_data = {}

        if "sample_metrics" in data:
            saved_data = data["sample_metrics"]
        else:
            for metric, stats in data.items():
                saved_data[metric] = {}
                for key, value in stats.items():
                    if isinstance(value, list):
                        saved_data[metric][key] = (value[0], value[1])
                    else:
                        saved_data[metric][key] = value

        all_data[json_file] = saved_data

    # Extract sample_metrics for this experiment (only from the first JSON file)
    sample_metrics = all_data[f'evaluation_results_{MODEL[0]}_10k.json']

    # Store sample_metrics by experiment key
    experiment_key = f"LOC_{experiment[1]}_MODE_{experiment[0]}"
    experiment_metrics[experiment_key] = sample_metrics

# Now plot combined histograms per metric across experiments

# Get metric names from one sample of the first experiment
some_exp_key = next(iter(experiment_metrics))
some_sample = next(iter(experiment_metrics[some_exp_key]))
metric_names = list(experiment_metrics[some_exp_key][some_sample].keys())

for metric in metric_names:
    # Collect all values for this metric per experiment
    values_per_experiment = {}
    all_values = []
    for exp_key, samples in experiment_metrics.items():
        values = [samples[sample][metric] for sample in samples]
        values_per_experiment[exp_key] = values
        all_values.extend(values)

    # Define common bins for all experiments for this metric
    bins = np.histogram_bin_edges(all_values, bins=30)

    # Compute histograms
    counts_per_exp = {}
    for exp_key, values in values_per_experiment.items():
        counts, _ = np.histogram(values, bins=bins)
        counts_per_exp[exp_key] = counts

    # Plot grouped bars
    width = 0.8 / len(experiment_metrics)
    x = np.arange(len(bins) - 1)

    plt.figure(figsize=(12, 6))
    for i, (exp_key, counts) in enumerate(counts_per_exp.items()):
        plt.bar(x + i * width, counts, width=width, label=exp_key)

    plt.xticks(
        x + width * (len(experiment_metrics) - 1) / 2,
        [f"{bins[i]:.3g}-{bins[i+1]:.3g}" for i in range(len(bins) - 1)],
        rotation=45, ha='right'
    )
    plt.xlabel(metric)
    plt.ylabel('Frequency')
    plt.title(f"Grouped Histogram of '{metric}' Across Experiments")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# --- 1) Dice Score vs Threshold Plot (one line per experiment) ---

plt.figure(figsize=(10, 6))

for exp_key, sample_metrics in experiment_metrics.items():
    # Extract Dice keys sorted by threshold number
    dice_thresholds = []
    dice_means = []

    # sample_metrics structure: { sample_name: {metric: value, ...}, ... }
    # We want mean values stored in `sample_metrics` dictionary keys that start with "Dice_"
    # But your previous example used a dict structure like:
    # data[key]["Mean ± SE"] — we don't have that here.
    # Instead, we have many samples with a single value each.
    # So, we will calculate the mean per dice threshold over all samples

    # Collect dice keys
    dice_keys = sorted([k for k in list(next(iter(sample_metrics.values())).keys()) if k.startswith("Dice_")],
                       key=lambda x: float(x.split("_")[1]))

    for key in dice_keys:
        thresh = float(key.split("_")[1])
        dice_thresholds.append(thresh)

        # Gather all samples' dice scores for this metric key
        values = [sample_metrics[sample][key] for sample in sample_metrics]
        mean_val = np.mean(values)
        dice_means.append(mean_val)

    plt.plot(dice_thresholds, dice_means, marker='o', label=exp_key)

plt.title("[UNet] Dice Score vs Threshold Across Experiments")
plt.xlabel("Threshold")
plt.ylabel("Dice Score (Mean)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# --- 2) Mean values bar chart for aMSE, MAE, SSIM (grouped bars per experiment) ---

metrics = ['aMSE', 'MAE', 'SSIM']

# Collect mean values per metric per experiment
means_per_metric = {metric: [] for metric in metrics}
experiment_keys = list(experiment_metrics.keys())

for metric in metrics:
    for exp_key in experiment_keys:
        sample_metrics = experiment_metrics[exp_key]
        values = [sample_metrics[sample][metric] for sample in sample_metrics]
        means_per_metric[metric].append(np.mean(values))

# Plot one bar plot per metric
for metric in metrics:
    values = means_per_metric[metric]
    
    # Determine sorting order: ascending for aMSE and MAE, descending for SSIM
    reverse = metric == 'SSIM'
    
    # Get indices of sorted values
    sorted_indices = np.argsort(values)[::-1] if reverse else np.argsort(values)
    
    # Set default colors
    colors = ['skyblue'] * len(values)
    
    # Assign colors to top 3
    if len(values) >= 1:
        colors[sorted_indices[0]] = 'green'
    if len(values) >= 2:
        colors[sorted_indices[1]] = 'yellow'
    if len(values) >= 3:
        colors[sorted_indices[2]] = 'red'
    
    # Plot
    plt.figure(figsize=(9, 6))  # Slightly larger figure
    bars = plt.bar(experiment_keys, values, color=colors, edgecolor='black')
    plt.title(f"Mean {metric} Across Experiments")
    plt.xlabel("Experiment")
    plt.ylabel(f"Mean {metric} Value")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add buffer above the tallest bar to avoid label collision
    y_max = max(values) * 1.15  # 15% headroom
    plt.ylim(top=y_max)
    
    # Add value labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01 * y_max, f"{yval:.6f}",
                 ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()
# %%
# Dice score plots
import numpy as np
import os
import json
import matplotlib.pyplot as plt

# Define models and their corresponding experiment lists
MODEL_EXPERIMENTS = {
    'nnUnet': [
        ['c', 'a_downsampling'], ['a', 'a_downsampling'],
        ['c', 'b_downsampling'], ['a', 'b_downsampling'],
        ['c', 'inputs'], ['a', 'inputs'],
        ['c', 'b_bottleneck'], ['a', 'b_bottleneck'],
        ['c', 'a_bottleneck'], ['a', 'a_bottleneck']
    ],
    'ViT': [
        ['MLP', 'one_token'],
        ['Linear', 'one_token'],
        ['MLP', 'mul_token'],
        ['MLP', 'embed_concat'],
        ['Linear', 'embed_concat'],
        ['MLP', 'embed_add'],
        ['Linear', 'embed_add']
    ]
}

DATASET_NAME = "Dataset900_Brain"
BASE_PATH = os.path.join(os.path.dirname(__file__), '..', 'performance_summaries')

all_dice_means = []
all_model_experiment_metrics = {}

# Step 1: Load data and collect mean Dice values
for model, experiments in MODEL_EXPERIMENTS.items():
    experiment_metrics = {}
    for exp in experiments:
        if model == "ViT":
            exp_folder = f'MODE_{exp[1]}_METHOD_{exp[0]}'
        else:
            exp_folder = f'LOC_{exp[1]}_MODE_{exp[0]}'

        json_folder = os.path.join(BASE_PATH, DATASET_NAME, model, exp_folder)
        json_file = f'evaluation_results_{model}_10k.json'
        eval_path = os.path.join(json_folder, json_file)

        if not os.path.exists(eval_path):
            print(f"Missing: {eval_path}")
            continue

        with open(eval_path, 'r') as f:
            data = json.load(f)

        if "sample_metrics" in data:
            sample_metrics = data["sample_metrics"]
        else:
            sample_metrics = {}
            for metric, samples in data.items():
                for sample_id, val in samples.items():
                    if sample_id not in sample_metrics:
                        sample_metrics[sample_id] = {}
                    sample_metrics[sample_id][metric] = val

        exp_key = exp_folder
        experiment_metrics[exp_key] = sample_metrics

        # Extract and store mean Dice values per threshold
        dice_keys = sorted(
            [k for k in next(iter(sample_metrics.values())).keys() if k.startswith("Dice_")],
            key=lambda x: float(x.split("_")[1])
        )
        for key in dice_keys:
            values = [sample[key] for sample in sample_metrics.values()]
            mean_val = np.mean(values)
            all_dice_means.append(mean_val)

    all_model_experiment_metrics[model] = experiment_metrics

# Step 2: Set y-axis limits based on Dice means only
y_min = max(0.0, min(all_dice_means) * 0.95)
y_max = min(1.0, max(all_dice_means) * 1.05)

# Step 3: Plot one figure per model
for model, experiment_metrics in all_model_experiment_metrics.items():
    plt.figure(figsize=(8, 6))

    for exp_key, sample_metrics in experiment_metrics.items():
        dice_keys = sorted(
            [k for k in next(iter(sample_metrics.values())).keys() if k.startswith("Dice_")],
            key=lambda x: float(x.split("_")[1])
        )

        thresholds = [float(k.split("_")[1]) for k in dice_keys]
        dice_means = [
            np.mean([sample[k] for sample in sample_metrics.values()])
            for k in dice_keys
        ]

        plt.plot(thresholds, dice_means, marker='o', label=exp_key)

    plt.title(f"[{model}] Dice Score vs Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Dice Score (Mean)")
    plt.grid(True)
    plt.ylim(y_min, y_max)
    plt.legend(fontsize='small', loc='lower left', bbox_to_anchor=(0, 0))
    plt.tight_layout()
    plt.show()


# %%
# Error over volume over the best scenario
import numpy as np
import torch
import os
import json
import matplotlib.pyplot as plt

# --- Configuration ---
DATASET_NAME = "Dataset900_Brain"
BASE_PATH = os.path.join(os.path.dirname(__file__), '..', 'performance_summaries')
from set_env import set_environment_variables
set_environment_variables()

# Define environment variables
nnUNet_preprocessed = os.getenv('nnUNet_preprocessed')
nnUNet_results = os.getenv('nnUNet_results')

MASKED = False  # Binary flag for masked evaluation

if MASKED:
    GT_FOLDER = os.path.join(nnUNet_results, DATASET_NAME, 'masked_gt')
    if not os.path.exists(GT_FOLDER) or len(os.listdir(GT_FOLDER)) == 0:
        raise FileNotFoundError(f"Masked ground truth folder is missing or empty: {GT_FOLDER}")
else:
    GT_FOLDER = os.path.join(nnUNet_preprocessed, DATASET_NAME, 'nnUNetPlans_3d_fullres')
MODEL_EXPERIMENTS = {
    'nnUnet': [
        ['c', 'b_downsampling']]
}

# --- Collect Volume and MSE ---
patient_volumes = []
patient_mses = []

for model, experiments in MODEL_EXPERIMENTS.items():
    for exp in experiments:
        if model == "ViT":
            exp_folder = f'MODE_{exp[1]}_METHOD_{exp[0]}'
        else:
            exp_folder = f'LOC_{exp[1]}_MODE_{exp[0]}'

        json_folder = os.path.join(BASE_PATH, DATASET_NAME, model, exp_folder)
        json_file = f'evaluation_results_{model}_10k.json'
        eval_path = os.path.join(json_folder, json_file)

        if not os.path.exists(eval_path):
            print(f"Missing: {eval_path}")
            continue

        with open(eval_path, 'r') as f:
            data = json.load(f)

        # Load sample metrics
        if "sample_metrics" in data:
            sample_metrics = data["sample_metrics"]
        else:
            sample_metrics = {}
            for metric, samples in data.items():
                for sample_id, val in samples.items():
                    if sample_id not in sample_metrics:
                        sample_metrics[sample_id] = {}
                    sample_metrics[sample_id][metric] = val

        # Process each patient
        for sample_id, metrics in sample_metrics.items():
            mse = metrics.get("aMSE", None)
            if mse is None:
                continue

            # Reconstruct GT filename
            gt_filename = sample_id.replace('.npy', '_seg.npy')
            gt_path = os.path.join(GT_FOLDER, gt_filename)

            if not os.path.exists(gt_path):
                print(f"Missing GT: {gt_path}")
                continue

            filter = lambda x: (x > 0) * x  # filter zeros out zeros

            gt_np = np.load(gt_path)
            gt_tensor = torch.tensor(gt_np)
            gt_filtered = filter(gt_tensor)
            volume = torch.sum(gt_filtered).item()

            patient_volumes.append(volume)
            patient_mses.append(mse)

# --- Plot ---
plt.figure(figsize=(10, 6))
plt.scatter(patient_volumes, patient_mses, alpha=0.6, edgecolors='k')
plt.xlabel("Ground Truth Volume (voxel count)")
plt.ylabel("MSE")
plt.title("Patient MSE vs Volume")
plt.grid(True)
plt.tight_layout()
plt.show()
# %%
# Number of parameters
import torch

def count_params(obj):
    if isinstance(obj, torch.Tensor):
        return obj.numel()
    elif isinstance(obj, dict):
        return sum(count_params(v) for v in obj.values())
    else:
        return 0  # ignore non-tensor, non-dict entries

experiment = ['Linear', 'one_token']
MODEL = ["ViT"]
# Path to your checkpoint
pth_path = f"/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset900_Brain/Trainer__nnUNetPlans__3d_fullres/fold_train_val_test/_10k_{MODEL[0]}/MODE_{experiment[1]}_METHOD_{experiment[0]}/checkpoint_{MODEL[0]}_best_ema_loss.pth"

checkpoint = torch.load(pth_path, map_location='cpu')

if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

total_params = count_params(state_dict)
print(f"Total parameters: {total_params}")
# %%

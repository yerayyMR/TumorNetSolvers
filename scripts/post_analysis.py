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

# Comparison of different patients and different cases of images
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

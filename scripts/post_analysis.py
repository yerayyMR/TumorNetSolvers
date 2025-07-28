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
import os
import time
import sys
# For different experiments
current_dir = os.path.dirname(os.path.abspath(__file__))  # scripts directory
src_path = os.path.abspath(os.path.join(current_dir, '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)
#nnUNet_preprocessed = os.environ.get('nnUNet_preprocessed')
#from TumorNetSolvers.inference.inference_utils import CustomDataset, get_settings_and_file_paths
from torch.utils.data import DataLoader
from scipy.ndimage import center_of_mass
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
EXPERIMENTS = [ ['c', 'a_upsampling']]

'''def save_full_ground_truths(dataset_name: str, data_folder: str, output_base: str):
    """
    Iterate through test set and save full 3D ground truth tumor masks as NIfTI files.
    """
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

DATASET_NAME = "Dataset900_Brain"
nnUNet_preprocessed = os.environ.get('nnUNet_preprocessed')
DATA_FOLDER = os.path.join(nnUNet_preprocessed, DATASET_NAME,"nnUNetPlans_3d_fullres")
OUTPUT_BASE = '/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/preprocessed_data/gt'

save_full_ground_truths(DATASET_NAME, DATA_FOLDER, OUTPUT_BASE)'''

atlasTissue_123 = np.load('/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/preprocessed_data/Dataset900_Brain/nnUNetPlans_3d_fullres/BRAIN_p123.npy')[0]
atlasTissue_209 = np.load('/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/preprocessed_data/Dataset900_Brain/nnUNetPlans_3d_fullres/BRAIN_p209.npy')[0]

for experiment in EXPERIMENTS:
    # === File paths ===
    paths = {
        'p123': [
            ('0_coeff', f'/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset900_Brain/nnUnet/LOC_{experiment[1]}_MODE_{experiment[0]}_best_FK_50__00/preds/_nnUnet_10k/masked/BRAIN_p123.npy'),
            ('0.5_coeff', f'/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset900_Brain/nnUnet/LOC_{experiment[1]}_MODE_{experiment[0]}_best_FK_50__05/preds/_nnUnet_10k/masked/BRAIN_p123.npy'),
            ('gt_coeff', f'/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset900_Brain/nnUnet/LOC_{experiment[1]}_MODE_{experiment[0]}_best_FK_50/preds/_nnUnet_10k/masked/BRAIN_p123.npy'),
            ('ground_truth_image', '/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/preprocessed_data/gt/BRAIN_p123/ground_truth_full.nii.gz'),
        ],
        'p209': [
            ('0_coeff', f'/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset900_Brain/nnUnet/LOC_{experiment[1]}_MODE_{experiment[0]}_best_FK_50__00/preds/_nnUnet_10k/masked/BRAIN_p209.npy'),
            ('0.5_coeff', f'/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset900_Brain/nnUnet/LOC_{experiment[1]}_MODE_{experiment[0]}_best_FK_50__05/preds/_nnUnet_10k/masked/BRAIN_p209.npy'),
            ('gt_coeff', f'/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset900_Brain/nnUnet/LOC_{experiment[1]}_MODE_{experiment[0]}_best_FK_50/preds/_nnUnet_10k/masked/BRAIN_p209.npy'),
            ('ground_truth_image', '/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/preprocessed_data/gt/BRAIN_p209/ground_truth_full.nii.gz'),
        ]
    }

    # === Plot setup ===
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'LOC_{experiment[1]}_MODE_{experiment[0]}', fontsize=20, weight='bold')

    for row_idx, patient in enumerate(['p1', 'p4']):
        patient_paths = dict(paths[patient])
    
        # Load ground truth to compute center of mass
        gt_data = nib.load(patient_paths['ground_truth_image']).get_fdata()
        z, y, x = map(int, np.round(center_of_mass(gt_data)))  # fix center of mass
        for col_idx, (label, path) in enumerate(paths[patient]):
            if path.endswith('.npy'):
                data = np.load(path)[0]  # remove batch dimension
            else:
                data = nib.load(path).get_fdata()

            print(data.shape)
            ax = axes[row_idx, col_idx]
            
            if patient == "p1":
                ax.imshow(atlasTissue_123[:,:,z], cmap='gray')
            else:
                ax.imshow(atlasTissue_209[:,:,z], cmap='gray')
            im = ax.imshow(data[:,:,z], cmap='Reds', alpha=0.9 * data[:, :, z], interpolation='none')
            ax.set_title(f"{patient.upper()} - {label}")
            ax.axis('off')
            fig.colorbar(im, ax=ax, shrink=0.6)

    plt.tight_layout()
    plt.show()

# %%
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass

MODEL = "nnUnet"
EXPERIMENTS = [['c', 'a_upsampling']]
PATIENT_ID = "BRAIN_p875"

# Load the tissue backgrounds
atlasTissue = np.load(f'/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/preprocessed_data/Dataset900_Brain/nnUNetPlans_3d_fullres/{PATIENT_ID}.npy')[0]

# Function to format label
def format_label(label):
    if label == 'gt_coeff':
        return 'GT coeff'
    if label.endswith('_coeff'):
        coeff = float(label.replace('_coeff', ''))
        return f"{int(coeff * 100)}% coeff"
    return label

for experiment in EXPERIMENTS:
    if MODEL != "ViT":
        paths = {
            'p': [
                ('0_coeff', f'/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset900_Brain/{MODEL}/LOC_{experiment[1]}_MODE_{experiment[0]}_best_FK_50_0/preds/_{MODEL}_10k/masked/{PATIENT_ID}.npy'),
                ('0.2_coeff', f'/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset900_Brain/{MODEL}/LOC_{experiment[1]}_MODE_{experiment[0]}_best_FK_50_0.2/preds/_{MODEL}_10k/masked/{PATIENT_ID}.npy'),
                ('0.4_coeff', f'/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset900_Brain/{MODEL}/LOC_{experiment[1]}_MODE_{experiment[0]}_best_FK_50_0.4/preds/_{MODEL}_10k/masked/{PATIENT_ID}.npy'),
                ('0.6_coeff', f'/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset900_Brain/{MODEL}/LOC_{experiment[1]}_MODE_{experiment[0]}_best_FK_50_0.6/preds/_{MODEL}_10k/masked/{PATIENT_ID}.npy'),
                ('0.8_coeff', f'/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset900_Brain/{MODEL}/LOC_{experiment[1]}_MODE_{experiment[0]}_best_FK_50_0.8/preds/_{MODEL}_10k/masked/{PATIENT_ID}.npy'),
                ('gt_coeff', f'/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset900_Brain/{MODEL}/LOC_{experiment[1]}_MODE_{experiment[0]}_best_FK_50_1/preds/_{MODEL}_10k/masked/{PATIENT_ID}.npy'),
                ('ground_truth_image', f'/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/preprocessed_data/gt/{PATIENT_ID}/ground_truth_full.nii.gz'),
            ]
        }
    else:
        paths = {
            'p': [
                ('0_coeff', f'/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset900_Brain/{MODEL}/MODE_{experiment[1]}_METHOD_{experiment[0]}_best_FK_50_0/preds/_{MODEL}_10k/masked/{PATIENT_ID}.npy'),
                ('0.2_coeff', f'/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset900_Brain/{MODEL}/MODE_{experiment[1]}_METHOD_{experiment[0]}_best_FK_50_0.2/preds/_{MODEL}_10k/masked/{PATIENT_ID}.npy'),
                ('0.4_coeff', f'/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset900_Brain/{MODEL}/MODE_{experiment[1]}_METHOD_{experiment[0]}_best_FK_50_0.4/preds/_{MODEL}_10k/masked/{PATIENT_ID}.npy'),
                ('0.6_coeff', f'/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset900_Brain/{MODEL}/MODE_{experiment[1]}_METHOD_{experiment[0]}_best_FK_50_0.6/preds/_{MODEL}_10k/masked/{PATIENT_ID}.npy'),
                ('0.8_coeff', f'/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset900_Brain/{MODEL}/MODE_{experiment[1]}_METHOD_{experiment[0]}_best_FK_50_0.8/preds/_{MODEL}_10k/masked/{PATIENT_ID}.npy'),
                ('gt_coeff', f'/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset900_Brain/{MODEL}/MODE_{experiment[1]}_METHOD_{experiment[0]}_best_FK_50_1/preds/_{MODEL}_10k/masked/{PATIENT_ID}.npy'),
                ('ground_truth_image', f'/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/preprocessed_data/Dataset900_Brain/gt/{PATIENT_ID}/ground_truth_full.nii.gz'),
            ]
        }

    fig, axes = plt.subplots(1, 6, figsize=(20, 10))

    patient = 'p'
    patient_paths = dict(paths[patient])

    # Load ground truth to get center of mass slice index
    gt_data = nib.load(patient_paths['ground_truth_image']).get_fdata()
    z, y, x = map(int, np.round(center_of_mass(gt_data)))

    for col_idx, (label, path) in enumerate(paths[patient][:6]):
        if path.endswith('.npy'):
            data = np.load(path)[0]
        else:
            data = nib.load(path).get_fdata()

        ax = axes[col_idx]

        rotated_atlas = np.rot90(atlasTissue[:, :, z], 2)
        rotated_data = np.rot90(data[:, :, z], 2)

        ax.imshow(rotated_atlas, cmap='gray')
        ax.imshow(rotated_data, cmap='Reds', alpha=0.9 * rotated_data, interpolation='none')
        ax.set_title(format_label(label), fontsize=24)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# %%
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import gridspec
from scipy.ndimage import center_of_mass

MODELS = ["nnUnet", "TumorSurrogate", "ViT"]
EXPERIMENTS = [["c", "b_downsampling"], ['c', 'a_downsampling'], ['Linear', 'one_token']]
ARCH_LABELS = ["U-Net", "TS", "ViT"]
PATIENT_ID = "BRAIN_p19" #1973#1911#12726

# Load background tissue (atlas) - original full atlas (no mask)
atlasTissue_209 = np.load(
    f'/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/preprocessed_data/Dataset900_Brain/nnUNetPlans_3d_fullres/{PATIENT_ID}.npy'
)[0]

# --- Step 1: Determine global vmin/vmax for difference plots ---
diff_slices = []

for model, experiment in zip(MODELS, EXPERIMENTS):
    if model == "TumorSurrogate":
        path = f'/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset900_Brain/{model}/correctResNet/LOC_{experiment[1]}_MODE_{experiment[0]}_ablation/preds/_{model}_10k/masked/{PATIENT_ID}.npy'
    elif model == "nnUnet":
        path = f'/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset900_Brain/{model}/correctResNet/LOC_{experiment[1]}_MODE_{experiment[0]}_ablation/preds/_{model}_10k/masked/{PATIENT_ID}.npy'
    else:
        path = f'/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset900_Brain/{model}/MODE_{experiment[1]}_METHOD_{experiment[0]}_ablation/preds/_{model}_10k/masked/{PATIENT_ID}.npy'

    gt_true = nib.load(
        f'/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/preprocessed_data/gt/{PATIENT_ID}/ground_truth_full.nii.gz'
    ).get_fdata()
    gt_pred = np.load(path)[0]
    z, _, _ = map(int, np.round(center_of_mass(gt_true)))
    diff = gt_pred - gt_true
    diff_slices.append(diff[:, :, z])

all_diffs = np.stack(diff_slices)
vmax_global = 0.9#np.ceil(np.max(np.abs(all_diffs)) * 10) / 10
vmin_global = -0.9#-vmax_global

# --- Step 2: Plot with fixed layout using GridSpec ---
fig = plt.figure(figsize=(16, 5 * len(MODELS)))
gs = gridspec.GridSpec(len(MODELS), 4, width_ratios=[1, 1, 1, 0.05], wspace=0.02, hspace=0.25)

titles = ["Prediction", "Ground Truth", "Difference (Pred - GT)"]
cmaps = ['Reds', 'Reds', 'bwr']

for row_idx, (model, experiment, arch_label) in enumerate(zip(MODELS, EXPERIMENTS, ARCH_LABELS)):
    if model == "TumorSurrogate":
        path = f'/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset900_Brain/{model}/correctResNet/LOC_{experiment[1]}_MODE_{experiment[0]}_ablation/preds/_{model}_10k/masked/{PATIENT_ID}.npy'
    elif model == "nnUnet":
        path = f'/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset900_Brain/{model}/correctResNet/LOC_{experiment[1]}_MODE_{experiment[0]}_ablation/preds/_{model}_10k/masked/{PATIENT_ID}.npy'
    else:
        path = f'/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset900_Brain/{model}/MODE_{experiment[1]}_METHOD_{experiment[0]}_ablation/preds/_{model}_10k/masked/{PATIENT_ID}.npy'

    gt_pred = np.load(path)[0]
    gt_true = nib.load(
        f'/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/preprocessed_data/gt/{PATIENT_ID}/ground_truth_full.nii.gz'
    ).get_fdata()
    z, _, _ = map(int, np.round(center_of_mass(gt_true)))
    diff = gt_pred - gt_true
    overlays = [gt_pred, gt_true, diff]

    for col_idx, (overlay, cmap, title) in enumerate(zip(overlays, cmaps, titles)):
        ax = fig.add_subplot(gs[row_idx, col_idx])

        # --- Apply 180-degree rotation to all images ---
        rotated_atlas = np.rot90(atlasTissue_209[:, :, z], 2)
        ax.imshow(rotated_atlas, cmap='gray', vmin=0, vmax=1, interpolation='none')

        rotated_overlay = np.rot90(overlay[:, :, z], 2)

        if title == "Difference (Pred - GT)":
            masked = np.ma.masked_where(np.abs(rotated_overlay) < 1e-6, rotated_overlay)
            im = ax.imshow(masked, cmap=cmap, vmin=vmin_global, vmax=vmax_global)
        else:
            ax.imshow(rotated_overlay, cmap=cmap, alpha=0.9 * rotated_overlay)

        ax.axis('off')

        # Add black border
        border = Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                           linewidth=2, edgecolor='black', facecolor='none', zorder=10, clip_on=False)
        ax.add_patch(border)

        if row_idx == 0:
            ax.set_title(title, fontsize=22, weight='bold', pad=12)

        if col_idx == 0:
            ax.text(-0.18, 0.5, arch_label, fontsize=22, weight='bold',
                    va='center', ha='right', transform=ax.transAxes)

    cax = fig.add_subplot(gs[row_idx, 3])
    # Create colorbar
    cbar = fig.colorbar(im, cax=cax)

    # Set label and font size
    #cbar.set_label("Difference (Pred - GT)", fontsize=22, weight='bold', labelpad=20)

    # Calculate midpoints
    mid_min = (vmin_global + 0) / 2
    mid_max = (0 + vmax_global) / 2

    # Define ticks
    ticks = [vmin_global, mid_min, 0, mid_max, vmax_global]

    # Set ticks and labels
    cbar.set_ticks(ticks)
    cbar.ax.set_yticklabels([f"{vmin_global:.2f}", f"{mid_min:.2f}", "0", f"{mid_max:.2f}", f"{vmax_global:.2f}"])

    # Show ticks and labels on both top and bottom
    cbar.ax.yaxis.set_ticks_position('both')
    cbar.ax.tick_params(labelsize=22, direction='out', length=6, top=True, bottom=True)

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
# Dice score plots
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.lines import Line2D

MODEL_EXPERIMENTS = {

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
        ['c', 'a_bottleneck_after'], ['a', 'a_bottleneck_after'],
        ['c', 'a_upsampling_after'], ['a', 'a_upsampling_after'],
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
}

DATASET_NAME = "Dataset900_Brain"
BASE_PATH = os.path.join(os.path.dirname(__file__), '..', 'performance_summaries')

# Load data
all_model_experiment_metrics = {}
all_dice_means = []

for model, experiments in MODEL_EXPERIMENTS.items():
    experiment_metrics = {}
    for method, mode in experiments:
        if model == "TumorSurrogate":
            exp_folder = f'LOC_{mode}_MODE_{method}_new'
        elif model == "ViT":
            exp_folder = f'MODE_{mode}_METHOD_{method}'
        else:
            exp_folder = f'LOC_{mode}_MODE_{method}'

        json_folder = os.path.join(BASE_PATH, DATASET_NAME, model, 'correctResNet' if model == "TumorSurrogate" else "", exp_folder)
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

        experiment_metrics[(method, mode)] = sample_metrics

        # Save for y-axis bounds
        dice_keys = sorted([k for k in next(iter(sample_metrics.values())).keys() if k.startswith("Dice_")],
                           key=lambda x: float(x.split("_")[1]))
        for key in dice_keys:
            values = [sample[key] for sample in sample_metrics.values()]
            mean_val = np.mean(values)
            all_dice_means.append(mean_val)

    all_model_experiment_metrics[model] = experiment_metrics

# Axis bounds
y_min = max(0.0, min(all_dice_means) * 0.95)
y_max = min(1.0, max(all_dice_means) * 1.05)

# Create a color palette from matplotlib for consistent colors across plots
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def get_colors(n):
    cmap = plt.get_cmap('tab10')
    return [cmap(i) for i in range(n)]

fig, axes = plt.subplots(nrows=1, ncols=len(all_model_experiment_metrics), figsize=(26, 7.5), sharey=True)
if len(all_model_experiment_metrics) == 1:
    axes = [axes]

for ax, (model, experiment_metrics) in zip(axes, all_model_experiment_metrics.items()):
    
    if model == "nnUnet":
        ax.set_title("U-Net", fontsize=24)
    elif model == "TumorSurrogate":
        ax.set_title("TS", fontsize=24)
    else:
        ax.set_title(f"{model}", fontsize=24)
    
    ax.set_xlabel("Threshold", fontsize=24)
    ax.grid(True)
    ax.set_ylim(y_min, y_max)

    ax.tick_params(axis='both', which='major', labelsize=20)

    legend_entries = []

    if model == "ViT":
        # Group ViT experiments by token_type (mode)
        vit_grouped = defaultdict(dict)
        for (method, mode), metrics in experiment_metrics.items():
            vit_grouped[mode][method] = metrics

        colors = get_colors(len(vit_grouped))

        for color, (mode, method_dict) in zip(colors, vit_grouped.items()):
            dashed_line = None
            solid_line = None

            for method, linestyle in [('MLP', '--'), ('Linear', '-')]:
                method_key = method
                if method_key not in method_dict:
                    method_key_ext = method + '_ext'
                    if method_key_ext in method_dict:
                        method_key = method_key_ext
                    else:
                        continue

                sample_metrics = method_dict[method_key]
                dice_keys = sorted([k for k in next(iter(sample_metrics.values())).keys() if k.startswith("Dice_")],
                                key=lambda x: float(x.split("_")[1]))
                thresholds = [float(k.split("_")[1]) for k in dice_keys]
                dice_means = [np.mean([sample[k] for sample in sample_metrics.values()]) for k in dice_keys]

                ax.plot(thresholds, dice_means, marker='o', linestyle=linestyle,
                        color=color, linewidth=2)

                if linestyle == '--':
                    dashed_line = Line2D([0], [0], color=color, linestyle='--', linewidth=2)
                else:
                    solid_line = Line2D([0], [0], color=color, linestyle='-', linewidth=2)

            if dashed_line and solid_line:
                # Title: bold group name
                title_text = r"$\bf{" + mode.replace("_", "\ ") + "}$"
                legend_entries.append((title_text, Line2D([0], [0], color='none', linewidth=0)))
                legend_entries.append(("Linear", solid_line))
                legend_entries.append(("MLP", dashed_line))

    else:
        # Group experiments by location (mode)
        grouped = defaultdict(dict)
        for (method, mode), metrics in experiment_metrics.items():
            grouped[mode][method] = metrics

        colors = get_colors(len(grouped))

        for color, (mode, method_dict) in zip(colors, grouped.items()):
            added_line = None
            concat_line = None

            for method, linestyle in [('c', '-'), ('a', '--')]:
                if method not in method_dict:
                    continue

                sample_metrics = method_dict[method]
                dice_keys = sorted([k for k in next(iter(sample_metrics.values())).keys() if k.startswith("Dice_")],
                                key=lambda x: float(x.split("_")[1]))
                thresholds = [float(k.split("_")[1]) for k in dice_keys]
                dice_means = [np.mean([sample[k] for sample in sample_metrics.values()]) for k in dice_keys]

                ax.plot(thresholds, dice_means, marker='o', linestyle=linestyle,
                        color=color, linewidth=2)

                if method == 'a':
                    added_line = Line2D([0], [0], color=color, linestyle='--', linewidth=2)
                elif method == 'c':
                    concat_line = Line2D([0], [0], color=color, linestyle='-', linewidth=2)

            if added_line and concat_line:
                # Determine title text depending on mode start
                if mode.startswith('a') or mode.startswith('b'):
                    # Split mode like 'a_downsampling' -> ['a', 'downsampling']
                    parts = mode.split('_', 1)
                    prefix = parts[0]  # 'a' or 'b'
                    location = parts[1] if len(parts) > 1 else ''

                    # Clean location names for display
                    location_map = {
                        'downsampling': 'down.',
                        'upsampling': 'up.',
                        'bottleneck': 'bott.',
                        'inputs': 'input',
                        'upsampling_skip': 'up. skip',
                        'bottleneck_after': 'bott. (m)',
                        'b_bottleneck_after': 'bott. (m)',  # fallback if needed
                        'upsampling_after': 'up.'
                    }
                    location_disp = location_map.get(location, location.replace('_', ' '))

                    if prefix == 'a':
                        base_title = f"after {location_disp}"
                    else:
                        base_title = f"before {location_disp}"
                else:
                    # fallback for unknown modes
                    base_title = mode.replace("_", " ")

                title_text = r"$\bf{" + base_title.replace(' ', r'\ ') + "}$"

                # Title entry (only in first column)
                legend_entries.append((title_text, Line2D([0], [0], color='none', linewidth=0)))

                # concat and added lines side by side (in next row)
                legend_entries.append(("concat", concat_line))
                legend_entries.append(("added", added_line))

    # Add custom legend
    if model == "nnUnet" or model == "TumorSurrogate":
        desired_rows_first_col = 12
        total_desired_entries = desired_rows_first_col * 2
        current_entries = len(legend_entries)
        padding_needed = total_desired_entries - current_entries

        for _ in range(padding_needed):
            legend_entries.append(("", Line2D([0], [0], color='none', linewidth=0)))
    # Prepare final legend handles and labels for two-column display
    legend_labels = [label for label, _ in legend_entries]
    legend_handles = [handle for _, handle in legend_entries]

    if model == "c":
        ax.legend(
            legend_handles,
            legend_labels,
            ncol=3,
            fontsize=12,  # ← Change font size here
            loc='lower left',
            bbox_to_anchor=(0, 0),
            labelspacing=0.4,
            handlelength=2.5,
            borderaxespad=0.5
        )
    else:
        ax.legend(
            legend_handles,
            legend_labels,
            ncol=2,
            fontsize=16,  # ← Change font size here
            loc='lower left',
            bbox_to_anchor=(0, 0),
            labelspacing=0.4,
            handlelength=2.5,
            borderaxespad=0.5
        )

axes[0].set_ylabel("Dice Score (Mean)", fontsize=24)
plt.tight_layout()
plt.show()


# --- Combined plot with best run from each model ---

# Dictionary to store the best mean Dice scores per model
best_model_results = {}

for model, experiment_metrics in all_model_experiment_metrics.items():
    best_dice_means = None
    best_thresholds = None
    best_avg = -1

    for sample_metrics in experiment_metrics.values():
        dice_keys = sorted([k for k in next(iter(sample_metrics.values())).keys() if k.startswith("Dice_")],
                           key=lambda x: float(x.split("_")[1]))
        thresholds = [float(k.split("_")[1]) for k in dice_keys]
        dice_means = [np.mean([sample[k] for sample in sample_metrics.values()]) for k in dice_keys]

        current_avg = np.mean(dice_means)
        if current_avg > best_avg:
            best_avg = current_avg
            best_dice_means = dice_means
            best_thresholds = thresholds

    if best_thresholds is not None and best_dice_means is not None:
        best_model_results[model] = (best_thresholds, best_dice_means)

# Plot all best results in a single plot
fig, ax = plt.subplots(figsize=(10, 6))

model_styles = {
    'nnUnet': {'color': 'blue', 'linestyle': '-', 'label': 'U-Net'},
    'TumorSurrogate': {'color': 'green', 'linestyle': '--', 'label': 'TS'},
    'ViT': {'color': 'red', 'linestyle': '-.', 'label': 'ViT'}
}

for model, (thresholds, dice_means) in best_model_results.items():
    style = model_styles.get(model, {})
    if model == "nnUnet":
        ax.plot(thresholds, dice_means, marker='o', linewidth=2,
                color=style.get('color', 'black'),
                linestyle=style.get('linestyle', '-'),
                label=style.get('label', 'UNet'))
    else:
        ax.plot(thresholds, dice_means, marker='o', linewidth=2,
                color=style.get('color', 'black'),
                linestyle=style.get('linestyle', '-'),
                label=style.get('label', model))

# Axis labels and legend
ax.set_title("Best Runs Across Models", fontsize=22)
ax.set_xlabel("Threshold", fontsize=22)
ax.set_ylabel("Dice Score (Mean)", fontsize=22)
ax.tick_params(axis='both', which='major', labelsize=22)
ax.legend(fontsize=22)
ax.grid(True)
ax.set_ylim(y_min, y_max)

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
        ['c', 'a_bottleneck_after'], ['a', 'a_bottleneck_after'],
        ['c', 'a_upsampling_after'], ['a', 'a_upsampling_after'],
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
}

# --- Collect Volume and MSE ---
patient_volumes = []
patient_mses = []

for model, experiments in MODEL_EXPERIMENTS.items():
    for exp in experiments:
        if model == "ViT":
            exp_folder = f'MODE_{exp[1]}_METHOD_{exp[0]}'
        elif model == "TumorSurrogate":
            exp_folder = f'MODE_{exp[1]}_METHOD_{exp[0]}_new'
        else:
            exp_folder = f'LOC_{exp[1]}_MODE_{exp[0]}'

        if model == "TumorSurrogate":
            json_folder = os.path.join(BASE_PATH, DATASET_NAME, model, 'correctResNet', exp_folder)
        else:
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
plt.figure(figsize=(18, 6))
plt.scatter(patient_volumes, patient_mses, alpha=0.6, edgecolors='k')
plt.xlabel("Ground Truth Volume (voxel count)")
plt.ylabel("MSE")
plt.title("Patient MSE vs Volume")
plt.grid(True)
plt.tight_layout()
plt.show()


# --- Collect data per model ---
model_volumes = {}
model_mses = {}

for model, experiments in MODEL_EXPERIMENTS.items():
    patient_volumes = []
    patient_mses = []

    for exp in experiments:
        if model == "ViT":
            exp_folder = f'MODE_{exp[1]}_METHOD_{exp[0]}_best_FK_50'
        else:
            exp_folder = f'LOC_{exp[1]}_MODE_{exp[0]}_best_FK_50'

        if model == "TumorSurrogate":
            json_folder = os.path.join(BASE_PATH, DATASET_NAME, model, 'correctResNet', exp_folder)
        else:
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

            gt_np = np.load(gt_path)
            gt_tensor = torch.tensor(gt_np)
            gt_filtered = (gt_tensor > 0) * gt_tensor  # filter zeros
            volume = torch.sum(gt_filtered).item()

            patient_volumes.append(volume)
            patient_mses.append(mse)

    model_volumes[model] = patient_volumes
    model_mses[model] = patient_mses

# --- Compute common axis limits ---
all_volumes = np.concatenate(list(model_volumes.values()))
all_mses = np.concatenate(list(model_mses.values()))
x_min, x_max = all_volumes.min(), all_volumes.max()
y_min, y_max = all_mses.min(), all_mses.max()

# --- Plot per model ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

for ax, model in zip(axes, MODEL_EXPERIMENTS.keys()):
    ax.scatter(model_volumes[model], model_mses[model], alpha=0.6, edgecolors='k')
    ax.set_title(model, fontsize=16)
    ax.grid(True)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("GT Volume (voxels)", fontsize=12)

axes[0].set_ylabel("MSE", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# %%
# Plot MSE vs parameters
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
DATASET_NAME = "Dataset900_Brain"
BASE_PATH = os.path.join(os.path.dirname(__file__), '..', 'performance_summaries')
PTH_BASE = "/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/preprocessed_data"

MODEL_EXPERIMENTS = {
    'nnUnet': [['c', 'a_upsampling']],
    'TumorSurrogate': [['c', 'a_bottleneck_after']],
    'ViT': [['Linear', 'embed_concat']]
}

SEPARATE_PLOTS = True  # <--- Toggle this for one figure vs three separate plots

for model, experiments in MODEL_EXPERIMENTS.items():
    for experiment in experiments:
        print(f"\nProcessing: Model={model}, Experiment={experiment}")

        # Build paths
        if model == "ViT":
            exp_folder = f'MODE_{experiment[1]}_METHOD_{experiment[0]}_best_FK_50'
        else:
            exp_folder = f'LOC_{experiment[1]}_MODE_{experiment[0]}_best_FK_50'

        if model == "TumorSurrogate":
            json_path = os.path.join(BASE_PATH, DATASET_NAME, model, 'correctResNet', exp_folder,
                                     f'evaluation_results_{model}_10k.json')
            pth_path = os.path.join(PTH_BASE, DATASET_NAME, 'param_dict.pth')
        else:
            json_path = os.path.join(BASE_PATH, DATASET_NAME, model, exp_folder,
                                     f'evaluation_results_{model}_10k.json')
            pth_path = os.path.join(PTH_BASE, DATASET_NAME, 'param_dict.pth')

        # Load .pth
        try:
            pth_data = torch.load(pth_path, map_location="cpu")
        except Exception as e:
            print(f"Failed to load .pth: {pth_path} - {e}")
            continue

        # Load JSON
        if not os.path.exists(json_path):
            print(f"Missing JSON: {json_path}")
            continue

        with open(json_path, 'r') as f:
            data = json.load(f)

        # Extract sample metrics
        if "sample_metrics" in data:
            sample_metrics = data["sample_metrics"]
        else:
            sample_metrics = {}
            for metric, samples in data.items():
                for sample_id, val in samples.items():
                    if sample_id not in sample_metrics:
                        sample_metrics[sample_id] = {}
                    sample_metrics[sample_id][metric] = val

        # --- Collect values ---
        mse_values = []
        D_values = []
        rho_values = []

        for sample_id, metrics in sample_metrics.items():
            mse = metrics.get("aMSE", None)
            if mse is None:
                continue

            patient_id = sample_id.replace('.npy', '')
            if patient_id not in pth_data:
                print(f"Missing in .pth: {patient_id}")
                continue

            tensor = pth_data[patient_id]
            if not isinstance(tensor, torch.Tensor) or tensor.numel() < 2:
                print(f"Invalid tensor for {patient_id}")
                continue

            D = tensor[-2].item()
            rho = tensor[-1].item()
            mse_values.append(mse)
            D_values.append(D)
            rho_values.append(rho)

        if not mse_values:
            print(f"No valid samples for {model} - {experiment}")
            continue

        # --- Plotting ---
        title_base = f"{model}"

        if SEPARATE_PLOTS:
            # Plot 1: MSE vs D
            plt.figure(figsize=(6, 5))
            plt.scatter(D_values, mse_values, alpha=0.6, edgecolors='k')
            plt.xlabel("D")
            plt.ylabel("MSE")
            plt.title(f"{title_base} - MSE vs D")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            # Plot 2: MSE vs rho
            plt.figure(figsize=(6, 5))
            plt.scatter(rho_values, mse_values, alpha=0.6, edgecolors='k')
            plt.xlabel("rho")
            plt.ylabel("MSE")
            plt.title(f"{title_base} - MSE vs rho")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            # Plot 3: rho vs D (colored by MSE)
            plt.figure(figsize=(6, 5))
            sc = plt.scatter(D_values, rho_values, c=mse_values, cmap='viridis', alpha=0.8, edgecolors='k')
            plt.xlabel("D")
            plt.ylabel("rho")
            plt.title(f"{title_base} - rho vs D")
            plt.grid(True)
            cbar = plt.colorbar(sc)
            cbar.set_label("MSE")
            plt.tight_layout()
            plt.show()

        else:
            # All plots in a single figure
            plt.figure(figsize=(18, 5))

            plt.subplot(1, 3, 1)
            plt.scatter(D_values, mse_values, alpha=0.6, edgecolors='k')
            plt.xlabel("D")
            plt.ylabel("MSE")
            plt.title("MSE vs D")
            plt.grid(True)

            plt.subplot(1, 3, 2)
            plt.scatter(rho_values, mse_values, alpha=0.6, edgecolors='k')
            plt.xlabel("rho")
            plt.ylabel("MSE")
            plt.title("MSE vs rho")
            plt.grid(True)

            plt.subplot(1, 3, 3)
            sc = plt.scatter(D_values, rho_values, c=mse_values, cmap='viridis', alpha=0.8, edgecolors='k')
            plt.xlabel("D")
            plt.ylabel("rho")
            plt.title("rho vs D")
            plt.grid(True)
            cbar = plt.colorbar(sc)
            cbar.set_label("MSE")

            plt.suptitle(title_base, fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
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

experiment = ['Linear', 'embed_concat']
MODELS = ["ViT"]
# Path to your checkpoint
pth_path = f"/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset300_Brain/Trainer__nnUNetPlans__3d_fullres/fold_train_val_test/_10k_{MODELS[0]}/MODE_{experiment[1]}_METHOD_{experiment[0]}_DTI_50_3/best_cases/checkpoint_{MODELS[0]}_7961_best_ema_loss.pth"

checkpoint = torch.load(pth_path, map_location='cpu')

if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

total_params = count_params(state_dict)
print(f"Total parameters: {total_params}")
# %%
# Compute AUDC
import os
import json
import numpy as np

# Define models and their corresponding experiment lists
MODEL_EXPERIMENTS = {
    'nnUnet': [
        ['c', 'a_downsampling'], ['a', 'a_downsampling'],
        ['c', 'b_downsampling'], ['a', 'b_downsampling'],
        ['c', 'inputs'], ['a', 'inputs'],
        ['c', 'b_bottleneck'], ['a', 'b_bottleneck'],
        ['c', 'a_bottleneck'], ['a', 'a_bottleneck']
    ],
    'TumorSurrogate': [
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

for model, experiments in MODEL_EXPERIMENTS.items():
    model_results = {}

    for exp in experiments:
        # Build experiment folder name
        if model == "ViT":
            exp_folder = f'MODE_{exp[1]}_METHOD_{exp[0]}'
        elif model == "TumorSurrogate":
            exp_folder = f'LOC_{exp[1]}_MODE_{exp[0]}_new'
        else:
            exp_folder = f'LOC_{exp[1]}_MODE_{exp[0]}'

        # Build path to JSON file
        if model == "TumorSurrogate":
            json_folder = os.path.join(BASE_PATH, DATASET_NAME, model, 'correctResNet', exp_folder)
        else:
            json_folder = os.path.join(BASE_PATH, DATASET_NAME, model, exp_folder)

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

# Optional: save results to a JSON file
with open("/home/home/yeray_jonas/tumornetsolvers/performance_summaries/all_audc_results.json", "w") as f:
    json.dump(all_audc_results, f, indent=4)
# %%
with open(json_path, 'r') as f:
    data = json.load(f)
print(f"Keys in {json_path}: {list(data.keys())[:10]}")  # print first 10 keys
# %%
# Combination of error over volume and rho vs D
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def organize_data_per_model(DATASET_NAME, BASE_PATH, PTH_BASE, MODEL_EXPERIMENTS, MASKED=False):
    from set_env import set_environment_variables
    set_environment_variables()
    nnUNet_preprocessed = os.environ.get('nnUNet_preprocessed')
    nnUNet_results = os.environ.get('nnUNet_results')

    if MASKED:
        GT_FOLDER = os.path.join(nnUNet_results, DATASET_NAME, 'masked_gt')
        if not os.path.exists(GT_FOLDER) or len(os.listdir(GT_FOLDER)) == 0:
            raise FileNotFoundError(f"Masked ground truth folder missing or empty: {GT_FOLDER}")
    else:
        GT_FOLDER = os.path.join(nnUNet_preprocessed, DATASET_NAME, 'nnUNetPlans_3d_fullres')

    model_volumes = {}
    model_mses = {}
    D_dict = {}
    rho_dict = {}
    mse_dict = {}

    for model, experiments in MODEL_EXPERIMENTS.items():
        model_volumes[model] = []
        model_mses[model] = []
        D_dict[model] = []
        rho_dict[model] = []
        mse_dict[model] = []

        for experiment in experiments:
            if model == "ViT":
                exp_folder = f'MODE_{experiment[1]}_METHOD_{experiment[0]}_best_FK_50'
            else:
                exp_folder = f'LOC_{experiment[1]}_MODE_{experiment[0]}_best_FK_50'

            if model == "TumorSurrogate":
                base_json_folder = os.path.join(BASE_PATH, DATASET_NAME, model, 'correctResNet', exp_folder)
                pth_path = os.path.join(PTH_BASE, DATASET_NAME, 'param_dict.pth')
            else:
                base_json_folder = os.path.join(BASE_PATH, DATASET_NAME, model, exp_folder)
                pth_path = os.path.join(PTH_BASE, DATASET_NAME, 'param_dict.pth')

            json_path = os.path.join(base_json_folder, f'evaluation_results_{model}_10k.json')

            if not os.path.exists(json_path):
                print(f"Missing JSON: {json_path}")
                continue

            try:
                pth_data = torch.load(pth_path, map_location="cpu")
            except Exception as e:
                print(f"Failed to load .pth: {pth_path} - {e}")
                continue

            with open(json_path, 'r') as f:
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

            for sample_id, metrics in sample_metrics.items():
                mse = metrics.get("aMSE", None)
                if mse is None:
                    continue

                patient_id = sample_id.replace('.npy', '')
                gt_filename = sample_id.replace('.npy', '_seg.npy')
                gt_path = os.path.join(GT_FOLDER, gt_filename)

                if not os.path.exists(gt_path) or patient_id not in pth_data:
                    continue

                gt_np = np.load(gt_path)
                gt_tensor = torch.tensor(gt_np)
                volume = torch.sum((gt_tensor > 0) * gt_tensor).item()

                tensor = pth_data[patient_id]
                if not isinstance(tensor, torch.Tensor) or tensor.numel() < 2:
                    continue

                D = tensor[-2].item()
                rho = tensor[-1].item()

                model_volumes[model].append(volume)
                model_mses[model].append(mse)
                D_dict[model].append(D)
                rho_dict[model].append(rho)
                mse_dict[model].append(mse)

    return model_volumes, model_mses, D_dict, rho_dict, mse_dict


'''def plot_model_specific(model_volumes, model_mses, D_dict, rho_dict, mse_dict):
    for model in model_volumes:
        vols = np.array(model_volumes[model])
        mses = np.array(model_mses[model]) * 1000  # Multiply MSE by 1000
        Ds = np.array(D_dict[model])
        rhos = np.array(rho_dict[model])
        mse_col = np.array(mse_dict[model]) * 1000  # Also colorbar values

        if not (vols.size and mses.size and Ds.size and rhos.size and mse_col.size):
            print(f"Skipping {model} due to missing data.")
            continue

        fig = plt.figure(figsize=(20, 8))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])

        vmin = np.min(mse_col)
        vmax = np.max(mse_col)
        ticks = np.linspace(vmin, vmax, 5)
        tick_labels = [f"{t:.1f}" for t in ticks]

        # First subplot: Volume vs MSE (colored by MSE)
        ax1 = fig.add_subplot(gs[0])
        sc1 = ax1.scatter(vols, mses, c=mse_col, cmap='viridis', alpha=0.8, edgecolors='k')
        ax1.set_xlabel("GT Volume (x10³ voxels)", fontsize=22, labelpad=10)
        ax1.set_ylabel("MSE (×10⁻³)", fontsize=22, labelpad=10)
        ax1.tick_params(axis='both', which='major', labelsize=20)
        ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:.1f}"))
        ax1.grid(True)

        # Second subplot: rho vs D, colored by MSE
        ax2 = fig.add_subplot(gs[1])
        sc2 = ax2.scatter(Ds, rhos, c=mse_col, cmap='viridis', alpha=0.8, edgecolors='k')
        ax2.set_xlabel("D", fontsize=22, labelpad=10)
        ax2.set_ylabel("ρ", fontsize=22, labelpad=10)
        ax2.tick_params(axis='both', which='major', labelsize=20)
        ax2.grid(True)

        # Single shared colorbar
        cbar = fig.colorbar(sc2, ax=ax2, location='right', pad=0.02)
        cbar.set_label("MSE (×10⁻³)", fontsize=22, labelpad=12)
        cbar.set_ticks(ticks)
        cbar.ax.set_yticklabels(tick_labels)
        cbar.ax.tick_params(labelsize=20, pad=10)

        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.show()'''

def plot_model_specific(model_volumes, model_mses, D_dict, rho_dict, mse_dict):
    name_map = {
        'TumorSurrogate': 'TS',
        'nnUnet': 'U-Net',
        'ViT': 'ViT'
    }

    models = list(model_volumes.keys())
    num_models = len(models)

    fig, axes = plt.subplots(
        nrows=num_models,
        ncols=2,
        figsize=(28, 8.5 * num_models),
        constrained_layout=False,
        gridspec_kw={'hspace': 0.25}
    )

    if num_models == 1:
        axes = np.array([axes])

    for i, model in enumerate(models):
        vols = np.array(model_volumes[model])
        mses = np.array(model_mses[model]) * 1000
        Ds = np.array(D_dict[model])
        rhos = np.array(rho_dict[model])
        mse_col = np.array(mse_dict[model]) * 1000

        if not (vols.size and mses.size and Ds.size and rhos.size and mse_col.size):
            print(f"Skipping {model} due to missing data.")
            continue

        vmin = np.min(mse_col)
        vmax = np.max(mse_col)
        ticks = np.linspace(vmin, vmax, 5)
        tick_labels = [f"{t:.3f}" for t in ticks]

        ax1 = axes[i, 0]
        sc1 = ax1.scatter(vols, mses, c=mse_col, cmap='viridis', alpha=0.85, edgecolors='k')
        ax1.set_xlabel("GT Volume (x10³ voxels)", fontsize=22, labelpad=12)
        ax1.set_ylabel("MSE (×10⁻³)", fontsize=22, labelpad=12)
        ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:.1f}"))
        ax1.tick_params(axis='both', which='major', labelsize=20)
        ax1.grid(True)

        ax2 = axes[i, 1]
        sc2 = ax2.scatter(Ds, rhos, c=mse_col, cmap='viridis', alpha=0.85, edgecolors='k')
        ax2.set_xlabel("D", fontsize=22, labelpad=12)
        ax2.set_ylabel("ρ", fontsize=22, labelpad=12)
        ax2.tick_params(axis='both', which='major', labelsize=20)
        ax2.grid(True)

        # Individual colorbar
        cbar = fig.colorbar(sc2, ax=[ax1, ax2], location='right', pad=0.02, shrink=0.85)
        cbar.set_label("MSE (×10⁻³)", fontsize=22, labelpad=14)
        cbar.set_ticks(ticks)
        cbar.ax.set_yticklabels(tick_labels)
        cbar.ax.tick_params(labelsize=20, pad=10)

        # --- Align model name with center of row ---
        # Get middle y position of the row (relative to figure coordinates)
        pos = ax1.get_position()
        mid_y = (pos.y0 + pos.y1) / 2

        fig.text(
            0.03, mid_y,
            name_map.get(model, model),
            va='center',
            ha='left',
            fontsize=26,
            weight='bold'
        )

    plt.tight_layout(rect=[0.06, 0.03, 1, 0.98])  # leave space for left-side labels
    plt.show()

# -----------------------------
# CONFIGURATION (Update as needed)
# -----------------------------
DATASET_NAME = "Dataset900_Brain"
BASE_PATH = os.path.join(os.path.dirname(__file__), '..', 'performance_summaries')
PTH_BASE = "/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/preprocessed_data"

MODEL_EXPERIMENTS = {
    #'nnUnet': [['c', 'a_upsampling']],
    'TumorSurrogate': [['c', 'a_bottleneck_after']],
    'ViT': [['Linear', 'embed_concat']]
}

# -----------------------------
# MAIN
# -----------------------------
model_volumes, model_mses, D_dict, rho_dict, mse_dict = organize_data_per_model(
    DATASET_NAME, BASE_PATH, PTH_BASE, MODEL_EXPERIMENTS, MASKED=False
)
plot_model_specific(model_volumes, model_mses, D_dict, rho_dict, mse_dict)
# %%

import os
import time
import sys
# For different experiments
current_dir = os.path.dirname(os.path.abspath(__file__))  # scripts directory
src_path = os.path.abspath(os.path.join(current_dir, '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)
from TumorNetSolvers.inference.inference_utils import CustomDataset, get_settings_and_file_paths
from torch.utils.data import DataLoader
from scipy.ndimage import center_of_mass
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
def save_full_ground_truths(dataset_name: str, data_folder: str, output_base: str):
    """
    Iterate through test set and save full 3D ground truth tumor masks as NIfTI files.
    """
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

DATASET_NAME = "Dataset300_Brain"
nnUNet_preprocessed = os.environ.get('nnUNet_preprocessed')
DATA_FOLDER = os.path.join(nnUNet_preprocessed, DATASET_NAME,"nnUNetPlans_3d_fullres")
OUTPUT_BASE = '/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/preprocessed_data/Dataset300_Brain/gt'

save_full_ground_truths(DATASET_NAME, DATA_FOLDER, OUTPUT_BASE)
# %%
# Error vs volume with the corresponding best and worst selected and given IDs
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# -----------------------------
# Data Organizer
# -----------------------------
def organize_data_per_model(DATASET_NAME, BASE_PATH, PTH_BASE, MODEL_EXPERIMENTS, MASKED=False):
    from set_env import set_environment_variables
    set_environment_variables()
    nnUNet_preprocessed = os.environ.get('nnUNet_preprocessed')
    nnUNet_results = os.environ.get('nnUNet_results')

    if MASKED:
        GT_FOLDER = os.path.join(nnUNet_results, DATASET_NAME, 'masked_gt')
        if not os.path.exists(GT_FOLDER) or len(os.listdir(GT_FOLDER)) == 0:
            raise FileNotFoundError(f"Masked ground truth folder missing or empty: {GT_FOLDER}")
    else:
        GT_FOLDER = os.path.join(nnUNet_preprocessed, DATASET_NAME, 'nnUNetPlans_3d_fullres')

    model_volumes = {}
    model_mses = {}
    D_dict = {}
    rho_dict = {}
    mse_dict = {}

    for model, experiments in MODEL_EXPERIMENTS.items():
        model_volumes[model] = []
        model_mses[model] = []
        D_dict[model] = []
        rho_dict[model] = []
        mse_dict[model] = []

        for experiment in experiments:
            if model == "ViT":
                exp_folder = f'MODE_{experiment[1]}_METHOD_{experiment[0]}_best_FK_50'
            else:
                exp_folder = f'LOC_{experiment[1]}_MODE_{experiment[0]}_best_FK_50'

            if model == "TumorSurrogate":
                base_json_folder = os.path.join(BASE_PATH, DATASET_NAME, model, 'correctResNet', exp_folder)
                pth_path = os.path.join(PTH_BASE, DATASET_NAME, 'param_dict.pth')
            else:
                base_json_folder = os.path.join(BASE_PATH, DATASET_NAME, model, exp_folder)
                pth_path = os.path.join(PTH_BASE, DATASET_NAME, 'param_dict.pth')

            json_path = os.path.join(base_json_folder, f'evaluation_results_{model}_10k.json')

            if not os.path.exists(json_path):
                print(f"Missing JSON: {json_path}")
                continue

            try:
                pth_data = torch.load(pth_path, map_location="cpu")
            except Exception as e:
                print(f"Failed to load .pth: {pth_path} - {e}")
                continue

            with open(json_path, 'r') as f:
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

            for sample_id, metrics in sample_metrics.items():
                mse = metrics.get("aMSE", None)
                if mse is None:
                    continue

                patient_id = sample_id.replace('.npy', '')
                gt_filename = sample_id.replace('.npy', '_seg.npy')
                gt_path = os.path.join(GT_FOLDER, gt_filename)

                if not os.path.exists(gt_path) or patient_id not in pth_data:
                    continue

                gt_np = np.load(gt_path)
                gt_tensor = torch.tensor(gt_np)
                volume = torch.sum((gt_tensor > 0) * gt_tensor).item()

                tensor = pth_data[patient_id]
                if not isinstance(tensor, torch.Tensor) or tensor.numel() < 2:
                    continue

                D = tensor[-2].item()
                rho = tensor[-1].item()

                model_volumes[model].append(volume)
                model_mses[model].append(mse)
                D_dict[model].append(D)
                rho_dict[model].append(rho)
                mse_dict[model].append((patient_id, mse))

    return model_volumes, model_mses, D_dict, rho_dict, mse_dict

# -----------------------------
# Plotting with Highlighted Best/Worst + Patient ID Print
# -----------------------------
def plot_model_specific(model_volumes, model_mses, D_dict, rho_dict, mse_dict):
    for model in model_volumes:
        vols = np.array(model_volumes[model])
        mses = np.array([mse for _, mse in mse_dict[model]])
        Ds = np.array(D_dict[model])
        rhos = np.array(rho_dict[model])
        patient_ids = [pid for pid, _ in mse_dict[model]]

        if not (vols.size and mses.size and Ds.size and rhos.size):
            print(f"Skipping {model} due to missing data.")
            continue

        # Find best and worst
        worst_idx = np.argmax(mses)
        best_idx = np.argmin(mses)
        worst_patient = patient_ids[worst_idx]
        best_patient = patient_ids[best_idx]
        worst_mse = mses[worst_idx]
        best_mse = mses[best_idx]

        print(f"\n========== Model: {model} ==========")
        print(f"🟥 Worst MSE:  {worst_mse:.5f} — Patient ID: {worst_patient} (index {worst_idx})")
        print(f"🟩 Best  MSE:  {best_mse:.5f} — Patient ID: {best_patient} (index {best_idx})")

        # ---- Plotting ----
        fig = plt.figure(figsize=(20, 8))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])

        vmin = np.min(mses)
        vmax = np.max(mses)
        ticks = np.linspace(vmin, vmax, 5)
        tick_labels = [f"{t:.3f}" for t in ticks]

        # -------- First subplot: Volume vs MSE --------
        ax1 = fig.add_subplot(gs[0])
        sc1 = ax1.scatter(vols, mses, c=mses, cmap='viridis', alpha=0.8, edgecolors='k')
        ax1.set_xlabel("GT Volume (x10³ voxels)", fontsize=22, labelpad=10)
        ax1.set_ylabel("MSE", fontsize=22, labelpad=10)
        ax1.tick_params(axis='both', which='major', labelsize=20)
        ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:.1f}"))
        ax1.grid(True)

        # Highlight best and worst
        ax1.scatter(vols[worst_idx], mses[worst_idx], s=150, c='red', marker='X', label='Worst MSE')
        ax1.scatter(vols[best_idx], mses[best_idx], s=150, c='green', marker='P', label='Best MSE')
        ax1.legend(fontsize=16)

        # -------- Second subplot: rho vs D --------
        ax2 = fig.add_subplot(gs[1])
        sc2 = ax2.scatter(Ds, rhos, c=mses, cmap='viridis', alpha=0.8, edgecolors='k')
        ax2.set_xlabel("D", fontsize=22, labelpad=10)
        ax2.set_ylabel("rho", fontsize=22, labelpad=10)
        ax2.tick_params(axis='both', which='major', labelsize=20)
        ax2.grid(True)

        # Highlight best and worst
        ax2.scatter(Ds[worst_idx], rhos[worst_idx], s=150, c='red', marker='X', label='Worst MSE')
        ax2.scatter(Ds[best_idx], rhos[best_idx], s=150, c='green', marker='P', label='Best MSE')
        ax2.legend(fontsize=16)

        # -------- Shared Colorbar --------
        cbar = fig.colorbar(sc2, ax=ax2, location='right', pad=0.02)
        cbar.set_label("MSE", fontsize=22, labelpad=10)
        cbar.set_ticks(ticks)
        cbar.ax.set_yticklabels(tick_labels)
        cbar.ax.tick_params(labelsize=20, pad=8)

        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.show()

# -----------------------------
# CONFIGURATION
# -----------------------------
DATASET_NAME = "Dataset900_Brain"
BASE_PATH = os.path.join(os.path.dirname(__file__), '..', 'performance_summaries')
PTH_BASE = "/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/preprocessed_data"

MODEL_EXPERIMENTS = {
    'nnUnet': [['c', 'a_upsampling']],
    'TumorSurrogate': [['c', 'a_bottleneck_after']],
    'ViT': [['Linear', 'embed_concat']]
}

# -----------------------------
# MAIN EXECUTION
# -----------------------------
model_volumes, model_mses, D_dict, rho_dict, mse_dict = organize_data_per_model(
    DATASET_NAME, BASE_PATH, PTH_BASE, MODEL_EXPERIMENTS, MASKED=False
)
plot_model_specific(model_volumes, model_mses, D_dict, rho_dict, mse_dict)

print("\n================= Cross-Model Worst Patient Rankings =================")

# Convert mse_dict to sorted lists for ranking
sorted_mse = {}
for model in mse_dict:
    # Sort descending by MSE
    sorted_mse[model] = sorted(mse_dict[model], key=lambda x: -x[1])

# For each model, find where its worst patient ranks in the others
for model in mse_dict:
    worst_patient, worst_mse = sorted_mse[model][0]
    print(f"\n🔎 Worst patient in {model}: {worst_patient} (MSE: {worst_mse:.5f})")

    for other_model in mse_dict:
        if other_model == model:
            continue
        other_sorted = sorted_mse[other_model]
        # Find index of this patient in other model
        rank_in_other = next((i for i, (pid, _) in enumerate(other_sorted) if pid == worst_patient), None)
        if rank_in_other is not None:
            print(f"  ↳ Ranks #{rank_in_other + 1} worst in {other_model}")
        else:
            print(f"  ↳ Not found in {other_model}")


# %%
# Individual plots on difference
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import gridspec
from scipy.ndimage import center_of_mass

# ---------------------
# Configuration
# ---------------------
MODELS = ["nnUnet", "TumorSurrogate", "ViT"]
EXPERIMENTS = [["c", "a_upsampling"], ['c', 'a_bottleneck_after'], ['Linear', 'embed_concat']]
ARCH_LABELS = ["U-Net", "TS", "ViT"]

# You can specify a different patient ID per model here
PATIENT_IDS = {
    "nnUnet": "BRAIN_p3650",
    "TumorSurrogate": "BRAIN_p3650",
    "ViT": "BRAIN_p3650"
}

# ---------------------
# Compute global vmin/vmax for all difference slices
# ---------------------
diff_slices = []

for model, experiment in zip(MODELS, EXPERIMENTS):
    pid = PATIENT_IDS[model]

    # Load ground truth
    gt_true_path = f'/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/preprocessed_data/gt/{pid}/ground_truth_full.nii.gz'
    gt_true_all = nib.load(gt_true_path).get_fdata()

    # Load prediction
    if model == "TumorSurrogate":
        pred_path = f'/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset900_Brain/{model}/correctResNet/LOC_{experiment[1]}_MODE_{experiment[0]}_best_FK_50/preds/_{model}_10k/masked/{pid}.npy'
    elif model == "nnUnet":
        pred_path = f'/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset900_Brain/{model}/LOC_{experiment[1]}_MODE_{experiment[0]}_best_FK_50/preds/_{model}_10k/masked/{pid}.npy'
    else:
        pred_path = f'/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset900_Brain/{model}/MODE_{experiment[1]}_METHOD_{experiment[0]}_best_FK_50/preds/_{model}_10k/masked/{pid}.npy'

    gt_pred = np.load(pred_path)[0]
    z, _, _ = map(int, np.round(center_of_mass(gt_true_all)))
    diff = gt_pred - gt_true_all
    diff_slices.append(diff[:, :, z])

# Compute global color scale
all_diffs = np.stack(diff_slices)
vmax_global = np.ceil(np.max(np.abs(all_diffs)) * 10) / 10
vmin_global = -vmax_global

# ---------------------
# Plot per model
# ---------------------
titles = ["Prediction", "Ground Truth", "Difference (Pred - GT)"]
cmaps = ['Reds', 'Reds', 'bwr']

for model, experiment, arch_label in zip(MODELS, EXPERIMENTS, ARCH_LABELS):
    pid = PATIENT_IDS[model]

    print(f"\n--- Plotting {arch_label} for {pid} ---")

    # Load atlas
    atlas_path = f'/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/preprocessed_data/Dataset900_Brain/nnUNetPlans_3d_fullres/{pid}.npy'
    atlasTissue = np.load(atlas_path)[0]

    # Load ground truth
    gt_true_path = f'/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/preprocessed_data/gt/{pid}/ground_truth_full.nii.gz'
    gt_true_all = nib.load(gt_true_path).get_fdata()

    # Load prediction
    if model == "TumorSurrogate":
        pred_path = f'/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset900_Brain/{model}/correctResNet/LOC_{experiment[1]}_MODE_{experiment[0]}_best_FK_50/preds/_{model}_10k/masked/{pid}.npy'
    elif model == "nnUnet":
        pred_path = f'/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset900_Brain/{model}/LOC_{experiment[1]}_MODE_{experiment[0]}_best_FK_50/preds/_{model}_10k/masked/{pid}.npy'
    else:
        pred_path = f'/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset900_Brain/{model}/MODE_{experiment[1]}_METHOD_{experiment[0]}_best_FK_50/preds/_{model}_10k/masked/{pid}.npy'

    gt_pred_all = np.load(pred_path)[0]
    z, _, _ = map(int, np.round(center_of_mass(gt_true_all)))

    diff = gt_pred_all - gt_true_all
    overlays = [gt_pred_all, gt_true_all, diff]

    # ---- Plot ----
    fig = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.02)

    for col_idx, (overlay, cmap, title) in enumerate(zip(overlays, cmaps, titles)):
        ax = fig.add_subplot(gs[0, col_idx])

        # Rotate both atlas and overlay
        rotated_atlas = np.rot90(atlasTissue[:, :, z], 2)
        rotated_overlay = np.rot90(overlay[:, :, z], 2)

        ax.imshow(rotated_atlas, cmap='gray', vmin=0, vmax=1, interpolation='none')

        if col_idx == 2:  # Difference
            masked = np.ma.masked_where(np.abs(rotated_overlay) < 1e-6, rotated_overlay)
            im = ax.imshow(masked, cmap=cmap, vmin=vmin_global, vmax=vmax_global)
        else:
            ax.imshow(rotated_overlay, cmap=cmap, alpha=0.9 * rotated_overlay)

        ax.axis('off')

        # Add black border
        border = Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                           linewidth=2, edgecolor='black', facecolor='none', zorder=10, clip_on=False)
        ax.add_patch(border)

        # Add column title
        ax.set_title(title, fontsize=20, weight='bold', pad=12)

        # Add left-side architecture label on the first column only
        if col_idx == 0:
            ax.text(-0.18, 0.5, arch_label, fontsize=22, weight='bold',
                    va='center', ha='right', transform=ax.transAxes)

    # Colorbar
    cax = fig.add_subplot(gs[0, 3])
    cbar = fig.colorbar(im, cax=cax)

    mid_min = (vmin_global + 0) / 2
    mid_max = (0 + vmax_global) / 2
    ticks = [vmin_global, mid_min, 0, mid_max, vmax_global]
    cbar.set_ticks(ticks)
    cbar.ax.set_yticklabels([f"{vmin_global:.2f}", f"{mid_min:.2f}", "0", f"{mid_max:.2f}", f"{vmax_global:.2f}"])
    cbar.ax.tick_params(labelsize=18, direction='out', length=6, top=True, bottom=True)

    plt.tight_layout()
    plt.show()


# %%
# Specific parameter variation comparison
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass

# === Configuration ===
PATIENT_ID = "p875"  # <<<<<<< Change this value only to switch patient
MODEL = "nnUnet"
EXPERIMENT = ['c', 'a_upsampling']
DATASET_NAME = "Dataset900_Brain"

# Paths
nnUNet_preprocessed = "/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/preprocessed_data"
nnUNet_results = "/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results"
DATA_FOLDER = os.path.join(nnUNet_preprocessed, DATASET_NAME, "nnUNetPlans_3d_fullres")

# Load background tissue
atlas_path = os.path.join(DATA_FOLDER, f"BRAIN_{PATIENT_ID}.npy")
atlasTissue = np.load(atlas_path)[0]

# Load ground truth and compute center of mass
gt_path = f"/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/preprocessed_data/gt/BRAIN_{PATIENT_ID}/ground_truth_full.nii.gz"
gt_data = nib.load(gt_path).get_fdata()
z, y, x = map(int, np.round(center_of_mass(gt_data)))

# Coefficient grid
COEFFICIENTS = [
    [0, 0], [0, 0.25], [0, 0.5], [0, 0.75], [0, 1],
    [0.25, 0], [0.25, 0.25], [0.25, 0.5], [0.25, 0.75], [0.25, 1],
    [0.5, 0], [0.5, 0.25], [0.5, 0.5], [0.5, 0.75], [0.5, 1],
    [0.75, 0], [0.75, 0.25], [0.75, 0.5], [0.75, 0.75], [0.75, 1],
    [1, 0], [1, 0.25], [1, 0.5], [1, 0.75], [1, 1]
]

# === Utility functions ===
# Utility functions
def format_label(val):
    return f"{int(val * 100)}%"

def coeff_str(val):
    if val in [0, 1]:
        return str(int(val))
    elif val == 0.5:
        return "0.5"
    else:
        return f"{val:.2f}"

# Plotting
fig, axes = plt.subplots(5, 5, figsize=(20, 20))
fig.subplots_adjust(wspace=0.02, hspace=0.02)

for idx, (d_val, rho_val) in enumerate(COEFFICIENTS):
    row = idx // 5
    col = idx % 5

    d_label = coeff_str(d_val)
    rho_label = coeff_str(rho_val)

    pred_path = os.path.join(
        nnUNet_results, DATASET_NAME, MODEL, "coefficient",
        f"LOC_{EXPERIMENT[1]}_MODE_{EXPERIMENT[0]}_best_FK_50_D_{d_label}_rho_{rho_label}",
        "preds", f"_{MODEL}_10k", "masked", f"BRAIN_{PATIENT_ID}.npy"
    )

    ax = axes[row, col]

    if os.path.exists(pred_path):
        data = np.load(pred_path)[0]
        slice_overlay = np.rot90(data[:, :, z], 2)
    else:
        print(f"Missing file: {pred_path}")
        slice_overlay = np.zeros_like(atlasTissue[:, :, z])

    slice_atlas = np.rot90(atlasTissue[:, :, z], 2)
    ax.imshow(slice_atlas, cmap='gray')
    ax.imshow(slice_overlay, cmap='Reds', alpha=0.9 * slice_overlay, interpolation='none')
    ax.axis('off')

    # Add column headers (top row only)
    if row == 0:
        ax.set_title(f"ρ = {format_label(rho_val)}", fontsize=26, fontweight='bold', pad=15)

    # Add row labels (leftmost column only)
    if col == 0:
        ax.annotate(f"D = {format_label(d_val)}", xy=(-0.25, 0.5), xycoords='axes fraction',
                    fontsize=26, fontweight='bold', rotation=90,
                    ha='center', va='center')

plt.tight_layout()
plt.show()

# %%
# Several patients in figure of coefficients percentage
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
import os

# Configuration
MODEL = "nnUnet"
EXPERIMENTS = [['c', 'a_upsampling']]
PATIENT_IDS = ["BRAIN_p875", "BRAIN_p9071", "BRAIN_p3650"]  # Add more patient IDs as needed

# Define slicing plane per patient
# Options: "axial", "coronal", "sagittal"
PLANE_PER_PATIENT = {
    "BRAIN_p875": "axial",
    "BRAIN_p9071": "coronal",
    "BRAIN_p3650": "sagittal"
}

# Format plot label
def format_label(label):
    if label == 'gt_coeff':
        return 'GT coeff'
    if label.endswith('_coeff'):
        coeff = float(label.replace('_coeff', ''))
        return f"{int(coeff * 100)}% coeff"
    return label

# Extract slice based on plane
def get_slice(data, plane, com):
    z, y, x = com
    if plane == 'axial':
        return np.rot90(data[:, :, z], 2)
    elif plane == 'coronal':
        return np.rot90(data[:, y, :], 2)
    elif plane == 'sagittal':
        return np.rot90(data[x, :, :], 2)
    else:
        raise ValueError(f"Unsupported plane: {plane}")

# Start plot
fig, axes = plt.subplots(len(PATIENT_IDS), 6, figsize=(24, 5 * len(PATIENT_IDS)))

if len(PATIENT_IDS) == 1:
    axes = np.expand_dims(axes, axis=0)

for row_idx, PATIENT_ID in enumerate(PATIENT_IDS):
    plane = PLANE_PER_PATIENT.get(PATIENT_ID, "axial")

    # Load tissue background
    atlasTissue = np.load(
        f'/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/preprocessed_data/Dataset900_Brain/nnUNetPlans_3d_fullres/{PATIENT_ID}.npy'
    )[0]

    for experiment in EXPERIMENTS:
        if MODEL != "ViT":
            paths = {
                'p': [
                    ('0_coeff', f'/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset900_Brain/{MODEL}/LOC_{experiment[1]}_MODE_{experiment[0]}_best_FK_50_0/preds/_{MODEL}_10k/masked/{PATIENT_ID}.npy'),
                    ('0.2_coeff', f'/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset900_Brain/{MODEL}/LOC_{experiment[1]}_MODE_{experiment[0]}_best_FK_50_0.2/preds/_{MODEL}_10k/masked/{PATIENT_ID}.npy'),
                    ('0.4_coeff', f'/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset900_Brain/{MODEL}/LOC_{experiment[1]}_MODE_{experiment[0]}_best_FK_50_0.4/preds/_{MODEL}_10k/masked/{PATIENT_ID}.npy'),
                    ('0.6_coeff', f'/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset900_Brain/{MODEL}/LOC_{experiment[1]}_MODE_{experiment[0]}_best_FK_50_0.6/preds/_{MODEL}_10k/masked/{PATIENT_ID}.npy'),
                    ('0.8_coeff', f'/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset900_Brain/{MODEL}/LOC_{experiment[1]}_MODE_{experiment[0]}_best_FK_50_0.8/preds/_{MODEL}_10k/masked/{PATIENT_ID}.npy'),
                    ('gt_coeff', f'/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results/Dataset900_Brain/{MODEL}/LOC_{experiment[1]}_MODE_{experiment[0]}_best_FK_50_1/preds/_{MODEL}_10k/masked/{PATIENT_ID}.npy'),
                    ('ground_truth_image', f'/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/preprocessed_data/gt/{PATIENT_ID}/ground_truth_full.nii.gz'),
                ]
            }
        else:
            raise NotImplementedError("ViT case not yet implemented in this version.")

        patient_paths = dict(paths['p'])
        gt_data = nib.load(patient_paths['ground_truth_image']).get_fdata()
        com = tuple(map(int, np.round(center_of_mass(gt_data))))

        for col_idx, (label, path) in enumerate(paths['p'][:6]):
            ax = axes[row_idx, col_idx]

            if not os.path.exists(path):
                ax.set_title("Missing file", fontsize=16)
                ax.axis('off')
                continue

            data = np.load(path)[0]
            tissue_slice = get_slice(atlasTissue, plane, com)
            data_slice = get_slice(data, plane, com)

            ax.imshow(tissue_slice, cmap='gray')
            ax.imshow(data_slice, cmap='Reds', alpha=0.9 * data_slice, interpolation='none')

            if row_idx == 0:
                ax.set_title(format_label(label), fontsize=32)

            ax.axis('off')

plt.tight_layout()
plt.show()
# %%

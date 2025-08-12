import os
import sys
# For different experiments
current_dir = os.path.dirname(os.path.abspath(__file__))  # scripts directory
src_path = os.path.abspath(os.path.join(current_dir, '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import gridspec
from matplotlib.ticker import FuncFormatter
from collections import defaultdict
from matplotlib.lines import Line2D
from scipy.ndimage import center_of_mass
import os
import json
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from scripts.set_env import set_environment_variables

from utils import save_full_ground_truths

def plt_coeffs(nnUNet_results, patient_ids, plane_per_patient, model_experiments, dataset, signature, coefficients, ending, gt_base_path):

    '''# Configuration
MODEL = "nnUnet"
EXPERIMENTS = [['c', 'a_upsampling']]
PATIENT_IDS = ["BRAIN_p875", "BRAIN_p9071", "BRAIN_p3650"]  # Add more patient IDs as needed

# Define slicing plane per patient
# Options: "axial", "coronal", "sagittal"
PLANE_PER_PATIENT = {
    "BRAIN_p875": "axial",
    "BRAIN_p9071": "coronal",
    "BRAIN_p3650": "sagittal"
}'''


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
    fig, axes = plt.subplots(len(patient_ids), len(coefficients), figsize=(24, 5 * len(patient_ids)))

    if len(patient_ids) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, patient_id in enumerate(patient_ids):
        plane = plane_per_patient.get(patient_id, "axial")

        # Load tissue background
        atlasTissue = np.load(
            f'/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/preprocessed_data/{dataset}/nnUNetPlans_3d_fullres/{patient_id}.npy'
        )[0]

        for model in model_experiments:
            for experiment in model_experiments[model]:
                paths_list = []
                for coefficient in coefficients:
                    if model != "ViT":
                        if coefficient[0] == 1:
                            paths_list.append(('gt_coeff', os.path.join(nnUNet_results, dataset, f"{model}_{signature}", f"LOC_{experiment[1]}_MODE_{experiment[0]}{'_'+ending if ending is not None else ''}_coeffs_{coefficient[0]}", 'preds', 'masked', f'{patient_id}.npy')))
                        else:
                            paths_list.append((f'{coefficient[0]}_coeff', os.path.join(nnUNet_results, dataset, f"{model}_{signature}", f"LOC_{experiment[1]}_MODE_{experiment[0]}{'_'+ending if ending is not None else ''}_coeffs_{coefficient[0]}", 'preds', f'_{model}_10k', 'masked', f'{patient_id}.npy')))
                    else:
                        if coefficient[0] == 1:
                            paths_list.append(('gt_coeff', os.path.join(nnUNet_results, dataset, f"{model}_{signature}", f"MODE_{experiment[1]}_METHOD_{experiment[0]}{'_'+ending if ending is not None else ''}_coeffs_{coefficient[0]}", 'preds', 'masked', f'{patient_id}.npy')))
                        else:
                            paths_list.append((f'{coefficient[0]}_coeff', os.path.join(nnUNet_results, dataset, f"{model}_{signature}", f"MODE_{experiment[1]}_METHOD_{experiment[0]}{'_'+ending if ending is not None else ''}_coeffs_{coefficient[0]}", 'preds', 'masked', f'{patient_id}.npy')))

                gt_path = f'/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/preprocessed_data/gt/{patient_id}/ground_truth_full.nii.gz'
                if not os.path.exists(gt_path):
                    save_full_ground_truths(gt_base_path, dataset)
                if not os.path.exists(gt_path):
                    raise FileNotFoundError(f"Path does not exist: {gt_path}")
                
                paths_list.append(('ground_truth_image', gt_path))
                paths = {'p': paths_list}
                    
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
                        ax.set_title(format_label(label), fontsize=40)

                    ax.axis('off')

    #plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.00000001, left=0.02, right=0.98, top=0.95, bottom=0.05)
    plt.show()

def plt_diff(nnUNet_results, models, experiments, patient_id, arch_labels, dataset, signature, ending, gt_base_path):

    '''
    MODELS = ["nnUnet", "TumorSurrogate", "ViT"]
EXPERIMENTS = [["c", "a_upsampling"], ['c', 'a_bottleneck_after'], ['Linear', 'embed_concat']]
ARCH_LABELS = ["U-Net", "TS", "ViT"]
PATIENT_ID = "BRAIN_p875" #1973#1911#12726'''

    # Load background tissue (atlas) - original full atlas (no mask)
    atlasTissue_209 = np.load(
        f'/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/preprocessed_data/{dataset}/nnUNetPlans_3d_fullres/{patient_id}.npy'
    )[0]

    # --- Step 1: Determine global vmin/vmax for difference plots ---
    diff_slices = []

    for model, experiment in zip(models, experiments):
        if model != "ViT":
            path = os.path.join(nnUNet_results, dataset, f"{model}_{signature}", f"LOC_{experiment[1]}_MODE_{experiment[0]}{'_'+ending if ending is not None else ''}", 'preds', 'masked', f'{patient_id}.npy')
        else:
            path = os.path.join(nnUNet_results, dataset, f"{model}_{signature}", f"MODE_{experiment[1]}_METHOD_{experiment[0]}{'_'+ending if ending is not None else ''}", 'preds', 'masked', f'{patient_id}.npy')

        gt_true_path = f'/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/preprocessed_data/gt/{patient_id}/ground_truth_full.nii.gz'
        if not os.path.exists(gt_true_path):
            save_full_ground_truths(gt_base_path, dataset)
        if not os.path.exists(gt_true_path):
            raise FileNotFoundError(f"Path does not exist: {gt_true_path}")

        gt_true = nib.load(gt_true_path).get_fdata()
        gt_pred = np.load(path)[0]
        z, _, _ = map(int, np.round(center_of_mass(gt_true)))
        diff = gt_pred - gt_true
        diff_slices.append(diff[:, :, z])

    #all_diffs = np.stack(diff_slices)
    vmax_global = 0.9#np.ceil(np.max(np.abs(all_diffs)) * 10) / 10
    vmin_global = -0.9#-vmax_global

    # --- Step 2: Plot with fixed layout using GridSpec ---
    fig = plt.figure(figsize=(16, 5 * len(models)))
    gs = gridspec.GridSpec(len(models), 4, width_ratios=[1, 1, 1, 0.05], wspace=0.02, hspace=0.25)

    titles = ["Prediction", "Ground Truth", "Difference (Pred - GT)"]
    cmaps = ['Reds', 'Reds', 'bwr']

    for row_idx, (model, experiment, arch_label) in enumerate(zip(models, experiments, arch_labels)):
        if model != "ViT":
            path = os.path.join(nnUNet_results, dataset, f"{model}_{signature}", f"LOC_{experiment[1]}_MODE_{experiment[0]}{'_'+ending if ending is not None else ''}", 'preds', 'masked', f'{patient_id}.npy')
        else:
            path = os.path.join(nnUNet_results, dataset, f"{model}_{signature}", f"MODE_{experiment[1]}_METHOD_{experiment[0]}{'_'+ending if ending is not None else ''}", 'preds', 'masked', f'{patient_id}.npy')

        gt_pred = np.load(path)[0]

        gt_true_path = f'/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/preprocessed_data/gt/{patient_id}/ground_truth_full.nii.gz'
        if not os.path.exists(gt_true_path):
            save_full_ground_truths(gt_base_path, dataset)
        if not os.path.exists(gt_true_path):
            raise FileNotFoundError(f"Path does not exist: {gt_true_path}")
        
        gt_true = nib.load(gt_true_path).get_fdata()
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
                im = ax.imshow(masked, cmap=cmap, vmin=-1, vmax=1)
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
        ticks = [-1, -0.5, 0, 0.5, 1]

        # Set ticks and labels
        cbar.set_ticks(ticks)
        cbar.ax.set_yticklabels(["-1", "-0.5", "0", "0.5", f"1"])

        # Show ticks and labels on both top and bottom
        cbar.ax.yaxis.set_ticks_position('both')
        cbar.ax.tick_params(labelsize=22, direction='out', length=6, top=True, bottom=True)

    plt.show()

def plt_diff_individual(nnUNet_results, models, experiments, patient_ids, arch_labels, dataset, signature, ending, gt_base_path):


    '''# ---------------------
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
}'''

    # ---------------------
    # Compute global vmin/vmax for all difference slices
    # ---------------------
    diff_slices = []

    for model, experiment in zip(models, experiments):
        pid = patient_ids[model]

        # Load ground truth
        gt_true_path = f'/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/preprocessed_data/gt/{pid}/ground_truth_full.nii.gz'
        if not os.path.exists(gt_true_path):
            save_full_ground_truths(gt_base_path, dataset)
        if not os.path.exists(gt_true_path):
            raise FileNotFoundError(f"Path does not exist: {gt_true_path}")
        gt_true_all = nib.load(gt_true_path).get_fdata()

        # Load prediction
        if model != "ViT":
            pred_path = os.path.join(nnUNet_results, dataset, f"{model}_{signature}", f"LOC_{experiment[1]}_MODE_{experiment[0]}{'_'+ending if ending is not None else ''}", 'preds', 'masked', f'{pid}.npy')
        else:
            pred_path = os.path.join(nnUNet_results, dataset, f"{model}_{signature}", f"MODE_{experiment[1]}_METHOD_{experiment[0]}{'_'+ending if ending is not None else ''}", 'preds', 'masked', f'{pid}.npy')

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

    for model, experiment, arch_label in zip(models, experiments, arch_labels):
        pid = patient_ids[model]

        print(f"\n--- Plotting {arch_label} for {pid} ---")

        # Load atlas
        atlas_path = f'/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/preprocessed_data/Dataset900_Brain/nnUNetPlans_3d_fullres/{pid}.npy'
        atlasTissue = np.load(atlas_path)[0]

        # Load ground truth
        gt_true_path = f'/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/preprocessed_data/gt/{pid}/ground_truth_full.nii.gz'
        if not os.path.exists(gt_true_path):
            save_full_ground_truths(gt_base_path, dataset)
        if not os.path.exists(gt_true_path):
            raise FileNotFoundError(f"Path does not exist: {gt_true_path}")
        gt_true_all = nib.load(gt_true_path).get_fdata()

        # Load prediction
        if model != "ViT":
            pred_path = os.path.join(nnUNet_results, dataset, f"{model}_{signature}", f"LOC_{experiment[1]}_MODE_{experiment[0]}{'_'+ending if ending is not None else ''}", 'preds', 'masked', f'{pid}.npy')
        else:
            pred_path = os.path.join(nnUNet_results, dataset, f"{model}_{signature}", f"MODE_{experiment[1]}_METHOD_{experiment[0]}{'_'+ending if ending is not None else ''}", 'preds', 'masked', f'{pid}.npy')

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

def plt_dice_scores(model_experiments, dataset, ending, signature):

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
MODEL_EXPERIMENTS = {

    'nnUnet': [
        ['c', 'a_upsampling']
    ],
    'TumorSurrogate': [
        ['c', 'a_bottleneck_after']
    ],
    'ViT': [
        ['Linear', 'embed_concat']
    ]
}

DATASET_NAME = "Dataset300_Brain"'''

    base_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'performance_summaries')

    # Load data
    all_model_experiment_metrics = {}
    all_dice_means = []

    for model, experiments in model_experiments.items():
        experiment_metrics = {}
        for method, mode in experiments:
            if model == "ViT":
                exp_folder = f"MODE_{mode}_METHOD_{method}{'_'+ending if ending is not None else ''}"
            else:
                exp_folder = f"LOC_{mode}_MODE_{method}{'_'+ending if ending is not None else ''}"

            json_folder = os.path.join(base_path, dataset, f"{model}_{signature}", exp_folder)
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
    y_min = 0.5#max(0.0, min(all_dice_means) * 0.95)
    y_max = 1#min(1.0, max(all_dice_means) * 1.05)

    # Create a color palette from matplotlib for consistent colors across plots

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

def plt_error_volume_rho_D(dataset, model_experiments, ending, signature):

    base_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'performance_summaries')
    pth_base = os.getenv('nnUNet_preprocessed')
    '''
    DATASET_NAME = "Dataset900_Brain"

MODEL_EXPERIMENTS = {
    #'nnUnet': [['c', 'a_upsampling']],
    'TumorSurrogate': [['c', 'a_bottleneck_after']],
    'ViT': [['Linear', 'embed_concat']]
}'''

    def organize_data_per_model(DATASET_NAME, BASE_PATH, PTH_BASE, MODEL_EXPERIMENTS, MASKED=False):
        
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
                    exp_folder = f"MODE_{experiment[1]}_METHOD_{experiment[0]}{'_'+ending if ending is not None else ''}"
                else:
                    exp_folder = f"LOC_{experiment[1]}_MODE_{experiment[0]}{'_'+ending if ending is not None else ''}"


                base_json_folder = os.path.join(BASE_PATH, DATASET_NAME, f"{model}_{signature}", exp_folder)
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
            if model == "TumorSurrogate":
                vmin = 0  # Ensure colorbar starts at 0
                vmax = 6  # You can adjust this if needed
                ticks = np.linspace(vmin, vmax, 5)
            elif model == "ViT":
                vmin = np.min(mse_col)
                vmax = 16
                ticks = np.linspace(vmin, vmax, 5)
            else:
                vmin = np.min(mse_col)
                vmax = np.max(mse_col)
                ticks = np.linspace(vmin, vmax, 5)
            tick_labels = [f"{t:.1f}" for t in ticks]

            ax1 = axes[i, 0]
            sc1 = ax1.scatter(vols, mses, c=mse_col, cmap='viridis', alpha=0.85, edgecolors='k', vmin=vmin, vmax=vmax)
            ax1.set_xlabel("GT Volume (x10³ voxels)", fontsize=22, labelpad=12)
            ax1.set_ylabel("MSE (x10⁻³)", fontsize=22, labelpad=12)
            ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:.1f}"))
            ax1.tick_params(axis='both', which='major', labelsize=20)
            ax1.grid(True)

            ax2 = axes[i, 1]
            sc2 = ax2.scatter(Ds, rhos, c=mse_col, cmap='viridis', alpha=0.85, edgecolors='k', vmin=vmin, vmax=vmax)
            ax2.set_xlabel("D", fontsize=22, labelpad=12)
            ax2.set_ylabel("ρ", fontsize=22, labelpad=12)
            ax2.tick_params(axis='both', which='major', labelsize=20)
            ax2.grid(True)

            # Individual colorbar
            cbar = fig.colorbar(sc2, ax=[ax1, ax2], location='right', pad=0.02, shrink=0.85)
            cbar.set_label("MSE (x10⁻³)", fontsize=22, labelpad=14)
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

    ###### Main calls for functions ######

    model_volumes, model_mses, D_dict, rho_dict, mse_dict = organize_data_per_model(
        dataset, base_path, pth_base, model_experiments, MASKED=False
    )
    plot_model_specific(model_volumes, model_mses, D_dict, rho_dict, mse_dict)

def plt_rho_D_comp(nnUNet_results, coefficients, dataset, model_experiments, patient_id, ending, signature, gt_base_path):

    '''PATIENT_ID = "p875"  # <<<<<<< Change this value only to switch patient
MODEL = "nnUnet"
EXPERIMENT = ['c', 'a_upsampling']
DATASET_NAME = "Dataset900_Brain"

# Paths
nnUNet_preprocessed = "/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/preprocessed_data"
nnUNet_results = "/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/results"
DATA_FOLDER = os.path.join(nnUNet_preprocessed, DATASET_NAME, "nnUNetPlans_3d_fullres")



# Coefficient grid
COEFFICIENTS = [
    [0, 0], [0, 0.25], [0, 0.5], [0, 0.75], [0, 1],
    [0.25, 0], [0.25, 0.25], [0.25, 0.5], [0.25, 0.75], [0.25, 1],
    [0.5, 0], [0.5, 0.25], [0.5, 0.5], [0.5, 0.75], [0.5, 1],
    [0.75, 0], [0.75, 0.25], [0.75, 0.5], [0.75, 0.75], [0.75, 1],
    [1, 0], [1, 0.25], [1, 0.5], [1, 0.75], [1, 1]
]'''

    nnUNet_preprocessed = os.environ.get('nnUNet_preprocessed')

    DATA_FOLDER = os.path.join(nnUNet_preprocessed, dataset, "nnUNetPlans_3d_fullres")

    # Load background tissue
    atlas_path = os.path.join(DATA_FOLDER, f"BRAIN_{patient_id}.npy")
    atlasTissue = np.load(atlas_path)[0]

    # Load ground truth and compute center of mass
    gt_path = f"/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/preprocessed_data/gt/BRAIN_{patient_id}/ground_truth_full.nii.gz"
    if not os.path.exists(gt_path):
        save_full_ground_truths(gt_base_path, dataset)
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Path does not exist: {gt_path}")
    gt_data = nib.load(gt_path).get_fdata()
    z, y, x = map(int, np.round(center_of_mass(gt_data)))
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

    for model in model_experiments:
        for experiment in model_experiments[model]:

            # Plotting
            fig, axes = plt.subplots(5, 5, figsize=(20, 20))
            fig.subplots_adjust(wspace=0.02, hspace=0.02)

            for idx, (d_val, rho_val) in enumerate(coefficients):
                row = idx // 5
                col = idx % 5

                d_label = coeff_str(d_val)
                rho_label = coeff_str(rho_val)

                pred_path = os.path.join(
                    nnUNet_results, dataset, f"{model}_{signature}",
                    f"LOC_{experiment[1]}_MODE_{experiment[0]}{'_'+ending if ending is not None else ''}_rho_{rho_label}_D_{d_label}",
                    "preds", "masked", f"BRAIN_{patient_id}.npy"
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
                    ax.set_title(f"ρ = {format_label(rho_val)}", fontsize=32, fontweight='bold', pad=15)

                # Add row labels (leftmost column only)
                if col == 0:
                    ax.annotate(f"D = {format_label(d_val)}", xy=(-0.25, 0.5), xycoords='axes fraction',
                                fontsize=32, fontweight='bold', rotation=90,
                                ha='center', va='center')

            plt.tight_layout()
            plt.show()





import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os
import sys
# For different experiments
current_dir = os.path.dirname(os.path.abspath(__file__))  # scripts directory
src_path = os.path.abspath(os.path.join(current_dir, '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)
import json
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from scripts.set_env import set_environment_variables

def mul_histo_comp(experiments, model, dataset):

    '''EXPERIMENTS = [ ['c', 'a_downsampling'], ['a', 'a_downsampling'],
                    ['c', 'b_downsampling'], ['a', 'b_downsampling'],
                    ['c', 'inputs'], ['a', 'inputs'],
                    ['c', 'b_bottleneck'], ['a', 'b_bottleneck'],
                    ['c', 'a_bottleneck'], ['a', 'a_bottleneck']]

MODEL = ['nnUnet']
DATASET_NAME = "Dataset900_Brain"'''

    # List of JSON files
    json_files = [
        f'evaluation_results_{model}_10k.json',
        f'output_summary_{model}_10k.json',
    ]
    # This dictionary will collect all sample_metrics per experiment key
    experiment_metrics = {}

    for experiment in experiments:
        if model != "ViT":
            json_folder = os.path.join(os.path.dirname(__file__), '..', 'performance_summaries', dataset , model, f'LOC_{experiment[1]}_MODE_{experiment[0]}')
        else:
            json_folder = os.path.join(os.path.dirname(__file__), '..', 'performance_summaries', dataset , model, f'MODE_{experiment[1]}_METHOD_{experiment[0]}')
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
        sample_metrics = all_data[f'evaluation_results_{model}_10k.json']

        # Store sample_metrics by experiment key
        if model != "ViT":
            experiment_key = f"LOC_{experiment[1]}_MODE_{experiment[0]}"
        else:
            experiment_key = f"MODE_{experiment[1]}_METHOD_{experiment[0]}"
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

def worst_best_error(model_experiments, dataset):

    base_path = os.path.join(os.path.dirname(__file__), '..', 'performance_summaries')
    pth_base = os.getenv('nnUNet_preprocessed')
    '''
    DATASET_NAME = "Dataset900_Brain"

MODEL_EXPERIMENTS = {
    #'nnUnet': [['c', 'a_upsampling']],
    'TumorSurrogate': [['c', 'a_bottleneck_after']],
    'ViT': [['Linear', 'embed_concat']]
}'''

    # -----------------------------
    # Data Organizer
    # -----------------------------
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
            print(f" Worst MSE:  {worst_mse:.5f} — Patient ID: {worst_patient} (index {worst_idx})")
            print(f" Best  MSE:  {best_mse:.5f} — Patient ID: {best_patient} (index {best_idx})")

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
    # MAIN EXECUTION
    # -----------------------------
    model_volumes, model_mses, D_dict, rho_dict, mse_dict = organize_data_per_model(
        dataset, base_path, pth_base, model_experiments, MASKED=False
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
        print(f"\n Worst patient in {model}: {worst_patient} (MSE: {worst_mse:.5f})")

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









import json
import numpy as np
from scipy.stats import iqr, sem

def compute_statistics_with_extremes(json_file, output_file):
    """
    Analyzes evaluation metrics from a JSON file, computes summary statistics, and identifies extreme values (highest/lowest).

    Args:
        json_file (str): Path to the input JSON file containing evaluation metrics (typically the output of running compute_metrics)
        
        output_file (str): Path to the output JSON file where the summary statistics (mean, median, SE, IQR) and extreme values 
                           (highest/lowest Dice and MSE) will be saved.

    Returns:
        dict: A dictionary containing the computed statistics and extreme values, structured similarly to the output file.
    """
    
    with open(json_file, 'r') as f:
        data = json.load(f)["sample_metrics"]
    
    metrics = {"aMSE": [], "MAE": [], "SSIM": []}
    dice_scores = {f"Dice_{i/10}": [] for i in range(1, 10)}
    mse_extremes = {"lowest_mse_file": ("", float("inf")), "highest_mse_file": ("", float("-inf"))}
    dice_extremes = {f"Dice_{i/10}": {"highest": ("", 0), "lowest": ("", 1)} for i in range(1, 10)}

    for key, values in data.items():
        metrics["aMSE"].append(values["aMSE"])
        metrics["MAE"].append(values["MAE"])
        metrics["SSIM"].append(values["SSIM"])

        if values["aMSE"] < mse_extremes["lowest_mse_file"][1]:
            mse_extremes["lowest_mse_file"] = (key, values["aMSE"])
        if values["aMSE"] > mse_extremes["highest_mse_file"][1]:
            mse_extremes["highest_mse_file"] = (key, values["aMSE"])

        for dice_key in dice_scores.keys():
            dice_value = values[dice_key]
            dice_scores[dice_key].append(dice_value)
            if dice_value > dice_extremes[dice_key]["highest"][1]:
                dice_extremes[dice_key]["highest"] = (key, dice_value)
            if dice_value < dice_extremes[dice_key]["lowest"][1]:
                dice_extremes[dice_key]["lowest"] = (key, dice_value)

    summary = {}
    for metric, values in metrics.items():
        summary[metric] = {
            "Mean ± SE": f"{np.mean(values):.8f} ± {sem(values):.8f}",
            "Median ± IQR": f"{np.median(values):.8f} ± {iqr(values):.8f}"
        }

    for dice_key, values in dice_scores.items():
        summary[dice_key] = {
            "Mean ± SE": f"{np.mean(values):.8f} ± {sem(values):.8f}",
            "Median ± IQR": f"{np.median(values):.8f} ± {iqr(values):.8f}",
            "Highest Dice Sample": dice_extremes[dice_key]["highest"],
            "Lowest Dice Sample": dice_extremes[dice_key]["lowest"]
        }

    summary["MSE_Extremes"] = mse_extremes

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=4)

    return summary

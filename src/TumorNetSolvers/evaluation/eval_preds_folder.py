import os
import numpy as np
import torch
from TumorNetSolvers.utils.metrics import compute_dice_score, compute_ssim


def compute_metrics(preds_folder, gt_folder):
    """
    Computes evaluation metrics for predictions and ground truths stored as .npy files in 
    two separate folders
    """
    filter= lambda x: (x>0)*x
    sample_metrics = {}
    for key in os.listdir(preds_folder):
        if key.endswith('.npy'):
            pred = torch.tensor(np.load(os.path.join(preds_folder, key)))
            gt = filter(torch.tensor(np.load(os.path.join(gt_folder, key.replace('.npy', '_seg.npy')))))
            
            # Compute individual metrics
            aMSE_value = torch.mean((pred - gt) ** 2).tolist()
            MAE_value = torch.mean(np.abs(pred - gt)).tolist()
            ssim_value = compute_ssim(gt, pred).tolist()

            # Compute Dice scores for various thresholds
            dice_scores = {}
            for threshold in np.arange(0.1, 1.0, 0.1):
                dice_scores[f'Dice_{threshold:.1f}'] = compute_dice_score(pred, gt, threshold).tolist()

            # Store all metrics for this sample
            sample_metrics[key] = {
                "aMSE": aMSE_value,
                "MAE": MAE_value,
                "SSIM": ssim_value,
                **dice_scores
            }

    return sample_metrics

import torch
import math
import numpy as np
from pathlib import Path
from kornia.metrics.ssim3d import SSIM3D
from kornia.metrics.ssim import SSIM


# Utility classes
class EMA:
    """ Exponentially Weighted Moving Average (EMA) """
    
    def __init__(self, alpha=0.1):
        """
        Initializes the EMA with a smoothing factor.
        Args:
            alpha (float): Smoothing factor (default is 0.1).
        """
        self.alpha = alpha
        self.value = None

    def update(self, new_value):
        """
        Updates the EMA with a new value.
        Args:
            new_value (float): New value to update the EMA with.
        Returns:
            float: Updated EMA value.
        """
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

    def get_value(self):
        """ Returns the current EMA value. """
        return self.value


class AverageMeter:
    """ 
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """ Resets the meter """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Updates the meter with a new value """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Metric functions
def compute_ssim(im1, im2, win_sz=7):
    """
    Computes the Structural Similarity Index (SSIM) between two images (2D or 3D).
    
    Args:
        im1 (ndarray or Tensor): First image to compare, must have the same shape as im2.
        im2 (ndarray or Tensor): Second image to compare.
        win_sz (int, optional): Window size for SSIM calculation. Default is 7.

    Returns:
        float: Mean SSIM value between the two images.

    Raises:
        ValueError: If images have different shapes or unsupported dimensions (not 2D or 3D).
    """
    # Ensure images have the same shape
    if im1.shape != im2.shape:
        raise ValueError("Input images must have the same shape.")
    
    if im1.ndim in [3, 5]:  # 3D images (B, C, D, H, W)
        if im1.ndim == 3:
            im1 = im1.unsqueeze(0).unsqueeze(0)
            im2 = im2.unsqueeze(0).unsqueeze(0)
        ssim3d = SSIM3D(window_size=win_sz)
        return ssim3d(im1, im2).mean()
    
    elif im1.ndim in [2, 4]:  # 2D images (B, C, H, W) or (C, H, W)
        if im1.ndim == 2:
            im1 = im1.unsqueeze(0).unsqueeze(0)
            im2 = im2.unsqueeze(0).unsqueeze(0)
        ssim2d = SSIM(window_size=win_sz)
        return ssim2d(im1, im2).mean()
    
    else:
        raise ValueError("Unsupported image dimensions. Please provide 2D or 3D images.")


# The following functions are copied from: 
# https://github.com/jonasw247/addon-tumor-surrogate/tree/master/tumor_surrogate_pytorch
# with possible minor adjustments

def compute_dice_score(u_pred, u_sim, threshold):
    """ Computes the Dice similarity score between two binary tensors. 
        copied from 
    
    Args:
        u_pred (Tensor): Predicted tensor (binary values).
        u_sim (Tensor): Ground truth tensor (binary values).
        threshold (float): Threshold for binarization.

    Returns:
        Tensor: The Dice similarity score (between 0 and 1).
    """
    # Calculate True Positives, False Positives, and False Negatives
    tp = torch.sum((u_pred > threshold) * (u_sim > threshold)).float()
    tpfp = torch.sum(u_pred > threshold).float()  # True Positives + False Positives
    tpfn = torch.sum(u_sim > threshold).float()  # True Positives + False Negatives

    # Avoid division by zero
    if tpfn + tpfp == 0:
        return torch.tensor(0.0)  # Return 0.0 if both are zero

    # Calculate Dice score
    dice_score = (2 * tp) / (tpfp + tpfn)
    return dice_score


def loss_function(u_sim, u_pred, mask):
    """ Computes the weighted loss function (MSE) for GM-WM regions.
    
    Args:
        u_sim (Tensor): Ground truth tensor.
        u_pred (Tensor): Model prediction tensor.
        mask (Tensor): Mask for GM-WM regions.

    Returns:
        Tensor: Computed loss value.
    """
    pred_loss = torch.mean(torch.abs(u_sim[u_sim >= 0.001] * mask[u_sim >= 0.001] - u_pred[u_sim >= 0.001] * mask[u_sim >= 0.001])**2)
    csf_loss = 0
    wm_gm_Loss = torch.mean(torch.abs(u_sim - u_pred)**2 * mask)
    full_loss = torch.mean(torch.abs(u_sim - u_pred)**2)

    if math.isnan(pred_loss.item()):
        pred_loss = 0
    loss = 0.95 * pred_loss + 0.05 * wm_gm_Loss + 0.01 * full_loss   
    return loss



# TODO: check if ever used else delete (doesn't work because of the preprocessing)
def mean_absolute_error(ground_truth, output, input_tensor):
    """
    Computes the Mean Absolute Error (MAE) for White Matter, Gray Matter, and CSF.
    
    Args:
        ground_truth (Tensor): The ground truth segmentation values.
        output (Tensor): The model's predicted segmentation values.
        input_tensor (Tensor): The input tensor indicating tissue types.
        
    Returns:
        Tuple[Optional[float], Optional[float], Optional[float]]: 
        MAE for WM, GM, and CSF, or None if there are no valid values.
    """
    
    # Detach tensors from computation graph and move them to CPU
    ground_truth = ground_truth.detach().cpu()
    output = output.detach().cpu()
    input_tensor = input_tensor.cpu()

    # Apply masking based on input tensor indicating tissue types
    wm_mask = (input_tensor == 3)  # White Matter
    gm_mask = (input_tensor == 2)  # Gray Matter
    csf_mask = (input_tensor == 1)  # CSF


    # Calculate MAE for WM, GM, and CSF using the masks
    mae_wm = torch.mean(torch.abs(output[wm_mask] - ground_truth[wm_mask])) if wm_mask.sum() > 0 else None
    mae_gm = torch.mean(torch.abs(output[gm_mask] - ground_truth[gm_mask])) if gm_mask.sum() > 0 else None
    mae_csf = torch.mean(torch.abs(output[csf_mask] - ground_truth[csf_mask])) if csf_mask.sum() > 0 else None
    
    return mae_wm, mae_gm, mae_csf


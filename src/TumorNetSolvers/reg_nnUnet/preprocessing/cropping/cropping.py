# Copyright 2019 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
# This file has been modified from its original version.
# Modified by Zeineb Haouari on December 5, 2024

import numpy as np
from scipy.ndimage import binary_fill_holes

# Hello! crop_to_nonzero is the function you are looking for. Ignore the rest.
from acvl_utils.cropping_and_padding.bounding_boxes import get_bbox_from_mask, crop_to_bbox, bounding_box_to_slice


def create_nonzero_mask(data):
    """

    :param data:
    :return: the mask is True where the data is nonzero
    """
    assert data.ndim in (3, 4), "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = data[0] != 0
    for c in range(1, data.shape[0]):
        nonzero_mask |= data[c] != 0
    return binary_fill_holes(nonzero_mask)

def crop_to_nonzero(data,out=None, nonzero_label =-1):
    """
    :param data: Input data array.
    :param out: Optional output array for labeling.
    :param nonzero_label: This will be written into the segmentation map.
    :return: Tuple of (data, out, bbox).
    """
    # Create a nonzero mask from the data
    nonzero_mask = create_nonzero_mask(data)

    # Create a nonzero mask without slicing
    nonzero_mask = nonzero_mask[None]  # Add a new axis

    # Prepare the output
    if out is not None:
        # If out is provided, label the nonzero areas
        out = np.where(nonzero_mask, out, nonzero_label)
    else:
        # Create the output based on the nonzero mask
        out = np.where(nonzero_mask, np.int8(0), np.float32(nonzero_label))
    
    # Return the original data (no cropping), the output, and the bounding box
    return data, out



def crop_to_nonzero_orig_mod(data, out=None, nonzero_label=-1):
    """

    :param data:
    :param out:
    :param nonzero_label: this will be written into the segmentation map
    :return:
    """
    nonzero_mask = create_nonzero_mask(data)
    bbox = get_bbox_from_mask(nonzero_mask)
    print(bbox)
    #bbox=[[0, 64], [0, 64], [0, 64]]
    slicer = bounding_box_to_slice(bbox)
    nonzero_mask = nonzero_mask[slicer][None]

    slicer = (slice(None), ) + slicer
    data = data[slicer]
    if out is not None:
        out = out[slicer]
        out[(out == 0) & (~nonzero_mask)] = nonzero_label
    else:
        out = np.where(nonzero_mask, np.int8(0), np.float32(nonzero_label))
    return data, out, bbox

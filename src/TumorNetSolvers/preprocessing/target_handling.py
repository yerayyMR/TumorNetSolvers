"""
Class for target handling, so far only relevant function is determine_num_input_channels
"""

from __future__ import annotations
from time import time
from typing import Union, List, Tuple, Type
import torch
from typing import TYPE_CHECKING


class RegressionManager(object):
    def __init__(self, inference_nonlin=None, output_range: Tuple[float, float] = None):
        """
        A manager class for handling image-to-image regression tasks.

        Args:
            inference_nonlin: Optional. A non-linearity to apply to the model's output.
                              If not provided, the identity function is used.
            output_range: Optional. A tuple specifying the (min, max) range for the output values.
                          If provided, the output will be clamped to this range.
        """
        self.inference_nonlin = inference_nonlin if inference_nonlin is not None else lambda x: x  # Identity for unbounded regression
        self.output_range = output_range

    def apply_inference_nonlin(self, predictions: Union[np.ndarray, torch.Tensor]) -> \
            Union[np.ndarray, torch.Tensor]:
        """
        Apply the non-linearity (if any) to the predictions and optionally clamp the output to a specified range.

        Args:
            predictions: The model's raw output.

        Returns:
            The processed output with the non-linearity applied and clamped to the specified range.
        """
        if isinstance(predictions, np.ndarray):
            predictions = torch.from_numpy(predictions)

        with torch.no_grad():
            predictions = predictions.float()
            predictions = self.inference_nonlin(predictions)
            if self.output_range is not None:
                predictions = torch.clamp(predictions, *self.output_range)

        return predictions


    @staticmethod
    def determine_num_input_channels(dataset_json: dict) -> int:
        """
        Determines the number of input channels for the regression model based on dataset configuration.

        Args:
            dataset_json: A dictionary containing the dataset configuration.

        Returns:
            The number of input channels.
        """
        if 'channel_names' in dataset_json:
            num_modalities = len(dataset_json['channel_names'])
        else:
            raise ValueError("dataset_json must contain'channel_names'.")
        return num_modalities



    def get_regressionmanager_class_from_plans(plans: dict) -> Type[RegressionManager]:
        """
        Retrieves the appropriate manager class for the task from the plans. Defaults to RegressionManager.

        Args:
            plans: A dictionary containing the configuration plans.

        Returns:
            The manager class to be used for handling the task.
        """
        return RegressionManager

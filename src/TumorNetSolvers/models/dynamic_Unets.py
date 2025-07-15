# Copyright 2019 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
# Copyright 2022 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
# Modified by Zeineb Haouari on December 5, 2024
# This file has been modified from its original version. Code adapted from:
# - nnUnet (https://github.com/MIC-DKFZ/nnUNet)
# - Dynamic Network Architectures (https://github.com/MIC-DKFZ/dynamic-network-architectures)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Contains definitions of extended dynamic Unets to account for the integration of biophysical 
param vector at the bottleneck
"""
import pydoc
import warnings
from typing import Union, List, Tuple, Type, Optional
import numpy as np
import torch
from torch import nn
from torch.nn.modules.dropout import _DropoutNd
from torch.nn.modules.conv import _ConvNd
from torch import Size

from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
#from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from TumorNetSolvers.models.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
#from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op
from batchgenerators.utilities.file_and_folder_operations import join

import wandb

class PlainConvEncoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 return_skips: bool = False,
                 nonlin_first: bool = False,
                 inputs_shape: torch.Size = None,
                 pool: str = 'conv',
                 param_dim: int = 0,
                 experiments: List[List[str]] = None
                 ):

        super().__init__()
        self.param_dim = param_dim
        self.experiments = experiments
        self.input_shape = inputs_shape
        self.return_skips = return_skips

        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages

        self.kernel_sizes = kernel_sizes
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.pool = pool

        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.output_channels = features_per_stage

        # ModuleLists for pooling, conv blocks, and param linear layers
        self.pools = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.param_fcs = nn.ModuleList()

        current_input_channels = input_channels

        for s in range(n_stages):

            if isinstance(strides[s], (list, tuple)):
                if not all(v == strides[s][0] for v in strides[s]):
                    raise ValueError(f"Inconsistent strides at stage {s}: {strides[s]}")

            if s > 0:
                if s == 1: # s == 0 is not considered since this is the stacked convolution directly at the+6+-----* beginning before any downsampling or anything
                    current_input_shape = list(inputs_shape[-3:])[0]
                else:
                    current_input_shape = current_input_shape // strides[s][0]
            else:
                current_input_shape = None

            # Define pooling layer using adjusted in_channels
            if self.pool in {'max', 'avg'}:
                if isinstance(strides[s], int) and strides[s] != 1 or \
                isinstance(strides[s], (tuple, list)) and any(i != 1 for i in strides[s]):
                    pooling = get_matching_pool_op(conv_op, pool_type=self.pool)(kernel_size=strides[s], stride=strides[s])
                else:
                    pooling = nn.Identity()
                conv_stride = 1
            elif self.pool == 'conv':
                pooling = nn.Identity()
                conv_stride = strides[s]
            else:
                raise RuntimeError(f"Unknown pooling type: {self.pool}")

            self.pools.append(pooling)

            # Adjust in_channels for the conv layer
            conv_in_channels = current_input_channels
            if (
                s > 0 and param_dim > 0 and self.experiments[0] == "c"
                and (
                    (self.experiments[1] == "a_downsampling" and self.pool != 'conv') or
                    (self.experiments[1] == "b_bottleneck" and s == n_stages - 1 and self.pool != 'conv') or
                    (self.experiments[1] == "b_downsampling" and self.pool == 'conv')
                )
            ):
                conv_in_channels += param_dim

            # Param linear projection
            if (param_dim > 0 and s > 0 and
                (
                (self.experiments[1] == "a_downsampling" and self.pool != 'conv') or
                (self.experiments[1] == "b_bottleneck" and s == n_stages - 1 and self.pool != 'conv')
                )):
                if self.experiments[0] == "c":
                    self.param_fcs.append(nn.Linear(in_features=param_dim, out_features=(current_input_shape//strides[s][0])**3 * self.param_dim))
                elif self.experiments[0] == "a":
                    self.param_fcs.append(nn.Linear(in_features=param_dim, out_features=(current_input_shape//strides[s][0])**3 * conv_in_channels))
            elif (param_dim > 0 and s > 0 and
                (self.experiments[1] == "b_downsampling" and self.pool == 'conv')):
                if self.experiments[0] == "c":
                    self.param_fcs.append(nn.Linear(in_features=param_dim, out_features=current_input_shape**3 * self.param_dim))
                elif self.experiments[0] == "a":
                    self.param_fcs.append(nn.Linear(in_features=param_dim, out_features=current_input_shape**3 * conv_in_channels))
            else:
                self.param_fcs.append(None)
            
            # In order to let the class called that it is the stacked convolution before the bottleneck (after the downsampling)
            if ( s > 0 and
                ((self.experiments[1] == "b_bottleneck" and s == n_stages - 1 and self.pool == 'conv') or
                (self.experiments[1] == "a_downsampling" and self.pool == 'conv'))):

                linearLayer = True

                if self.experiments[0] == "c":
                    concat = True
                else:
                    concat = False
            else:
                linearLayer = False
                concat = False

            # Convolution block
            conv_block = StackedConvBlocks(
                n_conv_per_stage[s], conv_op, conv_in_channels, features_per_stage[s],
                kernel_sizes[s], conv_stride,
                conv_bias, norm_op, norm_op_kwargs,
                dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first,
                self.param_dim, linearLayer, concat, current_input_shape
            )
            self.convs.append(conv_block)

            current_input_channels = features_per_stage[s]


    def forward(self, x: torch.Tensor, params: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        skips = []
        out = x

        for s in range(len(self.convs)):
            out = self.pools[s](out)

            # Prepare param injection if applicable
            if self.param_dim > 0 and s > 0 and (
                (self.experiments[1] == "a_downsampling" and self.pool != 'conv') or
                (self.experiments[1] == "b_bottleneck" and s == len(self.convs) - 1 and self.pool != 'conv') or
                (self.experiments[1] == "b_downsampling" and self.pool == 'conv')
            ):
                param_proj = self.param_fcs[s](params)
                # Determine spatial dimensions
                B = x.shape[0]
                spatial_shape = out.shape[-3:]  # assuming NCHW or NCDHW
                if self.experiments[0] == "c":
                    param_proj = param_proj.view(B, self.param_dim, *spatial_shape)
                    out = torch.cat((out, param_proj), dim=1)
                elif self.experiments[0] == "a":
                    param_proj = param_proj.view(B, out.shape[1], *spatial_shape)
                    out = out + param_proj  # broadcasted addition

            if (s > 0 and ((self.experiments[1] == "a_downsampling" and self.pool == "conv") or
                (self.experiments[1] == "b_bottleneck" and self.pool == "conv" and s == len(self.convs) - 1))):
                out = self.convs[s](out, params)
            else:
                out = self.convs[s](out)

            if self.return_skips:
                skips.append(out)

        return skips if self.return_skips else out
    
    def compute_conv_feature_map_size(self, input_size: Tuple[int, int, int]) -> int:
        """
        Estimate the total number of features produced by the encoder.

        Args:
            input_size (Tuple[int, int, int]): The spatial input dimensions (D, H, W)

        Returns:
            int: Total number of feature values across all stages.
        """
        output = 0
        current_shape = input_size

        for s in range(len(self.convs)):
            # Apply stride (pooling/downsampling)
            stride = self.strides[s]
            if isinstance(stride[0], torch.Tensor):  # Defensive fallback
                stride = [int(x.item()) for x in stride]

            current_shape = [i // j for i, j in zip(current_shape, stride)]

            # Count conv output
            spatial_size = np.prod(current_shape, dtype=np.int64)
            out_channels = self.output_channels[s]
            output += out_channels * spatial_size

        return output


class UNetDecoder(nn.Module):
    """
    Implements a U-Net decoder with support for parameter integration and deep supervision.

    Args:
        encoder: Encoder model providing skip connections.
        num_classes: Number of output classes.
        n_conv_per_stage: Number of convolutions per stage.
        deep_supervision: Enables deep supervision.
        nonlin_first: Non-linearity is applied before normalization.
        norm_op, norm_op_kwargs: Normalization operation and its parameters.
        dropout_op, dropout_op_kwargs: Dropout operation and its parameters.
        nonlin, nonlin_kwargs: Non-linearity and its parameters.
        conv_bias: Bias in convolution layers.
        param_dim: Dimension of parameter vector for bottleneck integration.
    """
    def __init__(self,
                 encoder: Union[PlainConvEncoder, ResidualEncoder],
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision,
                 nonlin_first: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 conv_bias: bool = None, 
                 param_dim: int =5,
                 experiments: List[List[str]] = None
                 ):
        super().__init__()
        self.param_dim=param_dim
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        self.experiments = experiments
        n_stages_encoder = len(encoder.output_channels)
        self.param_fcs = nn.ModuleList()
        
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
        conv_bias = encoder.conv_bias if conv_bias is None else conv_bias
        norm_op = encoder.norm_op if norm_op is None else norm_op
        norm_op_kwargs = encoder.norm_op_kwargs if norm_op_kwargs is None else norm_op_kwargs
        dropout_op = encoder.dropout_op if dropout_op is None else dropout_op
        dropout_op_kwargs = encoder.dropout_op_kwargs if dropout_op_kwargs is None else dropout_op_kwargs
        nonlin = encoder.nonlin if nonlin is None else nonlin
        nonlin_kwargs = encoder.nonlin_kwargs if nonlin_kwargs is None else nonlin_kwargs

        # Initialize lists to store layers
        stages = []
        transpconvs = []
        seg_layers = []

        # Start from the full input shape
        input_shape = list(encoder.input_shape[-3:])  # D, H, W

        # Apply all encoder strides to get initial low-res spatial shape
        for stride in encoder.strides[1:]:  # skip first if it's identity
            input_shape = [input_shape[i] // stride[i] for i in range(3)]

        for s in range(1, n_stages_encoder):
            # Adjust the number of input features for the first stage
            if s == 1 and self.experiments[1] == "a_bottleneck" and self.experiments[0] == "c":
                input_channels_transpconv = encoder.output_channels[-s] + param_dim
            else:
                input_channels_transpconv = encoder.output_channels[-s]
            
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]

            if param_dim > 0 and self.experiments[1] == "b_upsampling" and self.experiments[0] == "c":
                input_channels_transpconv += self.param_dim  # Injected before upsampling

            # Param FC projection
            if param_dim > 0 and self.experiments is not None:
                proj_channels = param_dim if self.experiments[0] == "c" else input_channels_transpconv

                if self.experiments[1] == "b_upsampling":
                    # Before upsampling → divide input_shape by cumulative_stride
                    spatial_shape = input_shape
                    spatial_volume = spatial_shape[0] * spatial_shape[1] * spatial_shape[2]
                    self.param_fcs.append(nn.Linear(param_dim, proj_channels * spatial_volume))
            
            # Define the transpose convolution layer
            transpconv_layer = transpconv_op(
                input_channels_transpconv , input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=conv_bias
            )
            transpconvs.append(transpconv_layer)

            if self.experiments[1] in ["b_upsampling", "a_upsampling", "a_upsampling_skip"]:
                upsampled_shape = [input_shape[i] * stride_for_transpconv[i] for i in range(3)]
                input_shape = upsampled_shape

            # Adjust input channels for conv block based on experiment
            if param_dim > 0 and self.experiments[1] == "a_upsampling" and self.experiments[0] == "c":
                conv_in_channels = input_features_skip * 2 + self.param_dim
            else:
                conv_in_channels = input_features_skip * 2

            if param_dim > 0 and self.experiments is not None:
                if self.experiments[1] == "a_upsampling":
                    proj_channels = param_dim if self.experiments[0] == "c" else input_features_skip
                    # After upsampling → use input_shape directly
                    spatial_volume = input_shape[0] * input_shape[1] * input_shape[2]
                    self.param_fcs.append(nn.Linear(param_dim, proj_channels * spatial_volume))
                elif self.experiments[1] == "a_upsampling_skip":
                    proj_channels = param_dim if self.experiments[0] == "c" else conv_in_channels # Though only addition will occur
                    # After upsampling → use input_shape directly
                    spatial_volume = input_shape[0] * input_shape[1] * input_shape[2]
                    self.param_fcs.append(nn.Linear(param_dim, proj_channels * spatial_volume))
                
                    
            
            # Define the stacked convolution blocks
            conv_blocks = StackedConvBlocks(
                n_conv_per_stage[s-1], encoder.conv_op, conv_in_channels, input_features_skip,
                encoder.kernel_sizes[-(s + 1)], 1,
                conv_bias,
                norm_op,
                norm_op_kwargs,
                dropout_op,
                dropout_op_kwargs,
                nonlin,
                nonlin_kwargs,
                nonlin_first
            )
            stages.append(conv_blocks)
            
            # Define the segmentation layer
            seg_layer = encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True)
            seg_layers.append(seg_layer)

        # Convert lists to ModuleList and move to GPU if necessary
        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips, params: Optional[torch.Tensor] = None):
        lres_input = skips[-1]
        seg_outputs = []
        '''for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.stages[s](x)'''
        for s in range(len(self.stages)):

            lres_input_mod = lres_input

            if self.param_dim > 0 and self.experiments[1] == "b_upsampling":
                fc = self.param_fcs[s]
                B = lres_input.shape[0]
                spatial_shape = lres_input.shape[2:]
                param_proj = fc(params)

                if self.experiments[0] == "a":
                    param_proj = param_proj.view(B, lres_input.shape[1], *spatial_shape)
                    lres_input_mod = lres_input_mod + param_proj
                elif self.experiments[0] == "c":
                    param_proj = param_proj.view(B, self.param_dim, *spatial_shape)
                    lres_input_mod = torch.cat((lres_input_mod, param_proj), dim=1)

            x = self.transpconvs[s](lres_input_mod)

            if self.param_dim > 0 and self.experiments[1] == "a_upsampling":
                fc = self.param_fcs[s]
                B = lres_input.shape[0]
                spatial_shape = x.shape[2:]
                if self.experiments[0] == "a":
                    param_proj = fc(params).view(B, x.shape[1], *spatial_shape)
                    x = x + param_proj
                elif self.experiments[0] == "c":
                    param_proj = fc(params).view(B, self.param_dim, *spatial_shape)
                    x = torch.cat((x, param_proj), dim=1)

            # Continue with usual U-Net decoder flow
            x = torch.cat((x, skips[-(s + 2)]), 1)
            if self.param_dim > 0 and self.experiments[1] == "a_upsampling_skip":
                fc = self.param_fcs[s]
                B = lres_input.shape[0]
                spatial_shape = x.shape[2:]
                if self.experiments[0] == "a":
                    param_proj = fc(params).view(B, x.shape[1], *spatial_shape)
                    x = x + param_proj
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]

        assert len(skip_sizes) == len(self.stages)

        output = np.int64(0)
        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output




class PlainConvUNetNew(nn.Module):
    """
    Implements a U-Net architecture with parameter integration at the bottleneck.

    Args:
        input_channels: Number of input channels.
        n_stages: Number of stages in the encoder.
        features_per_stage: Features at each stage.
        conv_op, kernel_sizes, strides: Convolution operation and parameters.
        n_conv_per_stage: Convolutions per encoder stage.
        num_classes: Number of output classes.
        n_conv_per_stage_decoder: Convolutions per decoder stage.
        conv_bias, norm_op, norm_op_kwargs: Convolution and normalization parameters.
        dropout_op, dropout_op_kwargs: Dropout operation and parameters.
        nonlin, nonlin_kwargs: Non-linearity and its parameters.
        deep_supervision: Enables deep supervision.
        nonlin_first: Non-linearity is applied before normalization.
        param_dim: Dimension of parameter vector.
    """
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[nn.Dropout]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False,
                 param_dim: int =5,
                 inputs_shape: torch.Size = None,
                 experiments: List[List[str]] = None
                 ):
        super().__init__()
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages
        assert len(n_conv_per_stage_decoder) == (n_stages - 1)
        self.param_dim=param_dim
        self.experiments=experiments

        if self.experiments[1] == "inputs":
            self.latent_space_sz = list(inputs_shape[-3:])[0]
            if self.experiments[0] == "c":
                self.param_fc = nn.Linear(
                    in_features=param_dim,
                    out_features=self.latent_space_sz ** 3 * param_dim
                )
                input_channels += param_dim
            elif self.experiments[0] == "a":
                self.param_fc = nn.Linear(
                    in_features=param_dim,
                    out_features=self.latent_space_sz ** 3 * input_channels
                )
        
        self.input_channels = input_channels

        self.encoder = PlainConvEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                        n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                        nonlin_first=nonlin_first, inputs_shape=inputs_shape, pool='conv', param_dim=self.param_dim, experiments=self.experiments)
        
        if self.experiments[1] == "a_bottleneck":
            latent_spatial_size = list(inputs_shape[-3:])  # e.g., [64, 64, 64]
            for stride in self.encoder.strides:
                latent_spatial_size = [i // s for i, s in zip(latent_spatial_size, stride)]

            self.latent_space_sz = latent_spatial_size[0]  # if cubic

            if self.experiments[0] == "c":
                self.param_fc = nn.Linear(
                    in_features=param_dim,
                    out_features=self.latent_space_sz ** 3 * param_dim
                )
            elif self.experiments[0] == "a":
                self.param_fc = nn.Linear(
                    in_features=param_dim,
                    out_features=self.latent_space_sz ** 3 * self.encoder.output_channels[-1]
                )

        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                   nonlin_first=nonlin_first, param_dim=self.param_dim, experiments=self.experiments)

    def integrateParams(self, param, latent_space_sz, skips, batch_size):
        param = param.to(skips[-1].device)  # Ensure param is on the same device as the skips
        self.param_fc = self.param_fc.to(skips[-1].device)
        #param_fc = nn.Linear(in_features=len(param[0]), out_features=latent_space_sz ** 3 * len(param[0])).to(skips[-1].device)
        '''wandb.log({
            "param_fc/weight_mean": self.param_fc.weight.mean().item(),
            "param_fc/weight_std": self.param_fc.weight.std().item(),
            "param_fc/bias_mean": self.param_fc.bias.mean().item(),
            "param_fc/bias_std": self.param_fc.bias.std().item(),
        })'''
        
        if self.experiments[0] == "c":
            p = self.param_fc(param).view(batch_size, param.size(1), latent_space_sz, latent_space_sz, latent_space_sz)
            # Concatenate along channel dimension
            z_cat = torch.cat((skips[-1], p), dim=1)
            skips[-1] = z_cat
        else:
            p = self.param_fc(param).view(batch_size, self.encoder.output_channels[-1], latent_space_sz, latent_space_sz, latent_space_sz)
            # Add param as bias
            skips[-1] = skips[-1] + p

    def forward(self, x, param):
        batch_size = x.size(0)

        if self.experiments[1] == "inputs":
            param = param.to(x.device)
            self.param_fc = self.param_fc.to(x.device)

            if self.experiments[0] == "c":
                # Concatenate param feature map to input
                p = self.param_fc(param).view(
                    batch_size, self.param_dim, self.latent_space_sz, self.latent_space_sz, self.latent_space_sz
                )
                x = torch.cat((x, p), dim=1)
            elif self.experiments[0] == "a":
                # Add param as bias
                p = self.param_fc(param).view(
                    batch_size, self.input_channels, self.latent_space_sz, self.latent_space_sz, self.latent_space_sz
                )
                x = x + p

        if self.param_dim > 0 and self.experiments[1] in ["b_downsampling", "a_downsampling", "b_bottleneck"]:
            skips = self.encoder(x, param)
        else:
            skips = self.encoder(x)

        if self.experiments[1] == "a_bottleneck":
            latent_space_sz = skips[-1].shape[-1]
            self.integrateParams(param, latent_space_sz, skips, batch_size)

        if self.param_dim > 0 and self.experiments[1] in ["b_upsampling", "a_upsampling", "a_upsampling_skip"]:
            return self.decoder(skips, param)
        else:
            return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op)
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)



def get_network_from_plans_new(arch_class_name, arch_kwargs, arch_kwargs_req_import, input_channels, output_channels, inputs_shape,
                           experiments=['c', 'b_bottleneck'], allow_init=True, deep_supervision: Union[bool, None] = None):
    architecture_kwargs = dict(**arch_kwargs)
    architecture_classes = {
    'PlainConvUnetNew': PlainConvUNetNew
    }
    for ri in arch_kwargs_req_import:
        if architecture_kwargs[ri] is not None:
            architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])

    nw_class = architecture_classes[arch_class_name]
    # sometimes things move around, this makes it so that we can at least recover some of that

    if deep_supervision is not None:
        architecture_kwargs['deep_supervision'] = deep_supervision
    
    

    network = nw_class(
        input_channels=input_channels,
        num_classes=output_channels,
        inputs_shape=inputs_shape,
        experiments=experiments,
        **architecture_kwargs
    )

    if hasattr(network, 'initialize') and allow_init:
        network.apply(network.initialize)

    return network


#original baseline slight modifs from: https://github.com/jonasw247/addon-tumor-surrogate/blob/master/tumor_surrogate_pytorch

import torch
from torch import nn
import torch.nn.functional as F

class IdentityLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class ConvLayer3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, act_func='relu', use_bn=True):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.use_bn = use_bn
        if self.use_bn:
            self.bn = nn.BatchNorm3d(out_channels)
        if act_func == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = None

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x

class ManyfoldConvBlock3D(nn.Module):
    def __init__(self, layers, shortcut, skip_pos, linear_layer=None, experiment=None, param_dim=5, width=None):
        super(ManyfoldConvBlock3D, self).__init__()
        self.skip_pos = skip_pos
        self.layers = nn.ModuleList(layers)
        self.shortcut = shortcut
        self.linear_layer=linear_layer
        self.experiment = experiment
        self.param_dim = param_dim
        self.width = width

    def forward(self, x, skip_x=None, params=None):
        if skip_x is None:  # encoder
            skip_x = self.shortcut(x)
        else:  # decoder
            skip_x = self.shortcut(skip_x)

        last_idx = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            if i == last_idx:
                # Use self.linear_layer as param_fc applied on params
                if self.param_dim > 0 and params is not None:
                    if self.experiment[1] == "b_downsampling" or self.experiment[1] == "b_upsampling" or self.experiment[1] == "b_upsampling_after":
                        # Apply param_fc (linear layer) on params
                        param_proj = self.linear_layer(params)

                        B = x.shape[0]
                        spatial_shape = x.shape[-3:]  # spatial shape before last conv

                        if self.experiment[0] == "c":
                            param_proj = param_proj.view(B, self.param_dim, *spatial_shape)
                            x = torch.cat((x, param_proj), dim=1)
                        elif self.experiment[0] == "a":
                            param_proj = param_proj.view(B, x.shape[1], *spatial_shape)
                            x = x + param_proj  # broadcasted addition

            elif i == 0:
                if self.param_dim > 0 and params is not None:
                    if self.experiment[1] == "a_downsampling" or self.experiment[1] == "a_upsampling_after":
                        # Apply param_fc (linear layer) on params
                        param_proj = self.linear_layer(params)

                        B = x.shape[0]
                        spatial_shape = x.shape[-3:]  # spatial shape before last conv

                        if self.experiment[0] == "c":
                            param_proj = param_proj.view(B, self.param_dim, *spatial_shape)
                            x = torch.cat((x, param_proj), dim=1)
                        elif self.experiment[0] == "a":
                            param_proj = param_proj.view(B, x.shape[1], *spatial_shape)
                            x = x + param_proj  # broadcasted addition

            x = layer(x)

            if i == self.skip_pos:
                x = x + skip_x
        return x

class TumorSurrogate(nn.Module):
    def __init__(self, widths, n_cells, strides, experiment, inputs_shape, param_dim):
        super().__init__()
        input_channel = 1  # Changed input channels to 1 for single channel input
        data_sz = list(inputs_shape[-3:])[0]  # e.g., [64, 64, 64]
        self.experiment = experiment
        self.param_dim = param_dim
        self.param_fc_after_upsample = nn.ModuleList()

        if experiment[1] == "inputs" and self.param_dim > 0:
            if experiment[0] == "c":
                self.param_fc = nn.Linear(
                    in_features=param_dim,
                    out_features=data_sz ** 3 * param_dim
                )
                input_channel += param_dim
            elif experiment[0] == "a":
                self.param_fc = nn.Linear(
                    in_features=param_dim,
                    out_features=data_sz ** 3 * input_channel
                )
        first_conv = ConvLayer3D(
            input_channel, 1, kernel_size=3, stride=1, use_bn=True
        )

        if experiment[1] == "inputs" and self.experiment[0] == "c" and self.param_dim > 0:
            input_channel = 1 
        
        first_conv_flag = True
        encoder_blocks = [first_conv]
        prev_s = None
        for width, n_cell, s in zip(widths, n_cells, strides):
            if prev_s == None:
                prev_s = s
            conv_layers = []
            shortcut = IdentityLayer()
            if s == 1:
                skip_pos = n_cell - 1
            else:
                skip_pos = n_cell - 2
            for i in range(n_cell):
                if i == n_cell - 1:  # last layer of block is pooling or stride conv
                    stride = s
                    if experiment[1] == "b_downsampling" and self.param_dim > 0 and stride != 1:
                        if experiment[0] == "c":
                            self.param_fc = nn.Linear(
                                in_features=param_dim,
                                out_features=data_sz ** 3 * param_dim
                            )
                            input_channel += param_dim
                        elif experiment[0] == "a":
                            self.param_fc = nn.Linear(
                                in_features=param_dim,
                                out_features=data_sz ** 3 * input_channel
                            )
                        data_sz = data_sz // s
                else:
                    stride = 1

                if i == 0 and experiment[1] == "a_downsampling" and not first_conv_flag and self.param_dim > 0:
                    data_sz = data_sz // prev_s
                    prev_s = s

                    if experiment[0] == "c":
                        self.param_fc = nn.Linear(
                                in_features=param_dim,
                                out_features=data_sz ** 3 * param_dim
                            )
                        input_channel += param_dim
                    elif experiment[0] == "a":
                        self.param_fc = nn.Linear(
                                in_features=param_dim,
                                out_features=data_sz ** 3 * input_channel
                            )
                conv_op = ConvLayer3D(in_channels=input_channel, out_channels=width, kernel_size=3, stride=stride, use_bn=True)
                conv_layers.append(conv_op)
                input_channel = width
                


            if ((experiment[1] == "b_downsampling" and s != 1) or (experiment[1] == "a_downsampling" and not first_conv_flag)) and self.param_dim > 0:
                conv_block = ManyfoldConvBlock3D(conv_layers, shortcut, skip_pos=skip_pos, linear_layer=self.param_fc, experiment=experiment, width=width)
            else:
                conv_block = ManyfoldConvBlock3D(conv_layers, shortcut, skip_pos=skip_pos)
            encoder_blocks.append(conv_block)

            if first_conv_flag:
                first_conv_flag=False


        if experiment[1] == "b_bottleneck" and self.param_dim > 0:
            for stride in strides:
                data_sz = data_sz // stride
        
        if (experiment[1] == "b_bottleneck") and self.param_dim > 0: # (experiment[1] == "b_bottleneck" or experiment[1] == "a_downsampling") and self.param_dim > 0: 
            if experiment[0] == "c":
                self.param_fc = nn.Linear(
                    in_features=param_dim,
                    out_features=data_sz ** 3 * param_dim
                )
                input_channel += param_dim
            elif experiment[0] == "a":
                self.param_fc = nn.Linear(
                    in_features=param_dim,
                    out_features=data_sz ** 3 * input_channel
                )

        if experiment[1] == "a_bottleneck" and experiment[0] == "c" and self.param_dim > 0:
            mid_conv = ConvLayer3D(
                input_channel, widths[-1] - 5, kernel_size=3, stride=1
            )
            input_channel = widths[-1] - 5
            encoder_blocks.append(mid_conv)
        else:
            mid_conv = ConvLayer3D(
                input_channel, widths[-1], kernel_size=3, stride=1
            )
            input_channel = widths[-1]
            encoder_blocks.append(mid_conv)

        if (experiment[1] == "a_bottleneck" or experiment[1] == "a_bottleneck_after") and self.param_dim > 0:
            for stride in strides:
                data_sz = data_sz // stride

            if experiment[0] == "c":
                self.param_fc = nn.Linear(
                    in_features=param_dim,
                    out_features=data_sz ** 3 * param_dim
                )
                input_channel += param_dim
            elif experiment[0] == "a":
                self.param_fc = nn.Linear(
                    in_features=param_dim,
                    out_features=data_sz ** 3 * input_channel
                )

        if experiment[1] in ["b_upsampling", "a_upsampling", "a_upsampling_after", "b_upsampling_after"] and self.param_dim > 0:
            for stride in strides:
                data_sz = data_sz // stride

        first_conv_flag = True
        prev_s = None
        decoder_blocks = []
        n_cells_decoder = [x + 1 for x in n_cells] # Augmented since the upsampling is done via an extra layer
        for width, n_cell, s in zip(widths, n_cells_decoder, strides):
            if prev_s == None:
                prev_s = s
            conv_layers = []
            if s == 1:
                skip_pos = n_cell - 1
            else:
                skip_pos = n_cell - 2
            shortcut = IdentityLayer()
            for i in range(n_cell):
                # BEFORE UPSAMPLING
                if i == n_cell - 1 and s != 1 and (experiment[1] == "b_upsampling" or experiment[1] == "b_upsampling_after") and self.param_dim > 0:
                    if experiment[0] == "c":
                        self.param_fc = nn.Linear(
                            in_features=param_dim,
                            out_features=data_sz ** 3 * param_dim
                        )
                    elif experiment[0] == "a":
                        self.param_fc = nn.Linear(
                            in_features=param_dim,
                            out_features=data_sz ** 3 * input_channel
                        )
                    data_sz = data_sz * s  # Upsampling increases size

                if i == 0 and experiment[1] == "a_upsampling_after" and not first_conv_flag and self.param_dim > 0:
                    data_sz = data_sz * prev_s
                    prev_s = s
                    if experiment[0] == "c":
                        self.param_fc = nn.Linear(
                            in_features=param_dim,
                            out_features=data_sz ** 3 * param_dim
                        )
                        input_channel += param_dim
                    elif experiment[0] == "a":
                        self.param_fc = nn.Linear(
                            in_features=param_dim,
                            out_features=data_sz ** 3 * input_channel
                        )

                if i == n_cell - 1 and s != 1:  # last layer of block is Upsampling
                    conv_op = nn.Upsample(scale_factor=s, mode='nearest')
                else:
                    conv_op = ConvLayer3D(in_channels=input_channel, out_channels=width, kernel_size=3, stride=1, use_bn=True)

                conv_layers.append(conv_op)
                input_channel = width
            if (((experiment[1] == "b_upsampling" or experiment[1] == "b_upsampling_after") and s != 1) or (experiment[1] == "a_upsampling_after" and not first_conv_flag)) and self.param_dim > 0:
                conv_block = ManyfoldConvBlock3D(
                    conv_layers, shortcut, skip_pos=skip_pos, linear_layer=self.param_fc,
                    experiment=experiment, width=width
                )
            else:
                conv_block = ManyfoldConvBlock3D(conv_layers, shortcut, skip_pos=skip_pos)
            decoder_blocks.append(conv_block)

            if first_conv_flag:
                first_conv_flag=False

            if s != 1: # Since an upsample layer is inserted for the upsampling an extra convolution is required
                if experiment[1] == "b_upsampling" and experiment[0] == "c" and self.param_dim > 0:
                    after_upscale_conv_b_upsampling = ConvLayer3D(
                        in_channels=input_channel + param_dim, out_channels=width,
                        kernel_size=3, stride=1
                    )
                    decoder_blocks.append(after_upscale_conv_b_upsampling)
                if experiment[1] == "a_upsampling" and self.param_dim > 0:
                    data_sz = data_sz * s

                    if experiment[0] == "c":
                        self.param_fc_after_upsample.append(nn.Linear(
                                in_features=param_dim,
                                out_features=data_sz ** 3 * param_dim
                            ))
                        input_channel += param_dim
                    elif experiment[0] == "a":
                        self.param_fc_after_upsample.append(nn.Linear(
                                in_features=param_dim,
                                out_features=data_sz ** 3 * input_channel
                            ))
                    after_upscale_conv = ConvLayer3D(
                        in_channels=input_channel, out_channels=width,
                        kernel_size=3, stride=1
                    )
                    input_channel = width
                elif experiment[1] == "b_upsampling_after" and experiment[0] == "c" and self.param_dim > 0:
                    after_upscale_conv = ConvLayer3D(
                        in_channels=input_channel + param_dim, out_channels=width,
                        kernel_size=3, stride=1
                    )
                else:
                    after_upscale_conv = ConvLayer3D(
                        in_channels=input_channel, out_channels=width,
                        kernel_size=3, stride=1
                    )
                decoder_blocks.append(after_upscale_conv)
        # Final layer
        last_channel = 1
        last_conv = ConvLayer3D(
            input_channel, last_channel, kernel_size=3, stride=1, act_func=None
        )
        decoder_blocks.append(last_conv)

        self.encoder_blocks = nn.ModuleList(encoder_blocks)
        self.decoder_blocks = nn.ModuleList(decoder_blocks)

    def forward(self, x, parameters):
        skips = []
        out = x

        # If param injection happens at the input level
        if self.experiment[1] == "inputs" and self.param_dim > 0:
            B = x.shape[0]
            spatial_shape = list(x.shape[-3:])  # assuming NCDHW

            param_proj = self.param_fc(parameters)

            if self.experiment[0] == "c":
                param_proj = param_proj.view(B, self.param_dim, *spatial_shape)
                out = torch.cat((out, param_proj), dim=1)
            elif self.experiment[0] == "a":
                param_proj = param_proj.view(B, out.shape[1], *spatial_shape)
                out = out + param_proj

        for s, block in enumerate(self.encoder_blocks):
            if isinstance(block, ManyfoldConvBlock3D):
                if ((self.experiment[1] == "b_downsampling" and s != 4) or (self.experiment[1] == "a_downsampling" and s > 1)) and self.param_dim > 0: # s=0 is input convolution, s=1 is the one after not of interest on the current implementation of a_downsampling
                    out = block(out, skip_x = None, params=parameters)
                else:
                    out = block(out)
                skips.append(out)

            else:
                if s == len(self.encoder_blocks) - 1 and (self.experiment[1] == "b_bottleneck") and self.param_dim > 0: #s == len(self.encoder_blocks) - 1 and (self.experiment[1] == "b_bottleneck" or self.experiment[1] == "a_downsampling") and self.param_dim > 0: 
                    B = x.shape[0]
                    spatial_shape = out.shape[-3:]
                    param_proj = self.param_fc(parameters)

                    if self.experiment[0] == "c":
                        param_proj = param_proj.view(B, self.param_dim, *spatial_shape)
                        out = torch.cat((out, param_proj), dim=1)
                    elif self.experiment[0] == "a":
                        param_proj = param_proj.view(B, out.shape[1], *spatial_shape)
                        out = out + param_proj
                out = block(out)  # first conv or mid_conv

        if self.experiment[1] == "a_bottleneck_after":
            skip_x = out

        # For b_bottleneck or a_bottleneck injection
        if (self.experiment[1] == "a_bottleneck" or self.experiment[1] == "a_bottleneck_after") and self.param_dim > 0:
            B = x.shape[0]
            spatial_shape = out.shape[-3:]
            param_proj = self.param_fc(parameters)

            if self.experiment[0] == "c":
                param_proj = param_proj.view(B, self.param_dim, *spatial_shape)
                out = torch.cat((out, param_proj), dim=1)
            elif self.experiment[0] == "a":
                param_proj = param_proj.view(B, out.shape[1], *spatial_shape)
                out = out + param_proj

        if self.experiment[1] != "a_bottleneck_after":
            skip_x = out
        b_upsampling = False
        a_upsampling_counter = 0
        # Decoder pass
        for i, block in enumerate(self.decoder_blocks):
            if isinstance(block, ManyfoldConvBlock3D):
                contains_upsample = any(isinstance(layer, nn.Upsample) for layer in block.layers)
                if self.param_dim > 0:
                    if contains_upsample:
                        if self.experiment[1] == "b_upsampling" or self.experiment[1] == "b_upsampling_after" or (self.experiment[1] == "a_upsampling_after" and i != 0):
                            out = block(out, skip_x=skip_x, params=parameters)
                            if self.experiment[0] == "c" and self.experiment[1] == "b_upsampling":
                                b_upsampling = True
                        else:
                            out = block(out, skip_x=skip_x)
                    else:
                        if self.experiment[1] == "a_upsampling_after" and i != 0:
                            out = block(out, skip_x=skip_x, params=parameters)
                        else:
                            out = block(out, skip_x=skip_x)
                else:
                    out = block(out, skip_x=skip_x)
            else:
                if b_upsampling:
                    out = block(out)
                    b_upsampling = False
                    continue
                if self.experiment[1] != "b_upsampling_after":
                    skip_x = out
                if self.param_dim > 0 and self.experiment[1] == "a_upsampling" and i != len(self.decoder_blocks) - 1 and len(self.param_fc_after_upsample) > 0:
                    param_fc = self.param_fc_after_upsample[a_upsampling_counter]
                    a_upsampling_counter += 1

                    B = x.shape[0]
                    spatial_shape = out.shape[-3:]
                    param_proj = param_fc(parameters)

                    if self.experiment[0] == "c":
                        param_proj = param_proj.view(B, self.param_dim, *spatial_shape)
                        out = torch.cat((out, param_proj), dim=1)
                    elif self.experiment[0] == "a":
                        param_proj = param_proj.view(B, out.shape[1], *spatial_shape)
                        out = out + param_proj
                out = block(out)
                if self.experiment[1] == "b_upsampling_after":
                    skip_x = out

        return torch.sigmoid(out)

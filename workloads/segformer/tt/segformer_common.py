# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import ttsim.front.ttnn as ttnn


class Conv:
    def __init__(
        self,
        conv_params,
        parameters,
        *,
        groups=1,  # Must accept groups parameter
        dtype=ttnn.bfloat8_b,
        output_layout=ttnn.TILE_LAYOUT,
    ) -> None:
        self.weights = parameters["weight"]
        self.bias = parameters["bias"]
        self.kernel_size = (self.weights.shape[2], self.weights.shape[3])
        self.conv_params = conv_params
        self.out_channels = self.weights.shape[0]
        self.groups = groups  # MUST store groups
        self.dtype = dtype
        self.output_layout = output_layout

    def __call__(self, device, input_tensor):
        batch_size = input_tensor.shape[0]
        in_channels = input_tensor.shape[1]
        input_height = input_tensor.shape[2]
        input_width = input_tensor.shape[3]

        stride_h = self.conv_params[0]
        stride_w = self.conv_params[1]
        pad_h = self.conv_params[2]
        pad_w = self.conv_params[3]

        out_height = (input_height + 2 * pad_h - self.kernel_size[0]) // stride_h + 1
        out_width = (input_width + 2 * pad_w - self.kernel_size[1]) // stride_w + 1

        output_tensor = ttnn.conv2d(
            input_tensor=input_tensor,
            weight_tensor=self.weights,
            bias_tensor=self.bias,
            in_channels=in_channels,
            out_channels=self.out_channels,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            kernel_size=self.kernel_size,
            stride=(stride_h, stride_w),
            padding=(pad_h, pad_w),
            dilation=(1, 1),
            groups=self.groups,  # MUST pass groups here
            device=device,
            dtype=self.dtype,
        )

        return output_tensor, out_height, out_width
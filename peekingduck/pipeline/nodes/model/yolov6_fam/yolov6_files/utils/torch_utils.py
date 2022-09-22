# Copyright 2022 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Torch Utility Functions."""

from __future__ import annotations
from typing import no_type_check
import torch

from torch import nn
from peekingduck.pipeline.nodes.model.yolov6_fam.yolov6_files.layers.common import (
    Conv,
)
from peekingduck.pipeline.nodes.model.yolov6_fam.yolov6_files.models.yolo import YOLOv6


@no_type_check
def fuse_conv_and_bn(conv_layer: nn.Conv2d, bn_layer: nn.BatchNorm2d) -> nn.Conv2d:
    """Fuse convolution and batchnorm layers.

    Args:
        conv_layer (nn.Conv2d): Convolution layer.
        bn_layer (nn.BatchNorm2d): Batchnorm layer.

    Returns:
        fusedconv (nn.Conv2d): The fused convolution layer.

    Reference:
         https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    """

    fusedconv = (
        nn.Conv2d(
            conv_layer.in_channels,
            conv_layer.out_channels,
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            groups=conv_layer.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv_layer.weight.device)
    )

    # prepare filters
    w_conv = conv_layer.weight.clone().view(conv_layer.out_channels, -1)
    w_bn = torch.diag(
        bn_layer.weight.div(torch.sqrt(bn_layer.eps + bn_layer.running_var))
    )
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = (
        torch.zeros(conv_layer.weight.size(0), device=conv_layer.weight.device)
        if conv_layer.bias is None
        else conv_layer.bias
    )
    b_bn = bn_layer.bias - bn_layer.weight.mul(bn_layer.running_mean).div(
        torch.sqrt(bn_layer.running_var + bn_layer.eps)
    )
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


@no_type_check
def fuse_model(model: type[YOLOv6]) -> YOLOv6:
    """Fuses Conv and BatchNorm layers.

    Args:
        model (YOLOv6): The model to fuse.

    Returns:
        YOLOv6: The fused model.
    """

    for module in model.modules():
        if isinstance(module, Conv) and hasattr(module, "bn"):
            module.conv = fuse_conv_and_bn(module.conv, module.bn)  # update conv
            delattr(module, "bn")  # remove batchnorm
            module.forward = module.forward_fuse  # update forward
    return model

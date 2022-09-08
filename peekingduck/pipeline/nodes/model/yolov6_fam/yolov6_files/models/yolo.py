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

"""
1. Moved initialize_weights from torch_utils to here and turned into private method.
2. Removed import *.
3. Changed `detect` to `head` for consistency. OK it does not work since weights
saved as `head.weights` etc.
4. Renamed Model to YOLOv6.
5. Changed `config` to `model_params` to avoid name conflicts with pkd's config.
6. Group make_divisible and build_network to methods.
"""
from __future__ import annotations
import math
from typing import Callable, Tuple

import torch
from torch import nn
from peekingduck.pipeline.nodes.model.yolov6_fam.yolov6_files.configs.config import (
    ModelParams,
)
from peekingduck.pipeline.nodes.model.yolov6_fam.yolov6_files.models.efficientrep import (
    EfficientRep,
)

# pylint: disable=line-too-long
from peekingduck.pipeline.nodes.model.yolov6_fam.yolov6_files.models.efficient_decoupled_head import (
    EfficientDecoupledHead,
    build_efficient_decoupled_head,
)
from peekingduck.pipeline.nodes.model.yolov6_fam.yolov6_files.models.rep_pan_neck import (
    RepPANNeck,
)
from peekingduck.pipeline.nodes.model.yolov6_fam.yolov6_files.layers.common import (
    RepVGGBlock,
)


class YOLOv6(nn.Module):
    """YOLOv6 model with backbone, neck and head.
    The default parts are EfficientRep Backbone, Rep-PAN and
    Efficient Decoupled Head.
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        config: type[ModelParams],
        num_classes: int,
        anchors: int,
    ):
        super().__init__()

        self.config = config
        self.num_classes = num_classes
        self.anchors = anchors

        # Build network
        self.num_layers = self.config.model.head.num_layers
        self.channels = self.config.num_channels

        self.backbone, self.neck, self.detect = self.build_network()

        # Init EfficientDecoupledHead head
        begin_indices = self.config.model.head.begin_indices
        out_indices_head = self.config.model.head.out_indices
        self.stride = self.detect.stride
        self.detect.i = begin_indices
        self.detect.f = out_indices_head
        self.detect._initialize_biases()

        # Init weights
        self._initialize_weights()

    # changed to isinstance checks
    def _initialize_weights(self) -> None:
        """Initialize model weights."""
        # initialize_weights(self) -> initialize_weights(self)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                module.eps = 1e-3
                module.momentum = 0.03

            elif isinstance(
                module, (nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.Hardswish, nn.SiLU)
            ):
                module.inplace = True

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call.

        Args:
            inputs (torch.Tensor): Input image.

        Shapes:
            inputs: (batch_size, channels, height, width)
            outputs: (batch_size, num_dets, (x, y, w, h, conf, (cls_confg of the
                     80 COCO classes))]

        Returns:
            (torch.Tensor): The decoded output with the shape (B,D,85) where
            B is the batch size, D is the number of detections. The 85 columns
            consist of the following values:
            [x, y, w, h, conf, (cls_conf of the 80 COCO classes)].
        """

        backbone_outputs = self.backbone(inputs)
        neck_outputs = self.neck(backbone_outputs)
        head_outputs = self.detect(neck_outputs)

        return head_outputs

    def _apply(self, fn: Callable) -> "YOLOv6":
        """Apply fn to all submodules.

        Args:
            fn (Callable): Function to apply to each submodule.

        Returns:
            (YOLOv6): self.

        Reference:
            https://github.com/pytorch/pytorch/blob/4618371da56c887195e2e1d16dad
            2b9686302800/torch/nn/modules/module.py#L808
            https://github.com/ultralytics/yolov5/blob/master/models/yolo.py
        """
        # Apply to(), cpu(), cuda(), half() to model tensors that are not
        # parameters or registered buffers
        self = super()._apply(fn)  # pylint: disable=self-cls-assignment
        self.detect.stride = fn(self.detect.stride)
        self.detect.grid = list(map(fn, self.detect.grid))
        return self

    @staticmethod
    def make_divisible(num: float, divisor: int) -> int:
        """Upward revision the value num to make it evenly divisible by the divisor.

        Args:
            num (float): The value to be divided.
            divisor (int): The divisor.

        Returns:
            (int): The ceiling of the division.
        """

        return math.ceil(num / divisor) * divisor

    def build_network(self) -> Tuple[EfficientRep, RepPANNeck, EfficientDecoupledHead]:
        """Builds the network and returns the backbone, neck and head.

        Returns:
            Tuple[EfficientRep, RepPANNeck, EfficientDecoupledHead]: Returns the
            backbone, neck and head of YOLOv6.
        """
        depth_mul = self.config.model.depth_multiple
        width_mul = self.config.model.width_multiple
        num_repeat_backbone = self.config.model.backbone.num_repeats
        channels_list_backbone = self.config.model.backbone.out_channels
        num_repeat_neck = self.config.model.neck.num_repeats
        channels_list_neck = self.config.model.neck.out_channels
        num_anchors = self.config.model.head.anchors
        num_repeat = [
            (max(round(i * depth_mul), 1) if i > 1 else i)
            for i in (num_repeat_backbone + num_repeat_neck)
        ]
        channels_list = [
            self.make_divisible(i * width_mul, 8)
            for i in (channels_list_backbone + channels_list_neck)
        ]

        backbone = EfficientRep(
            in_channels=self.channels,
            channels_list=channels_list,
            num_repeats=num_repeat,
        )

        neck = RepPANNeck(channels_list=channels_list, num_repeats=num_repeat)

        head_layers = build_efficient_decoupled_head(
            channels_list, num_anchors, self.num_classes
        )

        head = EfficientDecoupledHead(
            self.num_classes, self.anchors, self.num_layers, head_layers=head_layers
        )

        return backbone, neck, head

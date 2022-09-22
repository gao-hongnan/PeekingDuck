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

"""Head
1. Set init weights to private, don't want users to change them.
2. Training mode is toggle-able.
"""

import math
from typing import List
import torch
from torch import nn

from peekingduck.pipeline.nodes.model.yolov6_fam.yolov6_files.layers.common import Conv


class EfficientDecoupledHead(nn.Module):
    """Efficient Decoupled Head
    With hardware-aware design, the decoupled head is optimized with
    hybridchannels methods.
    """

    # pylint: disable=too-many-arguments, too-many-instance-attributes
    def __init__(
        self,
        num_classes: int = 80,
        anchors: int = 1,
        num_layers: int = 3,
        head_layers: nn.Sequential = None,
    ):
        super().__init__()

        assert head_layers is not None

        self.i: int
        self.f: List[int]  # pylint: disable=invalid-name

        self.num_classes = num_classes  # number of classes
        self.num_outputs = self.num_classes + 5  # number of outputs per anchor
        self.num_layers = num_layers  # number of detection layers
        self.num_anchors = anchors

        self.grid = [torch.zeros(1)] * num_layers
        self.prior_prob = 1e-2
        stride = [8, 16, 32]  # strides computed during build
        self.stride = torch.tensor(stride)

        # Init decouple head
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()

        # Efficient decoupled head layers
        for i in range(num_layers):
            idx = i * 6
            self.stems.append(head_layers[idx])
            self.cls_convs.append(head_layers[idx + 1])
            self.reg_convs.append(head_layers[idx + 2])
            self.cls_preds.append(head_layers[idx + 3])
            self.reg_preds.append(head_layers[idx + 4])
            self.obj_preds.append(head_layers[idx + 5])

    def _initialize_biases(self) -> None:
        """Initialize biases for head layers where
        cls refers to class prediction layer;
        reg refers to regression layer;
        obj refers to objectness prediction layer.
        """
        for conv in self.cls_preds:
            bias = conv.bias.view(self.num_anchors, -1)
            bias.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
            conv.bias = nn.Parameter(bias.view(-1), requires_grad=True)
        for conv in self.obj_preds:
            bias = conv.bias.view(self.num_anchors, -1)
            bias.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
            conv.bias = nn.Parameter(bias.view(-1), requires_grad=True)

    # pylint: disable=invalid-name, too-many-locals
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass of Efficient Decoupled Head, the head of YOLOv6.

        Args:
            inputs (Tuple[torch.Tensor]): A tuple of tensors to be unpacked and fed into the
            network.

        Returns:
            outputs (torch.Tensor): A list of tensors that represent the output of the
            network.
        """

        outputs = []
        for i in range(self.num_layers):
            inputs[i] = self.stems[i](inputs[i])
            cls_x = inputs[i]
            reg_x = inputs[i]
            cls_feat = self.cls_convs[i](cls_x)
            cls_output = self.cls_preds[i](cls_feat)
            reg_feat = self.reg_convs[i](reg_x)
            reg_output = self.reg_preds[i](reg_feat)
            obj_output = self.obj_preds[i](reg_feat)

            y = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1)
            batch_size, _, hsize, wsize = y.shape

            y = (
                y.view(batch_size, self.num_anchors, self.num_outputs, hsize, wsize)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )
            if self.grid[i].shape[2:4] != y.shape[2:4]:
                d = self.stride.device
                yv, xv = torch.meshgrid(
                    [torch.arange(hsize).to(d), torch.arange(wsize).to(d)]
                )
                self.grid[i] = (
                    torch.stack((xv, yv), 2)
                    .view(1, self.num_anchors, hsize, wsize, 2)
                    .float()
                )
            # inplace=True
            y[..., 0:2] = (y[..., 0:2] + self.grid[i]) * self.stride[i]  # xy
            y[..., 2:4] = torch.exp(y[..., 2:4]) * self.stride[i]  # wh

            outputs.append(y.view(batch_size, -1, self.num_outputs))

        return torch.cat(outputs, 1)


def build_efficient_decoupled_head(
    channels_list: List[int], num_anchors: int, num_classes: int
) -> nn.Sequential:
    """Builds the Efficient Decoupled Head layers.

    The output is fed into Detect.

    Args:
        channels_list (List[int]): Channels in and out for backbone and neck.
        num_anchors (int): Number of anchors.
        num_classes (int): Number of classes.

    Returns:
        head_layers (nn.Sequential): The Efficient Decoupled Head layers.
    """

    head_layers = nn.Sequential(
        # stem0
        Conv(
            in_channels=channels_list[6],
            out_channels=channels_list[6],
            kernel_size=1,
            stride=1,
        ),
        # cls_conv0
        Conv(
            in_channels=channels_list[6],
            out_channels=channels_list[6],
            kernel_size=3,
            stride=1,
        ),
        # reg_conv0
        Conv(
            in_channels=channels_list[6],
            out_channels=channels_list[6],
            kernel_size=3,
            stride=1,
        ),
        # cls_pred0
        nn.Conv2d(
            in_channels=channels_list[6],
            out_channels=num_classes * num_anchors,
            kernel_size=1,
        ),
        # reg_pred0
        nn.Conv2d(
            in_channels=channels_list[6], out_channels=4 * num_anchors, kernel_size=1
        ),
        # obj_pred0
        nn.Conv2d(
            in_channels=channels_list[6], out_channels=1 * num_anchors, kernel_size=1
        ),
        # stem1
        Conv(
            in_channels=channels_list[8],
            out_channels=channels_list[8],
            kernel_size=1,
            stride=1,
        ),
        # cls_conv1
        Conv(
            in_channels=channels_list[8],
            out_channels=channels_list[8],
            kernel_size=3,
            stride=1,
        ),
        # reg_conv1
        Conv(
            in_channels=channels_list[8],
            out_channels=channels_list[8],
            kernel_size=3,
            stride=1,
        ),
        # cls_pred1
        nn.Conv2d(
            in_channels=channels_list[8],
            out_channels=num_classes * num_anchors,
            kernel_size=1,
        ),
        # reg_pred1
        nn.Conv2d(
            in_channels=channels_list[8], out_channels=4 * num_anchors, kernel_size=1
        ),
        # obj_pred1
        nn.Conv2d(
            in_channels=channels_list[8], out_channels=1 * num_anchors, kernel_size=1
        ),
        # stem2
        Conv(
            in_channels=channels_list[10],
            out_channels=channels_list[10],
            kernel_size=1,
            stride=1,
        ),
        # cls_conv2
        Conv(
            in_channels=channels_list[10],
            out_channels=channels_list[10],
            kernel_size=3,
            stride=1,
        ),
        # reg_conv2
        Conv(
            in_channels=channels_list[10],
            out_channels=channels_list[10],
            kernel_size=3,
            stride=1,
        ),
        # cls_pred2
        nn.Conv2d(
            in_channels=channels_list[10],
            out_channels=num_classes * num_anchors,
            kernel_size=1,
        ),
        # reg_pred2
        nn.Conv2d(
            in_channels=channels_list[10], out_channels=4 * num_anchors, kernel_size=1
        ),
        # obj_pred2
        nn.Conv2d(
            in_channels=channels_list[10], out_channels=1 * num_anchors, kernel_size=1
        ),
    )
    return head_layers

# Modifications copyright 2022 AI Singapore
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
#
# Original copyright 2021 Megvii, Base Detection
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Backbone
1. Wonder if should remove assert or change args.
"""

from __future__ import annotations
from typing import Tuple, Optional, List

import torch
from torch import nn
from peekingduck.pipeline.nodes.model.yolov6_fam.yolov6_files.layers.common import (
    RepBlock,
    RepVGGBlock,
    SimSPPF,
)


# changing name will cause weights name mismatch.
class EfficientRep(nn.Module):
    """EfficientRep Backbone
    EfficientRep is handcrafted by hardware-aware neural network design.
    With rep-style struct, EfficientRep is friendly to high-computation hardware(e.g. GPU).
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels_list: Optional[List[int]] = None,
        num_repeats: Optional[List[int]] = None,
    ) -> None:
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None

        self.stem = RepVGGBlock(
            in_channels=in_channels,
            out_channels=channels_list[0],
            kernel_size=3,
            stride=2,
        )

        # pylint: disable=invalid-name
        self.ERBlock_2 = nn.Sequential(
            RepVGGBlock(
                in_channels=channels_list[0],
                out_channels=channels_list[1],
                kernel_size=3,
                stride=2,
            ),
            RepBlock(
                in_channels=channels_list[1],
                out_channels=channels_list[1],
                n=num_repeats[1],
                block=RepVGGBlock,
            ),
        )

        self.ERBlock_3 = nn.Sequential(
            RepVGGBlock(
                in_channels=channels_list[1],
                out_channels=channels_list[2],
                kernel_size=3,
                stride=2,
            ),
            RepBlock(
                in_channels=channels_list[2],
                out_channels=channels_list[2],
                n=num_repeats[2],
                block=RepVGGBlock,
            ),
        )

        self.ERBlock_4 = nn.Sequential(
            RepVGGBlock(
                in_channels=channels_list[2],
                out_channels=channels_list[3],
                kernel_size=3,
                stride=2,
            ),
            RepBlock(
                in_channels=channels_list[3],
                out_channels=channels_list[3],
                n=num_repeats[3],
                block=RepVGGBlock,
            ),
        )

        self.ERBlock_5 = nn.Sequential(
            RepVGGBlock(
                in_channels=channels_list[3],
                out_channels=channels_list[4],
                kernel_size=3,
                stride=2,
            ),
            RepBlock(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                n=num_repeats[4],
                block=RepVGGBlock,
            ),
            SimSPPF(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                kernel_size=5,
            ),
        )

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Forward pass of EfficientRep, the backbone of YOLOv6.

        Args:
            inputs (torch.Tensor): The input tensor of shape [batch_size, 3, height, width].

        Returns:
            Tuple[torch.Tensor]: A tuple of tensors combining [stem+ERBlock_2+ERBlock_3,
                ERBlock_4, ERBlock_5].
        """
        outputs = []

        inputs = self.stem(inputs)
        inputs = self.ERBlock_2(inputs)
        inputs = self.ERBlock_3(inputs)
        outputs.append(inputs)

        inputs = self.ERBlock_4(inputs)
        outputs.append(inputs)

        inputs = self.ERBlock_5(inputs)
        outputs.append(inputs)

        return tuple(outputs)

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

"""Common network blocks and functions.

1. SiLU is added to PyTorch so do not need self define.
2. Removed a few unncessary else after return.
3. Replaced all x with inputs accordingly;
4. Removed DetectBackend and other functions that are not used
5. https://stackoverflow.com/questions/1120927/which-is-better-in-python-del-or-delattr
6. SimConv and Conv should be unified as it duplicated everything except activation.
"""

from __future__ import annotations
import warnings

from typing import Tuple, Union, no_type_check, Optional
import numpy as np
import torch
from torch import nn


# pylint: disable=too-many-arguments
def conv_bn(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    groups: int = 1,
) -> nn.Sequential:
    """Basic cell for rep-style block, including conv and bn."""
    result = nn.Sequential()
    result.add_module(
        "conv",
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        ),
    )
    result.add_module("bn", nn.BatchNorm2d(num_features=out_channels))
    return result


# pylint: disable=invalid-name
class Conv(nn.Module):
    """Normal Conv with SiLU activation"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass of Conv Block.

        Args:
            inputs (torch.Tensor): Inputs to the Conv Block.
        """
        return self.act(self.bn(self.conv(inputs)))

    def forward_fuse(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass of Conv Block when bn and conv layers are fused.

        Args:
            inputs (torch.Tensor): Inputs to the Conv Block.
        """

        return self.act(self.conv(inputs))


class SimConv(nn.Module):
    """Normal Conv with ReLU activation"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass of SimConv Block.

        Args:
            inputs (torch.Tensor): The inputs to the SimConv Block.
        """
        return self.act(self.bn(self.conv(inputs)))

    def forward_fuse(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass of SimConv Block when bn and conv layers are fused.

        Args:
            inputs (torch.Tensor): The inputs to the SimConv Block.
        """
        return self.act(self.conv(inputs))


# pylint: disable=invalid-name
class SimSPPF(nn.Module):
    """Simplified SPPF with ReLU activation"""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 5
    ) -> None:
        super().__init__()
        hidden_channels = in_channels // 2  # hidden channels
        self.cv1 = SimConv(in_channels, hidden_channels, 1, 1)
        self.cv2 = SimConv(hidden_channels * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass of SimSPPF Block.

        Args:
            inputs (torch.Tensor): The inputs to the SimSPPF Block.
        """
        inputs = self.cv1(inputs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            maxpool_out_1 = self.m(inputs)
            maxpool_out_2 = self.m(maxpool_out_1)
            return self.cv2(
                torch.cat(
                    [inputs, maxpool_out_1, maxpool_out_2, self.m(maxpool_out_2)], 1
                )
            )


class Transpose(nn.Module):
    """Normal Transpose, default for upsampling"""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 2, stride: int = 2
    ) -> None:
        super().__init__()
        self.upsample_transpose = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=True,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """The forward pass of Transpose Block.

        Args:
            inputs (torch.Tensor): The inputs to the Transpose Block.
        """
        return self.upsample_transpose(inputs)


# pylint: disable=too-many-instance-attributes
class RepVGGBlock(nn.Module):
    """RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        padding_mode: str = "zeros",
        deploy: bool = False,
    ) -> None:
        """Initialization of the class.
        Args:
            in_channels (int): Number of channels in the inputs image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of
                the inputs. Default: 1
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from inputs
                channels to output channels. Default: 1
            padding_mode (string, optional): Default: 'zeros'
            deploy: Whether to be deploy status or training status. Default: False
        """

        super().__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
                padding_mode=padding_mode,
            )
        else:
            self.rbr_identity = (
                nn.BatchNorm2d(num_features=in_channels)
                if out_channels == in_channels and stride == 1
                else None
            )
            self.rbr_dense = conv_bn(
                in_channels, out_channels, kernel_size, stride, padding, groups
            )
            self.rbr_1x1 = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=padding_11,
                groups=groups,
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass of RepVGGBlock Block.

        Args:
            inputs (torch.Tensor): The inputs to the RepVGGBlock Block.
        """
        if hasattr(self, "rbr_reparam"):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(
            self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)
        )

    def get_equivalent_kernel_bias(
        self,
    ) -> Tuple[Union[torch.Tensor, int], Union[torch.Tensor, int]]:
        """Get equivalent kernel and bias for the block during deploy mode.

        Returns:
            Tuple[Union[torch.Tensor, int], Union[torch.Tensor, int]]: kernel and bias.
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self.pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    @staticmethod
    def pad_1x1_to_3x3_tensor(
        kernel1x1: Optional[Union[torch.Tensor, int]]
    ) -> Union[torch.Tensor, int]:
        """Pad 1x1 kernel to 3x3 kernel.

        Args:
            kernel1x1 (torch.Tensor): The 1x1 kernel.

        Returns:
            Union[torch.Tensor, int]: The 3x3 kernel.
        """
        if kernel1x1 is None:
            return 0

        # mypy error maybe due to it expects tensor...
        return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])  # type: ignore

    @no_type_check
    def _fuse_bn_tensor(
        self, branch: Union[nn.Sequential, nn.BatchNorm2d, None]
    ) -> Union[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]]:
        """Fuse the batch norm tensor.

        Args:
            branch (Union[nn.Sequential, nn.BatchNorm2d, None]):
                A branch type of the block: nn.Sequential or nn.BatchNorm2d.

        Returns:
            Union[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]]: Returns
            (0, 0) if branch is None, otherwise returns the fused kernel and bias.
        """
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    @no_type_check
    def switch_to_deploy(self) -> None:
        """Switch layers to deploy mode."""
        if hasattr(self, "rbr_reparam"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense.conv.in_channels,
            out_channels=self.rbr_dense.conv.out_channels,
            kernel_size=self.rbr_dense.conv.kernel_size,
            stride=self.rbr_dense.conv.stride,
            padding=self.rbr_dense.conv.padding,
            dilation=self.rbr_dense.conv.dilation,
            groups=self.rbr_dense.conv.groups,
            bias=True,
        )
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()

        del self.rbr_dense
        del self.rbr_1x1

        if hasattr(self, "rbr_identity"):
            del self.rbr_identity
        if hasattr(self, "id_tensor"):
            del self.id_tensor
        self.deploy = True


class RepBlock(nn.Module):
    """
    RepBlock is a stage block with rep-style basic block
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n: int = 1,
        block: type[RepVGGBlock] = RepVGGBlock,
    ):
        super().__init__()
        self.conv1 = block(in_channels, out_channels)
        self.block = (
            nn.Sequential(*(block(out_channels, out_channels) for _ in range(n - 1)))
            if n > 1
            else None
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Foward pass of RepBlock

        Args:
            inputs (torch.Tensor): The inputs to the RepBlock.
        """
        outputs = self.conv1(inputs)
        if self.block is not None:
            outputs = self.block(outputs)
        return outputs

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

"""Config"""
from dataclasses import field, dataclass
from typing import Any, Dict, Union, Type

# pylint: disable=too-few-public-methods
class RecursiveNamespace:  # without extending SimpleNamespace!
    """Extend SimpleNamespace to allow recursive instantiation."""

    # https://dev.to/taqkarim/extending-simplenamespace-for-nested-dictionaries-58e8
    @staticmethod
    def map_entry(entry: Any) -> Union[Any, Type["RecursiveNamespace"]]:
        """Map entry to RecursiveNamespace if necessary."""
        if isinstance(entry, dict):
            return RecursiveNamespace(**entry)
        return entry

    def __init__(self, **kwargs: Dict[str, Any]):  # type: ignore
        for key, val in kwargs.items():
            if isinstance(val, dict):
                setattr(self, key, RecursiveNamespace(**val))
            elif isinstance(val, list):
                setattr(self, key, list(map(self.map_entry, val)))
            else:  # this is the only addition
                setattr(self, key, val)


@dataclass(init=True, frozen=False)
class ModelParams:
    """YOLOv6 params from original repo.

    Link: https://github.com/meituan/YOLOv6/blob/main/configs/yolov6n.py
    """

    model_type: str
    num_channels = 3

    # for now I put what is the default in common.py's get_block.
    # check if this is needed in non-training mode?
    training_mode = "repvgg"

    model: RecursiveNamespace = field(init=False)

    def __post_init__(self) -> None:
        """Initialize model params by model type passed in from config."""
        if self.model_type == "yolov6n":

            # YOLOv6n model
            model = dict(
                type="YOLOv6n",
                pretrained=None,
                depth_multiple=0.33,
                width_multiple=0.25,
                backbone=dict(
                    type="EfficientRep",
                    num_repeats=[1, 6, 12, 18, 6],
                    out_channels=[64, 128, 256, 512, 1024],
                ),
                neck=dict(
                    type="RepPAN",
                    num_repeats=[12, 12, 12, 12],
                    out_channels=[256, 128, 128, 256, 256, 512],
                ),
                head=dict(
                    type="EffiDeHead",
                    in_channels=[128, 256, 512],
                    num_layers=3,
                    begin_indices=24,
                    anchors=1,
                    out_indices=[17, 20, 23],
                    strides=[8, 16, 32],
                    iou_type="ciou",
                ),
            )
        elif self.model_type == "yolov6s":
            model = dict(
                type="YOLOv6s",
                pretrained=None,
                depth_multiple=0.33,
                width_multiple=0.50,
                backbone=dict(
                    type="EfficientRep",
                    num_repeats=[1, 6, 12, 18, 6],
                    out_channels=[64, 128, 256, 512, 1024],
                ),
                neck=dict(
                    type="RepPAN",
                    num_repeats=[12, 12, 12, 12],
                    out_channels=[256, 128, 128, 256, 256, 512],
                ),
                head=dict(
                    type="EffiDeHead",
                    in_channels=[128, 256, 512],
                    num_layers=3,
                    begin_indices=24,
                    anchors=1,
                    out_indices=[17, 20, 23],
                    strides=[8, 16, 32],
                    iou_type="ciou",
                ),
            )
        elif self.model_type == "yolov6t":
            model = dict(
                type="YOLOv6t",
                pretrained=None,
                depth_multiple=0.25,
                width_multiple=0.50,
                backbone=dict(
                    type="EfficientRep",
                    num_repeats=[1, 6, 12, 18, 6],
                    out_channels=[64, 128, 256, 512, 1024],
                ),
                neck=dict(
                    type="RepPAN",
                    num_repeats=[12, 12, 12, 12],
                    out_channels=[256, 128, 128, 256, 256, 512],
                ),
                head=dict(
                    type="EffiDeHead",
                    in_channels=[128, 256, 512],
                    num_layers=3,
                    begin_indices=24,
                    anchors=1,
                    out_indices=[17, 20, 23],
                    strides=[8, 16, 32],
                    iou_type="ciou",
                ),
            )
        # model_params_namespace; not good to overwrite, quick fix for now
        self.model = RecursiveNamespace(**model)  # type: ignore

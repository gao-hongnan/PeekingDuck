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
1. Change prediction to predictions to consistency.
2. Changed x to prediction.
3. Changed torch.tensor(classes, device=predictions.device) to clone().detach().
https://stackoverflow.com/questions/55266154/pytorch-preferred-way-to-copy-a-tensor
"""

import time
from typing import List

import cv2
import numpy as np
import torch
import torchvision
from peekingduck.pipeline.utils.bbox.transforms import xywh2xyxy

# Settings
torch.set_printoptions(linewidth=320, precision=5, profile="long")

# format short g, %precision=5
np.set_printoptions(linewidth=320, formatter={"float_kind": "{:11.5g}".format})

# prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
cv2.setNumThreads(0)  # add this to whitelist


# # TODO: to use my own repo's central config way to call configs.
# @dataclass(init=True, frozen=True)
# class NMS:
#     """NMS configs."""

#     max_wh = 4096  # maximum box width and height
#     max_nms = 30000  # maximum number of boxes put into torchvision.ops.nms()
#     time_limit = 10.0  # quit the function when nms cost time exceed the limit time.
#     score_threshold: float
#     iou_thres: float
#     detect: List[int]
#     agnostic: bool
#     multi_label: bool
#     max_det: int


# pylint: disable=too-many-arguments, too-many-locals
def non_max_suppression(
    predictions: torch.Tensor,
    score_threshold: float = 0.25,
    iou_thres: float = 0.45,
    classes: torch.Tensor = None,
    agnostic: bool = False,
    multi_label: bool = False,
    max_det: int = 300,
) -> List[torch.Tensor]:
    """Runs Non-Maximum Suppression (NMS) on inference results.
    This code is borrowed from: https://github.com/ultralytics/yolov5/blob/
    47233e1698b89fc437a4fb9463c815e9171be955/utils/general.py#L775

    Args:
        predictions: (torch.Tensor), with shape [N, 5 + num_classes], N is the number of bboxes.
        score_threshold: (float) confidence threshold.
        iou_thres: (float) iou threshold.
        classes: (torch.Tensor), if a list is provided, nms only keep the classes you
         provide.
        agnostic: (bool), when it is set to True, we do class-independent nms, otherwise,
         different class would do nms respectively.
        multi_label: (bool), when it is set to True, one box can have multi labels,
         otherwise, one box only huave one label.
        max_det:(int), max number of output bboxes.

    Returns:
        output (List[torch.Tensor]): list of detections, echo item is one tensor
                with shape (num_boxes, 6), 6 is for [xyxy, conf, cls].
    """

    num_classes = predictions.shape[2] - 5  # number of classes
    pred_candidates = predictions[..., 4] > score_threshold  # candidates

    # Check the parameters.
    assert (
        0 <= score_threshold <= 1
    ), f"score_thresholdh must be in 0.0 to 1.0, however {score_threshold} is provided."
    assert (
        0 <= iou_thres <= 1
    ), f"iou_thres must be in 0.0 to 1.0, however {iou_thres} is provided."

    # Function settings.
    max_wh = 4096  # maximum box width and height
    max_nms = 30000  # maximum number of boxes put into torchvision.ops.nms()
    time_limit = 10.0  # quit the function when nms cost time exceed the limit time.
    multi_label &= num_classes > 1  # multiple labels per box

    tik = time.time()
    output = [torch.zeros((0, 6), device=predictions.device)] * predictions.shape[0]
    for img_idx, prediction in enumerate(predictions):  # image index, image inference
        prediction = prediction[pred_candidates[img_idx]]  # confidence

        # If no box remains, skip the next process.
        if not prediction.shape[0]:
            continue

        # confidence multiply the objectness
        prediction[:, 5:] *= prediction[:, 4:5]  # conf = obj_conf * cls_conf

        # (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(prediction[:, :4])

        # Detections matrix's shape is  (n,6), each row represents (xyxy, conf, cls)
        if multi_label:
            box_idx, class_idx = (
                (prediction[:, 5:] > score_threshold).nonzero(as_tuple=False).T
            )
            prediction = torch.cat(
                (
                    box[box_idx],
                    prediction[box_idx, class_idx + 5, None],
                    class_idx[:, None].float(),
                ),
                1,
            )
        else:  # Only keep the class with highest scores.
            conf, class_idx = prediction[:, 5:].max(1, keepdim=True)
            prediction = torch.cat((box, conf, class_idx.float()), 1)[
                conf.view(-1) > score_threshold
            ]

        # Filter by class, only keep boxes whose category is in classes.
        if classes is not None:
            prediction = prediction[
                (prediction[:, 5:6] == classes.clone().detach()).any(1)
            ]

        # Check shape
        num_box = prediction.shape[0]  # number of boxes
        if not num_box:  # no boxes kept.
            continue

        if num_box > max_nms:  # excess max boxes' number.
            # sort by confidence
            prediction = prediction[prediction[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        class_offset = prediction[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = (
            prediction[:, :4] + class_offset,
            prediction[:, 4],
        )  # boxes (offset by class), scores
        keep_box_idx = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if keep_box_idx.shape[0] > max_det:  # limit detections
            keep_box_idx = keep_box_idx[:max_det]

        output[img_idx] = prediction[keep_box_idx]
        if (time.time() - tik) > time_limit:
            print(f"WARNING: NMS cost time exceed the limited {time_limit}s.")
            break  # time limit exceeded

    return output

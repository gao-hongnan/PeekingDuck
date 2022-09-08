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

"""Detector module to predict object bbox from an image using YOLOv6.

Changes:
1. Removed redundant stuff such as tensorrt checks for now;
2. Moved NUM_CHANNELS to config.
3. Added model_switch from YOLOv6 to switch layers to deploy mode.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from peekingduck.pipeline.nodes.model.yolov6_fam.yolov6_files.configs.config import (
    ModelParams,
)
from peekingduck.pipeline.nodes.model.yolov6_fam.yolov6_files.data_utils.data_augment import (
    letterbox,
)
from peekingduck.pipeline.nodes.model.yolov6_fam.yolov6_files.layers.common import (
    RepVGGBlock,
)
from peekingduck.pipeline.nodes.model.yolov6_fam.yolov6_files.models.yolo import YOLOv6
from peekingduck.pipeline.nodes.model.yolov6_fam.yolov6_files.utils.nms import (
    non_max_suppression,
)
from peekingduck.pipeline.nodes.model.yolov6_fam.yolov6_files.utils.torch_utils import (
    fuse_model,
)
from peekingduck.pipeline.utils.bbox.transforms import xyxy2xyxyn


class Detector:  # pylint: disable=too-many-instance-attributes
    """Object detection class using YOLOv6 to predict object bboxes.

    Attributes:
        logger (logging.Logger): Events logger.
        config (Dict[str, Any]): YOLOv6 node configuration.
        model_dir (pathlib.Path): Path to directory of model weights files.
        device (torch.device): Represents the device on which the torch.Tensor
            will be allocated.
        half (bool): Flag to determine if half-precision should be used.
        yolov6 (YOLOv6): The YOLOv6 model for performing inference.
    """

    def __init__(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        model_dir: Path,
        class_names: List[str],
        detect_ids: List[int],
        model_format: str,
        model_type: str,
        num_classes: int,
        model_file: Dict[str, str],
        agnostic_nms: bool,
        fuse: bool,
        half: bool,
        input_size: int,
        iou_threshold: float,
        score_threshold: float,
        multi_label: bool,
        max_detections: int,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_params: type[ModelParams] = ModelParams(model_type)  # type: ignore

        self.max_detections = max_detections
        self.multi_label = multi_label

        self.class_names = class_names
        self.model_format = model_format
        self.model_type = model_type

        self.num_classes = num_classes

        self.model_path = model_dir / model_file[self.model_type]
        self.agnostic_nms = agnostic_nms
        self.fuse = fuse

        # Half-precision only supported on CUDA
        self.half = half and self.device.type == "cuda"
        self.input_size = (input_size, input_size)
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold

        self.update_detect_ids(detect_ids)

        self.yolov6 = self._create_yolov6_model()

        # 1. need to see DetectBackend to change the code below;
        # 2. define it under init to pass type checks.
        self.stride: int = int(self.yolov6.stride.max())

    @torch.no_grad()
    def predict_object_bbox_from_image(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detects bounding boxes of selected object categories from an image.

        The input image is first scaled according to the `input_size`
        configuration option. Detection results will be filtered according to
        `iou_threshold`, `score_threshold`, and `detect_ids` configuration
        options. Bounding boxes coordinates are then normalized w.r.t. the
        input `image` size.

        Args:
            image (np.ndarray): Input image.

        Returns:
            (Tuple[np.ndarray, np.ndarray, np.ndarray]): Returned tuple
            contains:
            - An array of detection bboxes
            - An array of human-friendly detection class names
            - An array of detection scores
        """
        # Store the original image size to normalize bbox later
        original_image_shape = image.shape

        # modify preprocess and removed scale
        preprocessed_image = self._preprocess(image)

        preprocessed_image_shape = preprocessed_image.shape  # without bs

        preprocessed_image = preprocessed_image.unsqueeze(0).to(self.device)
        preprocessed_image = (
            preprocessed_image.half() if self.half else preprocessed_image.float()
        )
        prediction = self.yolov6(preprocessed_image)

        # modified accordingly
        bboxes, classes, scores = self._postprocess(
            prediction, original_image_shape, preprocessed_image_shape
        )

        return bboxes, classes, scores

    def update_detect_ids(self, ids: List[int]) -> None:
        """Updates list of selected object category IDs. When the list is
        empty, all available object category IDs are detected.

        Args:
            ids: List of selected object category IDs
        """
        self.detect_ids = torch.Tensor(ids).to(self.device)  # type: ignore
        if self.half:
            self.detect_ids = self.detect_ids.half()

    def _create_yolov6_model(self) -> YOLOv6:
        """Creates a YOLOv6 model and loads its weights.

        Creates `detect_ids` as a `torch.Tensor`. Sets up `input_size` to a
        square shape. Logs model configurations.

        Returns:
            (YOLOv6): YOLOv6 model.
        """
        self.logger.info(
            "YOLOv6 model loaded with the following configs:\n\t"
            f"Model format: {self.model_format}\n\t"
            f"Model type: {self.model_type}\n\t"
            f"Input resolution: {self.input_size}\n\t"
            f"IDs being detected: {self.detect_ids.int().tolist()}\n\t"
            f"IOU threshold: {self.iou_threshold}\n\t"
            f"Score threshold: {self.score_threshold}\n\t"
            f"Class agnostic NMS: {self.agnostic_nms}\n\t"
            f"Half-precision floating-point: {self.half}\n\t"
            f"Fuse convolution and batch normalization layers: {self.fuse}"
        )
        return self._load_yolov6_weights()

    def _get_model(self) -> YOLOv6:
        """Constructs YOLOv6 model based on parsed configuration.

        Returns:
            (YOLOv6): YOLOv6 model.
        """
        # check why anchors is not directly unpacked from config?

        return YOLOv6(
            self.model_params,
            self.num_classes,
            self.model_params.model.head.anchors,
        )

    @staticmethod
    def model_switch(model: YOLOv6) -> None:
        """Model switch to deploy status"""
        for layer in model.modules():
            if isinstance(layer, RepVGGBlock):
                layer.switch_to_deploy()

    def _load_yolov6_weights(self) -> YOLOv6:
        """Loads YOLOv6 model weights.

        Returns:
            model (YOLOv6): YOLOv6 model.

        Raises:
            ValueError: `model_path` does not exist.
        """
        if self.model_path.is_file():
            ckpt = torch.load(str(self.model_path), map_location=self.device)
            model = self._get_model().to(self.device).float()
            model.load_state_dict(ckpt)

            if self.half:
                model.half()

            if self.fuse:
                model = fuse_model(model)

            if self.device.type != "cpu":
                # remove model.model because our model is not DetectBackend
                model(
                    torch.zeros(1, 3, *self.input_size)
                    .to(self.device)
                    .type_as(next(model.parameters()))
                )  # warmup

            # switch model to deploy status
            self.model_switch(model)

            model.eval()

            return model

        raise ValueError(
            f"Model file does not exist. Please check that {self.model_path} exists."
        )

    @staticmethod
    def rescale(
        boxes: torch.Tensor,
        original_image_shape: Tuple[int, ...],
        preprocessed_image_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """Rescale the BBOX output to the original image shape.
        # consider change var/arg name as it was confusing;
        since reshaping to original image shape makes me think its reshape
        to ori_shape.

        Using image 000000000139 from coco/images/val2017 as an example:
        - original image shape: (C, H, W) = (3, 426, 640) coincidentally one side is alr 640
        - preprocessed image shape: (C, H, W) = (3, 448, 640) see preprocess fn;
        - original_image_shape: [426, 640]
        - preprocessed_image_shape: [448, 640]
        - We just want to rescale the bbox on the preprocessed image to original image.
        """
        preprocessed_image_shape = preprocessed_image_shape[1:]
        ratio = min(
            preprocessed_image_shape[0] / original_image_shape[0],
            preprocessed_image_shape[1] / original_image_shape[1],
        )
        padding = (preprocessed_image_shape[1] - original_image_shape[1] * ratio) / 2, (
            preprocessed_image_shape[0] - original_image_shape[0] * ratio
        ) / 2

        boxes[:, [0, 2]] -= padding[0]
        boxes[:, [1, 3]] -= padding[1]
        boxes[:, :4] /= ratio

        boxes[:, 0].clamp_(0, original_image_shape[1])  # x1
        boxes[:, 1].clamp_(0, original_image_shape[0])  # y1
        boxes[:, 2].clamp_(0, original_image_shape[1])  # x2
        boxes[:, 3].clamp_(0, original_image_shape[0])  # y2

        return boxes

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for inference.

        Reference:
            YOLOv6.yolov6.core.inferer's `precess_image`.
        """
        image = letterbox(image, self.input_size, stride=self.stride)[0]
        # Convert
        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB

        # Numpy to Tensor
        image_tensor = torch.from_numpy(np.ascontiguousarray(image))
        # uint8 to fp16/32
        image_tensor = image_tensor.half() if self.half else image_tensor.float()
        image_tensor /= 255  # 0 - 255 to 0.0 - 1.0

        return image_tensor

    def _postprocess(
        self,
        prediction: torch.Tensor,
        original_image_shape: Tuple[int, ...],
        preprocessed_image_shape: Tuple[int, ...],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Postprocess image after inference.

        Args:
            prediction (torch.Tensor): The prediction tensor.
            original_image_shape (Tuple[int, ...]): The original image shape.
            preprocessed_image_shape (Tuple[int, ...]): The preprocessed image shape.

        Returns:
            bboxes, classes, scores (Tuple[np.ndarray, np.ndarray, np.ndarray]):
                The postprocessed bboxes, classes, scores.

        Reference:
            YOLOv6.yolov6.core.inferer's.
        """
        det = non_max_suppression(
            prediction,
            self.score_threshold,
            self.iou_threshold,
            self.detect_ids,
            self.agnostic_nms,
            self.multi_label,
            self.max_detections,
        )[0]

        # follow pkd's convention: early return if all are below score_threshold
        if not det.size(0):
            return np.empty((0, 4)), np.empty(0), np.empty(0)

        # print(original_image_shape, preprocessed_image_shape[1:]) -> (540, 710) [3, 512]

        # shaping shenenigans to be adhered and .round() needed for floating point rounding.
        det[:, :4] = self.rescale(
            boxes=det[:, :4],
            original_image_shape=original_image_shape,
            preprocessed_image_shape=preprocessed_image_shape,
        ).round()

        print("scaled bboxes", det[:, :4])
        # follow pkd convention to get bboxes etc
        output_np = det.cpu().detach().numpy()
        bboxes = xyxy2xyxyn(output_np[:, :4], *original_image_shape[:2])

        scores = output_np[:, 4]  # * output_np[:, 5]

        # to check why the indices is 5 not 6 : error prompted in output_np[:, 6]
        classes = np.array([self.class_names[int(i)] for i in output_np[:, 5]])
        return bboxes, classes, scores

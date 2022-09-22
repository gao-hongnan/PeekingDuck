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

"""Deleted irrelevant functions:
- augment_hsv
- mixup
- box_candidates
- random_affine
- get_transform_matrix
- mosaic_augmentation

Read up on https://medium.com/styria-data-science-tech-blog/using-yolo-algorithms-
to-conquer-the-global-data-science-challenge-bee7793b0e54
on letterbox.
"""

from typing import Tuple

import cv2
import numpy as np


# pylint: disable=too-many-arguments
def letterbox(
    image: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = True,
    scaleup: bool = True,
    stride: int = 32,
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """A type of resizing technique that preserves the aspect ratio of the original image.

    Args:
        image (np.ndarray): The image to be resized.
        new_shape (Tuple[int, int], optional): The new shape to resize to. Defaults to (640, 640).
        color (Tuple[int, int, int], optional): ImageNet mean to fill border.
                                                Defaults to (114, 114, 114).
        auto (bool, optional): Perform modulus on stride to get min rec.
                               Defaults to True.
        scaleup (bool, optional): Controls the scale up of the image. Defaults to True.
        stride (int, optional): The stride to take when auto is True. Defaults to 32.

    Returns:
        Tuple[np.ndarray, float, Tuple[float, float]]: _description_
    """
    # Resize and pad image while meeting stride-multiple constraints
    shape = image.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        ratio = min(ratio, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * ratio)), int(round(shape[0] * ratio))
    width_diff, height_diff = float(new_shape[1] - new_unpad[0]), float(
        new_shape[0] - new_unpad[1]
    )  # wh padding

    if auto:  # minimum rectangle
        width_diff, height_diff = np.mod(width_diff, stride), np.mod(
            height_diff, stride
        )  # wh padding

    width_diff /= 2  # divide padding into 2 sides
    height_diff /= 2

    if shape[::-1] != new_unpad:  # resize
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(height_diff - 0.1)), int(round(height_diff + 0.1))
    left, right = int(round(width_diff - 0.1)), int(round(width_diff + 0.1))
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border

    return image, ratio, (width_diff, height_diff)

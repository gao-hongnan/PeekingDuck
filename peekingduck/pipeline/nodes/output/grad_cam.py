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
Shows the outputs on your display.
"""

from typing import Any, Dict, Union

import cv2
import numpy as np

from peekingduck.pipeline.nodes.abstract_node import AbstractNode


class Node(AbstractNode):
    """Streams the output on your display.

    Inputs:
        |img_data|

        |filename_data|

    Outputs:
        |pipeline_end_data|

    Configs:
        window_name (:obj:`str`): **default = "PeekingDuck"** |br|
            Name of the displayed window.
        window_size (:obj:`Dict[str, Union[bool, int]]`):
            **default = { do_resizing: False, width: 1280, height: 720 }** |br|
            Resizes the displayed window to the chosen width and weight, if
            ``do_resizing`` is set to ``true``. The size of the displayed
            window can also be adjusted by clicking and dragging.
        window_loc (:obj:`Dict[str, int]`): **default = { x: 0, y: 0 }** |br|
            X and Y coordinates of the top left corner of the displayed window,
            with reference from the top left corner of the screen, in pixels.

    .. note::

        **See Also:**

        :ref:`PeekingDuck Viewer<pkd_viewer>`: a GUI for running PeekingDuck pipelines.

        .. figure:: /assets/diagrams/viewer_cat_computer.png

        The PeekingDuck Viewer offers a GUI to view and analyze pipeline output.
        It has controls to re-play output video, scrub to a frame of interest,
        zoom video, and a playlist for managing multiple pipelines.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.first_run = True

    # TODO: https://stackoverflow.com/questions/34966541/how-can-one-display-an-image-using-cv2-in-python

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Show the outputs on your display"""
        img = inputs["img"]
        gradcam_img = inputs["gradcam_image"]
        img = cv2.resize(img, gradcam_img.shape[1::-1])
        print(img.shape, gradcam_img.shape)
        concat_horizontal = cv2.hconcat([img, gradcam_img])
        print(concat_horizontal.shape)
        if self.window_size["do_resizing"]:
            concat_horizontal = cv2.resize(
                concat_horizontal,
                (self.window_size["height"], self.window_size["width"]),
            )

        cv2.imshow(self.window_name, concat_horizontal)
        cv2.waitKey(0)
        # cv2.imshow(self.window_name, gradcam_img)
        # if self.first_run:
        #     cv2.moveWindow(self.window_name, self.window_loc["x"], self.window_loc["y"])
        #     self.first_run = False
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyWindow(self.window_name)
            return {"pipeline_end": True}

        return {"pipeline_end": False}

    # def _get_config_types(self) -> Dict[str, Any]:
    #     """Returns dictionary mapping the node's config keys to respective types."""


def show_gradcam(image):
    pass


def show_resnet_gradcam(
    self,
    model,
    image: np.ndarray,
    original_image: np.ndarray,
    plot_gradcam: bool = True,
) -> np.ndarray:
    """Show the gradcam of the image for ResNet variants.
    This will not work on other types of network architectures.
    Args:
        image (np.ndarray): The input image.
        original_image (np.ndarray): The original image.
        plot_gradcam (bool): If True, will plot the gradcam. Defaults to True.
    Returns:
        gradcam_image (np.ndarray): The gradcam image.
    """
    # model = self.resnet50d
    # only for resnet variants! for simplicity we don't enumerate other models!
    # target_layers = [model.model.layer4[-1]]  # [model.backbone.layer4[-1]]
    target_layers = [model.layer4[-1]]
    reshape_transform = None

    # input is np array so turn to tensor as well as resize aand as well as turn to tensor by totensorv2
    # so no need permute since albu does it
    image = self.transforms(image=image)["image"].to(self.device)

    # if image tensor is 3 dim, unsqueeze it to 4 dim with 1 in front.
    image = image.unsqueeze(0)
    gradcam = GradCAM(
        model=model,
        target_layers=target_layers,
        use_cuda=False,
        reshape_transform=reshape_transform,
    )

    # # If targets is None, the highest scoring category will be used for every image in the batch.
    gradcam_output = gradcam(
        input_tensor=image,
        target_category=None,
        aug_smooth=False,
        eigen_smooth=False,
    )
    original_image = original_image / 255.0

    gradcam_image = show_cam_on_image(original_image, gradcam_output[0], use_rgb=False)

    if plot_gradcam:
        _fig, axes = plt.subplots(figsize=(8, 8), ncols=2)
        axes[0].imshow(original_image)
        # axes[0].set_title(f"y_true={y_true:.4f}")
        axes[1].imshow(gradcam_image)
        # axes[1].set_title(f"y_pred={y_pred}")
        plt.show()
        torch.cuda.empty_cache()

    return gradcam_image

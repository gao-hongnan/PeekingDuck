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

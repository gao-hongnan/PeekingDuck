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
- start function name with test_ to make pytest recognize it as a test.
- @pytest.mark.mlmodel allows us to pytest -m "mlmodel" to run only tests that have the mlmodel tag.
See https://madewithml.com/courses/mlops/testing/#markers
- conftest.py: in https://docs.pytest.org/en/6.2.x/fixture.html's conftest.py section, it says
that fixtures defined in conftest.py are shared across multiple files, and do not need to be
imported as pytest automatically detects them.
    - Other ref: https://stackoverflow.com/questions/34466027/in-pytest-what-is-the-use-of-conftest-
    py-files
- Bug on printing t1.jpg's bboxes for ground truth, the last two rows of the outputs are permuted,
to debug it I tried to add print statements at every junction of the code where "output" is there,
but cannot find out why beside very small diff in image tensors.
    - UPDATE: checked with Yier and difference is due to cv2.imread vs cv2.VideoCapture.
### QNS
- Ask on the mock torch cuda
"""

from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import numpy.testing as npt
import pytest
import torch
import yaml

from peekingduck.pipeline.nodes.base import WeightsDownloaderMixin
from peekingduck.pipeline.nodes.model.yolov6 import Node
from tests.conftest import PKD_DIR, get_groundtruth

GT_RESULTS = get_groundtruth(Path(__file__).resolve())
CONFIG_PATH = PKD_DIR / "configs" / "model" / "yolov6.yml"


@pytest.fixture(scope="function")
def yolov6_config():
    """Fixture to load the config file for yolov6 node.

    Note:
        It returns a config dict and this function will be used in/across
        multiple tests, hence the usage of the decorator @pytest.fixture.
    """
    with open(CONFIG_PATH) as infile:
        node_config = yaml.safe_load(infile)
    node_config["root"] = Path.cwd()

    return node_config


@pytest.fixture(
    params=[
        {"key": "iou_threshold", "value": -0.5},
        {"key": "iou_threshold", "value": 1.5},
        {"key": "score_threshold", "value": -0.5},
        {"key": "score_threshold", "value": 1.5},
    ],
)
def yolov6_bad_config_value(request, yolov6_config):
    """Fixture to load the config file with bad values for yolov6 node.

    Note:
        Since there are 4 different combinations of bad values,
        whenever this fixture is called, it will return 4 different config dicts
        with different bad values.

    Args:
        request (Type["pytest.fixtures.SubRequest"]): Calling request.param
            yields the param in params in the @pytest.fixture decorator.
            Notice we have 4 params in the decorator, hence 4 tests will be
            run when this fixture is called.
            The combinations of the params are:
            - iou_threshold = -0.5;
            - iou_threshold = 1.5;
            - score_threshold = -0.5;
            - score_threshold = 1.5;

        yolov6_config (Dict[str, Any]): This argument comes from the earlier
            defined fixture yolov6_config() and is the config dict for yolov6.

    Returns:
        yolov6_config (Dict[str, Any]): This is the config dict for yolov6 with
            bad values populated.
    """
    yolov6_config[request.param["key"]] = request.param["value"]
    return yolov6_config


@pytest.fixture(
    params=[
        # {"agnostic_nms": True, "fuse": True, "half": True},
        {"agnostic_nms": True, "fuse": True, "half": False},
        # {"agnostic_nms": True, "fuse": False, "half": True},
        # {"agnostic_nms": True, "fuse": False, "half": False},
        # {"agnostic_nms": False, "fuse": True, "half": True},
        # {"agnostic_nms": False, "fuse": True, "half": False},
        # {"agnostic_nms": False, "fuse": False, "half": True},
        # {"agnostic_nms": False, "fuse": False, "half": False},
    ]
)
def yolov6_matrix_config(request, yolov6_config):
    """Similar to yolov6_bad_config_value, this fixture loads different config
    targeting other configurations.

    Note:
        Since there are 8 different combinations of values, whenever this
        fixture is called, it will return 8 different config dicts with different
        values.
    """

    yolov6_config.update(request.param)
    return yolov6_config


@pytest.fixture(params=["yolov6n", "yolov6t"])
def yolov6_config_cpu(request, yolov6_matrix_config):
    """Similar to the previous fixtures, this fixture mainly loads different model types.

    Note:
        This takes in the yolov6_matrix_config fixture as an argument and with
        2 params defined in the decorator, it will return 2 * 8 config dicts = 16
        different config dicts.

    Yields:
        Iterator[Dict[str, Any]]: The config dict for yolov6 with different model types.
    """
    yolov6_matrix_config["model_type"] = request.param
    with mock.patch("torch.cuda.is_available", return_value=False):
        yield yolov6_matrix_config


# The below print statement is for us to see what happens when we run tests
# with the above fixtures.


def test_yolov6_node_config(yolov6_config):
    print(f"fixture for yolov6_config={yolov6_config}")


def test_yolov6_bad_config_value(yolov6_bad_config_value):
    print(f"fixture for yolov6_bad_config_value={yolov6_bad_config_value}\n")


def test_yolov6_matrix_config(yolov6_matrix_config):
    print(f"fixture for yolov6_matrix_config={yolov6_matrix_config}\n")


def test_yolov6_config_cpu(yolov6_config_cpu):
    print(f"fixture for yolov6_config_cpu={yolov6_config_cpu}\n")


# To make debug easier, I only restrict matrix to 1 combo of params.
@pytest.mark.mlmodel
class TestYOLOv6:
    def test_no_human_image(self, no_human_image, yolov6_config_cpu):
        """Test the yolov6 node with no human image.

        Note:
            - This test will run with the yolov6_config_cpu fixture (i.e. no gpu);
            - no_human_image is a fixture defined in the conftest.py file and
            have 2 images;
            - Thus this test will run num of config in yolo6_config_cpu fixture
            multiply by 2 images (i.e. 8 * 2 = 16 configs).
        """

        no_human_img = cv2.imread(no_human_image)
        yolov6 = Node(yolov6_config_cpu)
        output = yolov6.run({"img": no_human_img})
        expected_output = {
            "bboxes": np.empty((0, 4), dtype=np.float32),
            "bbox_labels": np.empty((0)),
            "bbox_scores": np.empty((0), dtype=np.float32),
        }
        assert output.keys() == expected_output.keys()
        npt.assert_equal(output["bboxes"], expected_output["bboxes"])
        npt.assert_equal(output["bbox_labels"], expected_output["bbox_labels"])
        npt.assert_equal(output["bbox_scores"], expected_output["bbox_scores"])

    def test_detect_human_bboxes(self, human_image, yolov6_config_cpu):
        human_img = cv2.imread(human_image)

        yolov6 = Node(yolov6_config_cpu)
        output = yolov6.run({"img": human_img})

        assert "bboxes" in output
        assert output["bboxes"].size > 0

        model_type = yolov6.config["model_type"]
        image_name = Path(human_image).stem
        expected = GT_RESULTS[model_type][image_name]

        npt.assert_allclose(output["bboxes"], expected["bboxes"], atol=1e-3)
        npt.assert_equal(output["bbox_labels"], expected["bbox_labels"])
        npt.assert_allclose(output["bbox_scores"], expected["bbox_scores"], atol=1e-2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
    def test_detect_human_bboxes_gpu(self, human_image, yolov6_matrix_config):
        human_img = cv2.imread(human_image)
        # Ran on YOLOX-tiny only due to GPU OOM error on some systems
        yolov6 = Node(yolov6_matrix_config)
        output = yolov6.run({"img": human_img})

        assert "bboxes" in output
        assert output["bboxes"].size > 0

        model_type = yolov6.config["model_type"]
        image_name = Path(human_image).stem
        expected = GT_RESULTS[model_type][image_name]

        npt.assert_allclose(output["bboxes"], expected["bboxes"], atol=1e-3)
        npt.assert_equal(output["bbox_labels"], expected["bbox_labels"])
        npt.assert_allclose(output["bbox_scores"], expected["bbox_scores"], atol=1e-2)

    def test_get_detect_ids(self, yolov6_config):
        yolov6 = Node(yolov6_config)
        assert yolov6.model.detect_ids == [0]

    def test_invalid_config_detect_ids(self, yolov6_config):
        yolov6_config["detect"] = 1
        with pytest.raises(TypeError):
            _ = Node(config=yolov6_config)

    def test_invalid_config_value(self, yolov6_bad_config_value):
        with pytest.raises(ValueError) as excinfo:
            _ = Node(config=yolov6_bad_config_value)
        assert "_threshold must be between [0.0, 1.0]" in str(excinfo.value)

    @mock.patch.object(WeightsDownloaderMixin, "_has_weights", return_value=True)
    def test_invalid_config_model_files(self, _, yolov6_config):
        with pytest.raises(ValueError) as excinfo:
            yolov6_config["weights"][yolov6_config["model_format"]]["model_file"][
                yolov6_config["model_type"]
            ] = "some/invalid/path"
            _ = Node(config=yolov6_config)
        assert "Model file does not exist. Please check that" in str(excinfo.value)

    def test_invalid_image(self, no_human_image, yolov6_config):
        no_human_img = cv2.imread(no_human_image)
        yolov6 = Node(yolov6_config)
        # Potentially passing in a file path or a tuple from image reader
        # output
        with pytest.raises(TypeError) as excinfo:
            _ = yolov6.run({"img": Path.cwd()})
        assert "image must be a np.ndarray" == str(excinfo.value)
        with pytest.raises(TypeError) as excinfo:
            _ = yolov6.run({"img": ("image name", no_human_img)})
        assert "image must be a np.ndarray" == str(excinfo.value)

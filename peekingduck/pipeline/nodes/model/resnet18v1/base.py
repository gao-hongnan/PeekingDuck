"""
Base class to load model as custom models will have varying class (output neurons)
1. can load torch basic model
2. can override.

**Ideally whatever user used in training MUST be used in inference. This part
is slightly decoupled because we only focus on inference.**
"""


import torch
import torch.nn as nn
from typing import Dict, Any
from abc import ABC, abstractmethod
import torchvision

dummy_config = {
    "in_channels": 3,
    "num_classes": 2,
    "pretrained": True,
    "model_name": "resnet18",
}

import os
import random


import numpy as np
import torch


def seed_all(seed: int = 1992) -> None:
    """Seed all random number generators."""
    print(f"Using Seed Number {seed}")

    os.environ["PYTHONHASHSEED"] = str(
        seed
    )  # set PYTHONHASHSEED env var at fixed value
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)  # pytorch (both CPU and CUDA)
    np.random.seed(seed)  # for numpy pseudo-random generator
    # set fixed value for python built-in pseudo-random generator
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


seed_all(1992)


class TorchVision(nn.Module, ABC):
    in_channels: int
    pretrained: bool
    model_name: str

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        # defines 1 or 3 channels for input grayscale or RGB
        self.in_channels = config["in_channels"]
        self.pretrained = config["pretrained"]
        self.model_name = config["model_name"]
        self.num_classes = config["num_classes"]

        self.backbone = self.load_backbone()
        self.head = self.load_head()

        self.model = self.create_model()
        # self._init_weights(self.model)

    def model_summary(self):
        """torchsummary wrapper"""

    def load_backbone(self):
        """Loads a pretrained model from torchvision.models.

        Args:
            model_name (str): Name of model to load.
            pretrained (bool): If True, loads a pretrained model. Defaults to True.

        Raises:
        """
        # getattr because model_name is a string
        # do not need to state num_classes since it will be overridden later.
        backbone = getattr(torchvision.models, self.model_name)(
            pretrained=self.pretrained
        )
        return backbone

    def load_head(self):
        num_in_features = self.backbone.fc.in_features
        num_out_features = self.num_classes
        return nn.Linear(num_in_features, num_out_features)

    def create_model(self):
        # model = self.backbone
        # model.fc = self.head
        self.backbone.fc = self.head
        return self.backbone

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def create_model():
    num_classes = 2
    model = torchvision.models.resnet18(pretrained=True, num_classes=1000)

    num_in_features = model.fc.in_features
    num_out_features = num_classes
    model.fc = nn.Linear(num_in_features, num_out_features)
    return model


def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(
        model_1.state_dict().items(), model_2.state_dict().items()
    ):
        if key_item_1[1].device == key_item_2[1].device and torch.equal(
            key_item_1[1], key_item_2[1]
        ):
            pass
        else:
            models_differ += 1
            if key_item_1[0] == key_item_2[0]:
                _device = (
                    f"device {key_item_1[1].device}, {key_item_2[1].device}"
                    if key_item_1[1].device != key_item_2[1].device
                    else ""
                )
                print(key_item_1[1][0], key_item_2[1][0])

                print(f"Mismtach {_device} found at", key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print("Models match perfectly! :)")


if __name__ == "__main__":
    base = TorchVision(config=dummy_config)
    model_1 = base.model

    # model_2 = create_model()
    print(model_1.state_dict().keys())
    # print(model_2)
    # compare_models(model_1, model_2)

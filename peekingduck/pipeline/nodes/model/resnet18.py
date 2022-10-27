"""
Casting classification model.
"""

import os
import random
from tabnanny import check
from typing import Any, Dict, Union

import albumentations
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch.transforms import ToTensorV2
from peekingduck.pipeline.nodes.model.resnet18v1.base import TorchVision
from peekingduck.pipeline.nodes.node import AbstractNode
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
from collections import OrderedDict
import pandas as pd

# pylint: disable=W0223

IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def create_model():
    model = torchvision.models.resnet18(pretrained=True, num_classes=1000)
    num_ftrs = model.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model.fc = nn.Linear(num_ftrs, 2)
    return model


class LitResnet(LightningModule):
    def __init__(self, lr=0.05):
        super().__init__()

        self.save_hyperparameters()
        self.model = create_model()

    def forward(self, x):
        # print(x)
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def save_checkpoints(self):
        # TODO: https://pytorch-lightning.readthedocs.io/en/stable/common/checkpointing_basic.html
        pass

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(
        #     self.parameters(),
        #     lr=self.hparams.lr,
        #     momentum=0.9,
        #     weight_decay=5e-4,
        # )
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        # steps_per_epoch = 45000 // BATCH_SIZE
        # scheduler_dict = {
        #     "scheduler": OneCycleLR(
        #         optimizer,
        #         0.1,
        #         epochs=self.trainer.max_epochs,
        #         steps_per_epoch=steps_per_epoch,
        #     ),
        #     "interval": "step",
        # }
        return {"optimizer": optimizer}
        # return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}


class Node(AbstractNode):
    """Initializes and uses a ResNet to predict defects on steel."""

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

        self.plot_gradcam: bool

        self.num_classes: int
        self.checkpoint: str  # path to checkpoint for custom weights
        self.pretrained: bool = (
            config["pretrained"] or not self.checkpoint
        )  # False or not False = True, handles where checkpoint is provided, then pretrained is set to False to reduce overhead load

        self.class_label_map = config["class_label_map"]

        self.image_size = config["augmentation"]["image_size"]
        self.mean = config["augmentation"]["mean"]
        self.std = config["augmentation"]["std"]
        self.transforms = self._get_transforms()

        self.device = self._init_device()

        self.model = TorchVision(config).model.to(self.device)

        # self.model = LitResnet(lr=0.001).to(self.device)
        # self.model = self._create_model()
        # print(self.model.state_dict().keys())
        self.model.eval()
        self._load_checkpoint(config["checkpoint_path"])

        # TODO: logger or torchsummary to show model architecture

    def _is_model(self):
        """Check if model name is valid."""
        # implement this and raise RuntimeError if model name is invalid
        pass

    def lightning_to_torch_dicts(self, model_dict):

        new_dict = OrderedDict()

        for k, v in model_dict.items():
            if "model." in k:
                k = k.replace("model.", "")
            new_dict[k] = v
        return new_dict

    def _init_device(self) -> torch.device:
        """Copy from Yier's HF code.

        May not need for our use case, but keeping for now.
        """
        device = torch.device(
            self.config["device"] if torch.cuda.is_available() else "cpu"
        )

        return device

    def _create_loader(self):
        """Creates a PyTorch DataLoader for the dataset."""
        # implement this and return a DataLoader
        # FIXME: for now we use 1 batch size with input.visual, this should
        # be considered in major refactoring to accept batch size
        # ref: see yolo repo and timm etc.
        pass

    def _create_model(self):
        # TorchVisionHub
        model = torchvision.models.resnet18(pretrained=True, num_classes=1000)

        num_in_features = model.fc.in_features
        num_out_features = self.num_classes
        model.fc = nn.Linear(num_in_features, num_out_features)
        return model

    def _load_state_dict_from_url(self):
        """"""

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Loads the model checkpoint.

        Args:
            checkpoint_path (str): Path to the model checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint["state_dict"]
        state_dict = self.lightning_to_torch_dicts(state_dict)
        self.model.load_state_dict(state_dict)

    def _get_transforms(self):
        # can put tta #TODO put abstract method?
        # also compare this with torch transforms
        return albumentations.Compose(
            [
                albumentations.Resize(
                    self.image_size,
                    self.image_size,
                ),
                albumentations.Normalize(
                    mean=self.mean,
                    std=self.std,
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(p=1.0),
            ]
        )

    @torch.inference_mode(mode=True)
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        # FIXME: here will be different if we use albu vs transforms
        # so think of a way to abstract this (consider my old code?)
        # __call__ method
        # Note by construction of ToTensorV2, this np array will be converted to torch

        image = self.transforms(image=image)["image"].to(self.device)

        # this is because by input visual (?) the outer dim is removed so no batch size.
        # we need to add it back to make it a batch of 1 for model to do inference (by definition)
        # made explicit to note that we are adding batch dim
        image = image.unsqueeze(0) if len(image.shape) != 4 else image

        logits = self.model(image)  # model.forward(image)
        # only 1 image per inference take 0 index.
        # FIXME: softmax vs sigmoid depends on model
        probs = getattr(torch.nn, "Softmax")(dim=1)(logits).cpu().numpy()[0]
        pred_score = probs.max() * 100
        class_name = self.class_label_map[np.argmax(probs)]
        return {"pred_label": class_name, "pred_score": pred_score}

    def _get_model_artifacts(self) -> Dict[str, Any]:
        """Returns the model artifacts from MLOps directory"""

    def _tta(self):
        """TTA"""

    def _ensemble_hill_climb(self):
        """Ensemble Hill Climb, can include voting etc"""

    def _get_oof(self):
        """Get OOF predictions. Typically done during training of KFolds. Placeholder
        This can be important for ensemble models!!! A lot of winning solutions are
        based on this statistical technique.
        """

    def _get_pseudo_labels(self):
        """Get for unlabelled data."""

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Reads the image input and returns the predicted class label and probability.
        Args:
              inputs (dict): Dictionary with key "img".
        Returns:
              outputs (dict): Dictionary with keys "pred_label", "pred_score" and "gradcam_image".
        """
        # this line usually goes in dataset class but for now keep here.
        img = cv2.cvtColor(inputs["img"], cv2.COLOR_BGR2RGB)

        reshaped_original_image = cv2.resize(img, (self.image_size, self.image_size))
        prediction_dict = self.predict(img)

        gradcam_image = self.show_resnet_gradcam(
            self.model, img, reshaped_original_image, self.plot_gradcam
        )

        return {
            **prediction_dict,
            # "gradcam_image": gradcam_image,
        }

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

        gradcam_image = show_cam_on_image(
            original_image, gradcam_output[0], use_rgb=False
        )

        if plot_gradcam:
            _fig, axes = plt.subplots(figsize=(8, 8), ncols=2)
            axes[0].imshow(original_image)
            # axes[0].set_title(f"y_true={y_true:.4f}")
            axes[1].imshow(gradcam_image)
            # axes[1].set_title(f"y_pred={y_pred}")
            plt.show()
            torch.cuda.empty_cache()

        return gradcam_image

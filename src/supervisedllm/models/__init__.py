"""All Deep Neural Models are implemented in this package."""
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from torch import nn

from ..utils.helper import create_instance


class AModel(nn.Module, ABC):
    """Abstract model class and all new NN models should be sub-class of this class.

    In order one to use all feature (training, logging, ...) of the `kiwissen-base` framework
    one must implement the new models as sub-class of ``AModel`` class. This abstract class defines
    several abstract methods that are used for the trainin and evaluation.


    Args:
        metrics (List[dict]): Metrics used to benchmark the model.
        transformations (dict, optional): List of transformations that are applied on the image
            before it is passed to the model. For example normalization.

    """

    def __init__(self, model_name, **kwargs):
        super().__init__()
        if "metrics" in kwargs:
            metrics = create_instance("metrics", kwargs)
            if not isinstance(metrics, list):
                metrics = [metrics]
            self.metrics = metrics
        else:
            self.metrics = None
        self._model_name = model_name

    @abstractmethod
    def new_stats(self) -> dict:
        """Create dictionary where it will hold the results (_loss_ and _metrics_) after each training step.

        Returns:
            dict: with the name/value pairs of the metrics and losses.
        """
        raise NotImplementedError("The new_stats method is not implemented in your class!")

    @abstractmethod
    def loss(self, *inputs) -> dict:
        """Definition of the loss for each specific model that will be used for training.

        Returns:
            dict: with the name/value pairs of the metrics and losses.
        """
        raise NotImplementedError("The loss method is not implemented in your class!")

    @abstractmethod
    def train_step(self, minibatch: dict, optimizer: dict, step: int, scheduler: Any = None) -> dict:
        """The procedure executed during one training step.

        Args:
            minibatch (dict): Input minibatch.
            optimizer (dict): Optimizers used for calculating and updating the gradients
            step (int): Number of gradient update steps so far
            scheduler (Any, optional): Schedulers for annialing different parameters. Defaults to None.

        Returns:
            dict: The losses and the metrics for this training step.
        """
        raise NotImplementedError("The train_step method is not implemented in your class!")

    @abstractmethod
    def validate_step(self, minibatch: dict) -> dict:
        """The procedure executed during one validation step

        Args:
            minibatch (dict): Input minibatch.

        Returns:
            dict: The losses and the metrics for this training step.
        """
        raise NotImplementedError("The validate_step method is not implemented in your class!")

    def transform(self, x: np.ndarray) -> torch.Tensor:
        """Transform the input before passing it to the model.

        Args:
            x (np.ndarray): input

        Returns:
            torch.Tensor: transofrmed input
        """
        if self.transformations is not None:
            return self.transformations(image=x.cpu().numpy().transpose(1, 2, 0))["image"]
        return x

    @property
    def device(self):
        """The device where the model is located.

        Returns:
            torch.Device:
        """
        return next(self.parameters()).device

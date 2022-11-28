from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..metrics import accuracy
from ..utils.helper import clip_grad_norm
from . import AModel
from .blocks import BasicResNetBlock, ResNet


class CIFAR10VanillaClassifier(AModel):
    def __init__(self, **kwargs):
        super(CIFAR10VanillaClassifier, self).__init__(**kwargs)

        self._init_components()
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def _init_components(self) -> None:
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def loss(self, y: torch.Tensor, y_target: torch.Tensor) -> dict:
        loss = self.loss_fn(y, y_target)
        acc = accuracy(y, y_target)
        stats = self.new_stats()
        stats["loss"] = loss
        stats["accuracy"] = acc

        return stats

    def train_step(
        self,
        minibatch: Tuple[torch.Tensor],
        optimizer: dict,
        step: int,
        scheduler: Any = None,
    ) -> dict:
        # Optimizers initialization:
        x, label = minibatch[0].to(self.device, non_blocking=True), minibatch[1].to(self.device, non_blocking=True)
        optimizer["optimizer"]["opt"].zero_grad()

        # Train loss
        logits = self.forward(x)
        loss_stats = self.loss(logits, label)
        loss_stats["loss"].backward()
        clip_grad_norm(self.parameters(), optimizer["optimizer"])
        optimizer["optimizer"]["opt"].step()

        return loss_stats

    def validate_step(self, minibatch: Tuple[torch.Tensor]) -> dict:
        # Evaluate model
        x, label = minibatch[0].to(self.device, non_blocking=True), minibatch[1].to(self.device, non_blocking=True)
        logits = self.forward(x)
        loss_stats = self.loss(logits, label)

        return loss_stats

    def new_stats(self) -> dict:
        stats = dict()
        stats["loss"] = 0
        return stats


class ResNet18(nn.Module):
    def __init__(self, path: str, device="cpu"):
        """Constructs a ResNet-18 model."""
        super(ResNet18, self).__init__()

        self.model = ResNet(block=BasicResNetBlock, layers=[2, 2, 2, 2], num_classes=2, grayscale=False)
        self.to(device)
        if path is not None:
            self.model.load_state_dict(torch.load(path, map_location=device))

    def forward(self, x):
        return self.model(x)

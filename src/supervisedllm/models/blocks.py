from functools import partial

import torch
from torch import nn as nn
from torch.nn import functional as F
import math
from ..utils.helper import get_class_nonlinearity


def init_weights(m, init_stdv):
    if type(m) in [nn.Linear]:
        nn.init.xavier_uniform_(m.weight)
        # m.weight.data.normal_(0, init_stdv)
        m.bias.data.fill_(0)


def _build_layers(
    activation_fn,
    input_dim: int,
    layer_normalization: bool,
    layers: list,
    out_activation,
    output_dim: int,
) -> nn.Sequential:
    layer_sizes = [input_dim] + list(map(int, layers))
    layers = nn.Sequential()
    for i in range(len(layer_sizes) - 1):
        layers.add_module(f"layer {i}", nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        if layer_normalization:
            layers.add_module(f"layer norm {i}", nn.LayerNorm(layer_sizes[i + 1]))
        layers.add_module(f"activation {i}", activation_fn())
    layers.add_module("output layer", nn.Linear(layer_sizes[-1], output_dim))
    if out_activation is not None:
        out_activation_fn = get_class_nonlinearity(out_activation)
        layers.add_module("out activation", out_activation_fn())
    return layers


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        layers,
        output_dim,
        activation="LeakyReLU",
        out_activation=None,
        layer_normalization=True,
        custom_init_large_width=False,
        init_stdv=None,
    ):

        super(MLP, self).__init__()
        activation_fn = get_class_nonlinearity(activation)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.init_stdv = init_stdv
        self.layers = _build_layers(
            activation_fn,
            input_dim,
            layer_normalization,
            layers,
            out_activation,
            output_dim,
        )
        self.custom_init = custom_init_large_width
        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def init_weights(self):
        if self.custom_init:
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=2.0)
                    if module.bias.data is not None:
                        nn.init.normal_(module.bias, std=0.1)
        elif self.init_stdv is not None:
            self.layers.apply(partial(init_weights, init_stdv=self.init_stdv))

    @property
    def device(self):
        return next(self.parameters()).device


class BasicResNetBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicResNetBlock, self).__init__()
        self.conv1 = self.__conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = self.__conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

    def __conv3x3(self, in_planes, out_planes, stride=1):
        """3x3 convolution with padding"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, grayscale):
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1, padding=2)
        self.fc = nn.Linear(2048 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2.0 / n) ** 0.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas

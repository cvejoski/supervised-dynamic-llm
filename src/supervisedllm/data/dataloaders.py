"""
Definition of dataloaders for pedestrian detection datasets.
"""

import logging
from abc import ABC
from typing import List


import torch

from torch.utils.data import DataLoader
from kiwissenbase.utils.helper import create_class_instance

from .datasets import CaltechPedestrian, CityPersons, GenericVisionDataset

sampler = torch.utils.data.RandomSampler
DistributedSampler = torch.utils.data.distributed.DistributedSampler


def collate_fn(batch):
    return tuple(zip(*batch))


class ADataLoader(ABC):
    """Abstract Data Loader class that implements the common methods for the different
        pedestrian detection datasets.

    This class also supports distributed training in the data parallelization fashion. When
    ``world_size!=-1`` the dataloaders are created in the data parallel fashion.

    Args:
        device (torch.device): The device on which the data will be loaded. Accepted options are
            one of the `cuda` devices or the `cpu` device.
        rank (int, optional): The device number where the data will be loaded in case of
            distributed training. Defaults to 0.
        world_size (int, optional): Total number of devices used for distributed training. The distributed
            training is activated when ``worlsize!=-1``. Defaults to -1.
        validation_batch_size (int, optional): The batch size for the validation dataset.
        different_size_target (bool, optional): ``True`` if each image contain's different number of targets.
        train_transform (dict, optional): List of transformations that are applied on the train image
            before it is passed to the model.
        validation_transform (dict, optional): List of transformations that are applied on the
            validation image before it is passed to the model.

    .. note::
        All the keyword arguments that are accepted by the torch/kiwissen Dataset or by the torch DataLoader
        class can be passed.

    .. note::
        For the ``train_transform`` and the ``validation_transform`` arguments we bind the `Albumentations`
        library (<https://albumentations.ai/>). All the tranformations available there can be used here. In
        order to create a transformation one has to pass a list of dictionary. Each dictonary in the list
        corresponds to one transformation. See the code example below.

    Example:

    .. code-block:: python

        train_transform_ = {
            "min_area": 1024
            "min_visibility": 0.1
            "transformations": [{
                "module": "albumentations"
                "name": HorizontalFlip
                "args": {
                    "p": 0.5
                }
            }]
        }

        device = torch.device("cuda:0")
        data_loader = CityPersonsDataLoader(device, root_dir="path/to/data", batch_size=8,
            validation_batch_size=16, train_transform=train_transform_)
    """

    def __init__(self, device: torch.device, rank: int = 0, world_size=-1, **kwargs) -> None:
        super().__init__()
        self._device = device
        self._rank = rank
        self._world_size = world_size
        self._data_path = kwargs.pop("root_dir")
        self._batch_size = kwargs.pop("batch_size")
        self._validation_batch_size = kwargs.pop("validation_batch_size", self._batch_size)
        self._n_workers = kwargs.pop("num_workers", 0)
        self._different_size_target = kwargs.pop("different_size_target", False)
        self.group_pedestrian_classes = kwargs.pop("group_pedestrian_classes", False)
        self.data_subset = kwargs.pop("subset", "all")
        self.return_image_path = kwargs.pop("return_image_path", False)
        train_transform = kwargs.pop("train_transform", {})
        validation_transform = kwargs.pop("validation_transform", {})

        self.train_transform = build_transform(
            train_transform.get("transformations", []), train_transform.get("min_area", 0), train_transform.get("min_visibility", 0)
        )

        self.validation_transform = build_transform(
            validation_transform.get("transformations", []),
            validation_transform.get("min_area", 0),
            validation_transform.get("min_visibility", 0),
        )

        return kwargs

    def n_batches(self, split: str):
        valid_modes = ("train", "test", "validate")
        msg = "Unknown value '{}' for argument split. " "Valid values are {{{}}}."
        msg = msg.format(split, iterable_to_str(valid_modes))
        verify_str_arg(split, "split", valid_modes, msg)

        return len(getattr(self, split).dataset) // self._batch_size // abs(self._world_size)

    @property
    def train(self) -> DataLoader:
        """Get the train data loader.

        Returns:
            DataLoader:
        """
        return self._train_loader

    @property
    def validate(self):
        """Get the validation data loader.

        Returns:
            DataLoader:
        """
        return self._valid_loader

    @property
    def test(self):
        """Get the test data loader.

        Returns:
            DataLoader:
        """
        return self._test_loader

    @property
    def n_train_batches(self) -> int:
        """Number of training batches.

        Returns:
            int:
        """
        return self.n_batches("train")

    @property
    def n_validate_batches(self):
        """Number of validation batches.

        Returns:
            int:
        """
        return self.n_batches("validate")

    @property
    def n_test_batches(self):
        """Number of test batches.

        Returns:
            int:
        """
        return self.n_batches("test")

    @property
    def train_set_size(self) -> int:
        """Number of examples in the training set.

        Returns:
            int:
        """
        return len(self.train.dataset)

    @property
    def val_set_size(self):
        """Number of examples in the validation set.

        Returns:
            int:
        """
        return len(self.validate.dataset)

    @property
    def test_set_size(self):
        """Number of examples in the test set.

        Returns:
            int:
        """
        return len(self.test.dataset)

    def _init_dataloaders(self, train_dataset, valid_dataset, test_dataset, **kwargs):
        train_sampler = None
        valid_sampler = None
        test_sampler = None
        col_fn = None
        if self._world_size != -1:
            train_sampler = DistributedSampler(train_dataset, self._world_size, self._rank)
            valid_sampler = DistributedSampler(valid_dataset, self._world_size, self._rank)
            test_sampler = DistributedSampler(test_dataset, self._world_size, self._rank)
        if self._different_size_target:
            col_fn = collate_fn
        self._train_loader = DataLoader(
            train_dataset,
            batch_size=self._batch_size,
            num_workers=self._n_workers,
            sampler=train_sampler,
            shuffle=train_sampler is None,
            collate_fn=col_fn,
            **kwargs,
        )
        self._valid_loader = DataLoader(
            valid_dataset,
            batch_size=self._validation_batch_size,
            num_workers=self._n_workers,
            sampler=valid_sampler,
            shuffle=valid_sampler is None,
            collate_fn=col_fn,
            **kwargs,
        )
        self._test_loader = DataLoader(
            test_dataset,
            batch_size=self._validation_batch_size,
            num_workers=self._n_workers,
            sampler=test_sampler,
            shuffle=test_sampler is None,
            collate_fn=col_fn,
            **kwargs,
        )


def build_transform(input_transform_params: List[dict], min_area: int, min_visibility: float) -> A.Compose:
    """Build `Albumentation` transformation from a list of dictionaries.

    Args:
        input_transform_params (List[dict]): List of transformations described as dictionary
            See the code example below.
        min_area (int): of the bounding box after transformation
        min_visibility (float): of the bounding box after transformation.

    .. note:
        If after transformation the bonding box is smaller than ``min_area`` or ``min_visibility``
        it will be removed.

    .. code-

    Returns:
        A.Compose:
    """
    sequence = []
    for trans_parameters in input_transform_params:
        module = trans_parameters["module"]
        class_name = trans_parameters["name"]
        args = trans_parameters["args"]
        sequence.append(create_class_instance(module, class_name, args))
    sequence.append(ToTensorV2())

    return A.Compose(
        sequence,
        bbox_params=A.BboxParams(
            format="pascal_voc",
            min_area=min_area,
            min_visibility=min_visibility,
            label_fields=["class_labels", "bboxesVisRatio", "bboxesHeight"],
        ),
    )


class CIFAR10DataLoader(ADataLoader):
    def __init__(self, device: torch.device, rank: int = 0, world_size=-1, **kwargs):
        logger = logging.getLogger(self.__class__.__name__)
        kwargs = super().__init__(device, rank, world_size, **kwargs)
        train_dataset = datasets.CIFAR10(self._data_path, transform=self.train_transform, download=True, train=True)
        test_dataset = datasets.CIFAR10(self._data_path, transform=self.validation_transform, download=True, train=False)

        self._init_dataloaders(train_dataset, test_dataset, test_dataset, **kwargs)
        logger.info("Train Dataset Stats: %s", train_dataset)
        logger.info("Test Dataset Stats: %s", test_dataset)


class CaltechPedastrianDataLoader(ADataLoader):
    """Data loader for the `Caltech Pedestrian` <https://authors.library.caltech.edu/87172/1/05206631.pdf> dataset."""

    def __init__(self, device: torch.device, rank: int = 0, world_size=-1, **kwargs):
        logger = logging.getLogger(self.__class__.__name__)
        kwargs = super().__init__(device, rank, world_size, **kwargs)

        train_dataset = CaltechPedestrian(
            self._data_path,
            transform=self.train_transform,
            download=True,
            train=True,
            subset=self.data_subset,
            group_pedestrian_classes=self.group_pedestrian_classes,
            return_image_path=self.return_image_path,
        )
        test_dataset = CaltechPedestrian(
            self._data_path,
            transform=self.validation_transform,
            download=True,
            train=False,
            subset=self.data_subset,
            group_pedestrian_classes=self.group_pedestrian_classes,
            return_image_path=self.return_image_path,
        )

        self._init_dataloaders(train_dataset, test_dataset, test_dataset, **kwargs)
        logger.info("Train Dataset Stats: %s", train_dataset)
        logger.info("Test Dataset Stats: %s", test_dataset)


class CityPersonsDataLoader(ADataLoader):
    """Data loader for the `CityPersons` <http://www.cityscapes-dataset.com/> dataset."""

    def __init__(self, device: torch.device, rank: int = 0, world_size=-1, **kwargs):
        logger = logging.getLogger(self.__class__.__name__)
        kwargs = super().__init__(device, rank, world_size, **kwargs)

        train_dataset = CityPersons(
            self._data_path,
            transform=self.train_transform,
            split="train",
            subset=self.data_subset,
            group_pedestrian_classes=self.group_pedestrian_classes,
            return_image_path=self.return_image_path,
        )
        val_dataset = CityPersons(
            self._data_path,
            transform=self.validation_transform,
            split="val",
            subset=self.data_subset,
            group_pedestrian_classes=self.group_pedestrian_classes,
            return_image_path=self.return_image_path,
        )
        test_dataset = CityPersons(
            self._data_path,
            transform=self.validation_transform,
            split="test",
            subset=self.data_subset,
            group_pedestrian_classes=self.group_pedestrian_classes,
            return_image_path=self.return_image_path,
        )

        self._init_dataloaders(train_dataset, val_dataset, test_dataset, **kwargs)
        logger.info("Train Dataset Stats: %s", train_dataset)
        logger.info("Validation Dataset Stats: %s", val_dataset)
        logger.info("Test Dataset Stats: %s", test_dataset)


class EuroCityPersonsDataLoader(ADataLoader):
    """Data loader for the `EuroCityPersons` <https://eurocity-dataset.tudelft.nl/> dataset."""

    def __init__(self, device: torch.device, rank: int = 0, world_size=-1, **kwargs):
        logger = logging.getLogger(self.__class__.__name__)
        kwargs = super().__init__(device, rank, world_size, **kwargs)
        time = kwargs.pop("time", "day")

        train_dataset = EuroCityPersons(
            self._data_path,
            time=time,
            transform=self.train_transform,
            split="train",
            subset=self.data_subset,
            group_pedestrian_classes=self.group_pedestrian_classes,
            return_image_path=self.return_image_path,
        )
        val_dataset = EuroCityPersons(
            self._data_path,
            time=time,
            transform=self.validation_transform,
            split="val",
            subset=self.data_subset,
            group_pedestrian_classes=self.group_pedestrian_classes,
            return_image_path=self.return_image_path,
        )
        test_dataset = EuroCityPersons(
            self._data_path,
            time=time,
            transform=self.validation_transform,
            split="test",
            subset=self.data_subset,
            group_pedestrian_classes=self.group_pedestrian_classes,
            return_image_path=self.return_image_path,
        )

        self._init_dataloaders(train_dataset, val_dataset, test_dataset, **kwargs)
        logger.info("Train Dataset Stats: %s", train_dataset)
        logger.info("Validation Dataset Stats: %s", val_dataset)
        logger.info("Test Dataset Stats: %s", test_dataset)


class GenericVisionDataLoader(ADataLoader):
    def __init__(self, device: torch.device, rank: int = 0, world_size=-1, **kwargs) -> None:
        logger = logging.getLogger(self.__class__.__name__)
        kwargs = super().__init__(device, rank, world_size, **kwargs)
        self._target_path = kwargs.pop("target_path")

        dataset = GenericVisionDataset(
            self._data_path,
            self._target_path,
            transform=self.train_transform,
        )

        sampler_ = None
        if self._world_size != -1:
            sampler_ = DistributedSampler(dataset, self._world_size, self._rank)
        if self._different_size_target:
            col_fn = collate_fn
        self._data_loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            num_workers=self._n_workers,
            sampler=sampler_,
            collate_fn=col_fn,
        )
        logger.info("Dataset Stats: %s", dataset)

    @property
    def data(self):
        return self._data_loader

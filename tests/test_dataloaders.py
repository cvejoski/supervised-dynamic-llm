import os

import albumentations as A
import pytest
import torch

from kiwissenbase import data_path
from kiwissenbase.data.dataloaders import (
    CaltechPedastrianDataLoader,
    CityPersonsDataLoader,
    EuroCityPersonsDataLoader,
    build_transform,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from kiwissenbase import data_path

citypersons_root = data_path / "Cityscapes"
eurocitypersons_root = data_path / "ECP"
caltech_root = data_path / "CaltechPedestrian"


def test_building_input_transformation():

    params = [
        {
            "module": "albumentations",
            "name": "Normalize",
            "args": {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
        },
        {
            "module": "albumentations",
            "name": "CenterCrop",
            "args": {"width": 128, "height": 128},
        },
    ]

    transform = build_transform(params, min_area=0, min_visibility=0.0)

    assert transform is not None
    assert isinstance(transform[0], A.Normalize)
    assert isinstance(transform[1], A.CenterCrop)


@pytest.mark.skipif(not os.path.exists(caltech_root), reason="The dataset is not downloaded")
def test_caltechpedestrian_dataloader():
    params = dict()
    params["root_dir"] = caltech_root
    params["valid_size"] = 0.1
    params["batch_size"] = 32
    params["validation_batch_size"] = 32
    params["num_workers"] = 4
    params["world_size"] = -1
    params["rank"] = 1
    params["subset"] = "annotated-pedestrians"
    params["different_size_target"] = True
    caltech_dl = CaltechPedastrianDataLoader(device, **params)

    assert caltech_dl.train_set_size == 1_609
    assert caltech_dl.test_set_size == 1_601
    train = caltech_dl.train.__iter__()
    x = next(train)
    assert len(x[0]) == 32


@pytest.mark.skipif(not os.path.exists(citypersons_root), reason="The dataset is not downloaded")
def test_citypersons_dataloader():
    params = dict()
    params["root_dir"] = citypersons_root
    params["batch_size"] = 32
    params["validation_batch_size"] = 32
    params["num_workers"] = 4
    params["world_size"] = -1
    params["rank"] = 1

    citypersons_dl = CityPersonsDataLoader(device, **params)

    assert citypersons_dl.train_set_size == 2_975
    assert citypersons_dl.val_set_size == 500
    assert citypersons_dl.test_set_size == 1_525


@pytest.mark.skipif(not os.path.exists(eurocitypersons_root), reason="The dataset is not downloaded")
def test_eurocitypersons_dataloader():
    params = dict()
    params["root_dir"] = eurocitypersons_root
    params["batch_size"] = 32
    params["validation_batch_size"] = 32
    params["time"] = "day"
    params["num_workers"] = 4
    params["world_size"] = -1
    params["rank"] = 1

    eurocitypersons_dl = EuroCityPersonsDataLoader(device, **params)

    assert eurocitypersons_dl.train_set_size == 23_892
    assert eurocitypersons_dl.val_set_size == 4_266
    assert eurocitypersons_dl.test_set_size == 12_059

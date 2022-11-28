# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring
import os

import pytest
import torch

from supervisedllm import data_path
from supervisedllm.data.dataloaders import TopicDataLoader

arxiv_root = data_path / "preprocessed" / "arxiv"


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@pytest.mark.skipif(not os.path.exists(arxiv_root), reason="The dataset is not downloaded")
def test_axiv_dataloader():
    params = dict()
    params["root_dir"] = arxiv_root
    params["batch_size"] = 32
    params["validation_batch_size"] = 32
    params["num_workers"] = 4
    params["world_size"] = -1
    params["rank"] = 1
    params["is_dynamic"] = True

    arxiv_dl = TopicDataLoader(device, **params)

    assert arxiv_dl.train_set_size == 7_805
    assert arxiv_dl.test_set_size == 434
    train = arxiv_dl.train.__iter__()
    x = next(train)
    assert len(x["reward"]) == 32

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring
import os

import pytest

from supervisedllm import data_path
from supervisedllm.data.datasets import TopicDataset

arxiv_root = data_path / "preprocessed" / "arxiv"


@pytest.mark.skipif(not os.path.exists(arxiv_root), reason="The dataset is not downloaded")
class TestArxiv:
    def test_initialize_arxiv(self):
        train = TopicDataset(
            arxiv_root,
            ds_type="train",
            is_dynamic=True,
            use_covar=False,
            use_tmp_covariates=False,
            normalize_data=False,
            word_emb_type="bow",
        )
        test = TopicDataset(
            arxiv_root,
            ds_type="test",
            is_dynamic=True,
            use_covar=False,
            use_tmp_covariates=False,
            normalize_data=False,
            word_emb_type="bow",
        )

        assert len(train) == 7_805
        assert len(test) == 434

    @pytest.mark.skipif(not os.path.exists(arxiv_root), reason="The dataset is not downloaded")
    def test_get_arxiv_document(self):
        train = TopicDataset(
            arxiv_root,
            ds_type="train",
            is_dynamic=True,
            use_covar=False,
            use_tmp_covariates=False,
            normalize_data=False,
            word_emb_type="bow",
        )
        x = train[3]
        assert x is not None
        assert isinstance(x, dict)
        assert "seq" in x
        assert "seq_len" in x
        assert "bow" in x


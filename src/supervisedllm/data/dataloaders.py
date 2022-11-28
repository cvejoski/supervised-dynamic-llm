"""
Definition of dataloaders.
"""

from abc import ABC
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

from supervisedllm.utils.helper import iterable_to_str, verify_str_arg

from .datasets import TopicDataset

sampler = torch.utils.data.RandomSampler
DistributedSampler = torch.utils.data.distributed.DistributedSampler


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

    .. note::
        All the keyword arguments that are accepted by the torch/kiwissen Dataset or by the torch DataLoader
        class can be passed.

    Example:

    .. code-block:: python


        device = torch.device("cuda:0")
        data_loader = TopicDataLoader(device, root_dir="path/to/data", batch_size=8,
            validation_batch_size=16)
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


class TopicDataLoader(ADataLoader):
    def __init__(self, device, rank: int = 0, world_size=-1, **kwargs):
        super().__init__(device, rank, world_size, **kwargs)

        data_dir = kwargs.get("root_dir")
        is_dynamic = kwargs.get("is_dynamic", False)
        use_covariates = kwargs.get("use_covariates", False)
        use_tmp_covariates = kwargs.get("use_tmp_covariates", False)
        word_emb_type = kwargs.get("word_emb_type", "bow")
        normalize = kwargs.get("normalize", False)
        n_workers = kwargs.get("n_workers", 8)
        reward_field = kwargs.get("reward_field", "reward")
        transformer_name = kwargs.get("transformer_name", None)
        tokenizer = self.get_transformer_tokenizer(transformer_name)
        train_dataset = TopicDataset(
            data_dir, "train", is_dynamic, use_covariates, use_tmp_covariates, normalize, word_emb_type, tokenizer, reward_field
        )
        valid_dataset = TopicDataset(
            data_dir,
            "validation",
            is_dynamic,
            use_covariates,
            use_tmp_covariates,
            train_dataset.cov_stand,
            word_emb_type,
            tokenizer,
            reward_field,
        )
        test_dataset = TopicDataset(
            data_dir,
            "test",
            is_dynamic,
            use_covariates,
            use_tmp_covariates,
            train_dataset.cov_stand,
            word_emb_type,
            tokenizer,
            reward_field,
        )
        predict_dataset = TopicDataset(
            data_dir,
            "prediction",
            is_dynamic,
            use_covariates,
            use_tmp_covariates,
            train_dataset.cov_stand,
            word_emb_type,
            tokenizer,
            reward_field,
        )
        if is_dynamic:
            valid_dataset.corpus_per_time_period_avg = train_dataset.corpus_per_time_period_avg
            test_dataset.corpus_per_time_period_avg = train_dataset.corpus_per_time_period_avg

            valid_dataset.reward_per_time_period_avg = train_dataset.reward_per_time_period_avg
            test_dataset.reward_per_time_period_avg = train_dataset.reward_per_time_period_avg

        train_sampler = None
        valid_sampler = None
        test_sampler = None
        predict_sampler = None

        if self._world_size != -1:
            train_sampler = DistributedSampler(train_dataset, self._world_size, self._rank)
            valid_sampler = DistributedSampler(valid_dataset, self._world_size, self._rank)
            test_sampler = DistributedSampler(test_dataset, self._world_size, self._rank)
            if is_dynamic:
                predict_sampler = DistributedSampler(predict_dataset, self._world_size, self._rank)

        self._train_loader = DataLoader(
            train_dataset,
            drop_last=False,
            sampler=train_sampler,
            shuffle=train_sampler is None,
            batch_size=self._batch_size,
            num_workers=n_workers,
            pin_memory=True,
        )
        self._valid_loader = DataLoader(
            valid_dataset,
            drop_last=False,
            sampler=valid_sampler,
            shuffle=valid_sampler is None,
            batch_size=self._validation_batch_size,
            num_workers=n_workers,
            pin_memory=True,
        )
        self._test_loader = DataLoader(
            test_dataset,
            drop_last=False,
            sampler=test_sampler,
            shuffle=test_sampler is None,
            batch_size=self._validation_batch_size,
            num_workers=n_workers,
            pin_memory=True,
        )
        # if is_dynamic:
        self._predict_loader = DataLoader(
            predict_dataset, drop_last=False, sampler=predict_sampler, shuffle=predict_sampler is None, batch_size=self._batch_size
        )

        self._rewards_values = train_dataset.rewards_categories()
        self._number_of_reward_categories = train_dataset.number_of_rewards_categories()
        self._type_of_rewards = train_dataset.type_of_rewards()

    @property
    def vocabulary_dim(self):
        return self.train.dataset.data["bow"].shape[1]

    @property
    def word_embeddings_dim(self):
        return self.train.dataset.vocab.vectors.shape[1]

    @property
    def number_of_documents(self):
        return self.train_set_size + self.validation_set_size + self.test_set_size

    @property
    def rewards_values(self):
        if self._type_of_rewards == "discrete":
            return self._rewards_values
        else:
            return None

    @property
    def number_of_reward_categories(self):
        if self._type_of_rewards == "discrete":
            return self._number_of_reward_categories

        return np.inf

    @property
    def type_of_rewards(self):
        return self._type_of_rewards

    @property
    def word_emb_type(self):
        return self.train.dataset.word_emb_type

    def get_transformer_tokenizer(self, tokenizer_name):
        if tokenizer_name is None:
            return None
        elif tokenizer_name == "bert":
            from transformers import BertTokenizer

            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
            tokenizer.name = "bert-base-uncased"
        elif tokenizer_name == "roberta":
            from transformers import RobertaTokenizer

            tokenizer = RobertaTokenizer.from_pretrained("roberta-base", do_lower_case=True)
            tokenizer.name = "roberta-base"
        elif tokenizer_name == "albert":
            from transformers import AlbertTokenizer

            tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2", do_lower_case=True)
            tokenizer.name = "albert-base-v2"
        else:
            raise ValueError("No matching backbone network")
        tokenizer = partial(
            tokenizer, add_special_tokens=True, truncation=True, padding="max_length", return_attention_mask=True, return_tensors="pt"
        )
        return tokenizer

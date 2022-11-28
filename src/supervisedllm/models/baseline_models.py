from collections import defaultdict
from typing import Any, Tuple

import torch
from torch import nn

from supervisedllm.data.dataloaders import ADataLoader
from supervisedllm.models import AModel
from supervisedllm.models.blocks import MLP
from supervisedllm.utils.helper import clip_grad_norm


def exp_nllloss(y_hat, y, var):
    dist = torch.distributions.exponential.Exponential(y_hat)
    log_likelihood = dist.log_prob(y).sum()
    return -log_likelihood


def load_backbone(name, output_attentions=False):
    if name == "bert":
        from transformers import BertModel  # , BertTokenizer

        backbone = BertModel.from_pretrained("bert-base-uncased", output_attentions=output_attentions)
    elif name == "roberta":
        from transformers import RobertaModel  # , RobertaTokenizer

        backbone = RobertaModel.from_pretrained("roberta-base", output_attentions=output_attentions)
    elif name == "albert":
        from transformers import AlbertModel  # , AlbertTokenizer

        backbone = AlbertModel.from_pretrained("albert-base-v2", output_attentions=output_attentions)
    else:
        raise ValueError("No matching backbone network")

    return backbone


class NonSequentialModel(AModel):
    def __init__(self, model_name, **kwargs):
        AModel.__init__(self, model_name, **kwargs)

    @classmethod
    def get_parameters(cls, data_loader: ADataLoader):
        """
        here we provide an example of the minimum set of parameters requiered to instantiate the model
        """
        parameters_sample = {
            "vocabulary_dim": data_loader.vocabulary_dim,
            "dropout": 0.1,
            "output_layers_dim": [10, 10],
            "bow_layers_dim": [10, 10],
            "cov_layers_dim": [10, 10],
            "bow_emb_dim": 50,
            "cov_emb_dim": 50,
            "output_dim": 1,
            "output_transformation": "relu",
            "word_emb_type": data_loader.word_emb_type,
            "covariates_dim": data_loader.train.dataset.covariates_size,
        }

        return parameters_sample

    def set_parameters(self, **kwargs):
        self.covariates_dim = kwargs.get("covariates_dim")
        self.vocabulary_dim = kwargs.get("vocabulary_dim")
        self.output_dim = kwargs.get("output_dim")
        self.dropout = kwargs.get("dropout")
        self.output_layers_dim = kwargs.get("output_layers_dim")
        self.bow_layers_dim = kwargs.get("bow_layers_dim")
        self.cov_layers_dim = kwargs.get("cov_layers_dim")
        self.cov_emb_dim = kwargs.get("cov_emb_dim")
        self.bow_emb_dim = kwargs.get("bow_emb_dim")
        self.word_emb_type = kwargs.get("word_emb_type")
        self.output_transformation = kwargs.get("output_transformation")

    @classmethod
    def get_inference_parameters(cls):
        inference_parameters = super().get_inference_parameters()
        inference_parameters.update({"regularizers": {"nll": None}})
        inference_parameters.update({"model_eval": "RMSE"})

        return inference_parameters

    def define_deep_models(self):
        if self.covariates_dim == 0:
            self.bow2emb = MLP(
                input_dim=self.vocabulary_dim,
                output_dim=self.bow_emb_dim,
                layers_dim=self.bow_layers_dim,
                dropout=self.dropout,
                output_transformation="relu",
            )
            self.regressor = MLP(
                input_dim=self.bow_emb_dim,
                output_dim=self.output_dim,
                layers_dim=self.output_layers_dim,
                dropout=self.dropout,
                output_transformation=self.output_transformation,
            )
        else:
            self.bow2emb = MLP(
                input_dim=self.vocabulary_dim,
                output_dim=self.bow_emb_dim,
                layers_dim=self.bow_layers_dim,
                dropout=self.dropout,
                output_transformation="relu",
            )
            self.cov2emb = MLP(
                input_dim=self.covariates_dim,
                output_dim=self.cov_emb_dim,
                layers_dim=self.cov_layers_dim,
                dropout=self.dropout,
                output_transformation="relu",
            )
            self.regressor = MLP(
                input_dim=self.bow_emb_dim + self.cov_emb_dim,
                output_dim=self.output_dim,
                layers_dim=self.output_layers_dim,
                dropout=self.dropout,
                output_transformation=self.output_transformation,
            )

    def forward(self, x):
        """
        parameters
        ----------
        data ()
        returns
        -------
        z (batch_size*sequence_lenght,self.hidden_state_dim)
        recognition_parameters = (z_mean,z_var)
        (batch_size*sequence_lenght,self.hidden_state_dim)
        likelihood_parameters = (likelihood_mean,likelihood_variance)
        """

        bow = x["bow"].to(self.device, non_blocking=True)

        emb = bow
        if self.word_emb_type == "bow":
            emb = bow.float() / bow.sum(1, True)

        emb = self.bow2emb(emb)
        if self.covariates_dim != 0:
            cov_emb = self.cov2emb(x["covariates"].to(self.device, non_blocking=True))
            emb = torch.cat((emb, cov_emb), dim=1)
        y = self.regressor(emb)
        return y

    def train_step(self, minibatch: Tuple[torch.Tensor], optimizer: dict, step: int, scheduler: Any = None) -> dict:
        optimizer["optimizer"]["opt"].zero_grad()

        # Train loss
        y_predict = self.forward(minibatch)
        loss_metrics = self.loss(y_predict, minibatch["reward"])
        loss_metrics["loss"].backward()

        clip_grad_norm(self.parameters(), optimizer["optimizer"])
        optimizer["optimizer"]["opt"].step()

        return loss_metrics

    def validate_step(self, minibatch: Tuple[torch.Tensor]) -> dict:
        y_predict = self.forward(minibatch)
        loss_metrics = self.loss(y_predict, minibatch["reward"])
        return loss_metrics

    def new_stats(self) -> dict:
        stats = {}
        stats["loss"] = torch.tensor(0.0, device=self.device)

        return stats


class NonSequentialClassifier(NonSequentialModel):
    name_: str = "nonsequential_classifier"

    def __init__(self, **kwargs):
        NonSequentialModel.__init__(self, model_name=self.name_, **kwargs)
        self.set_parameters(**kwargs)
        self.define_deep_models()
        self.define_loss_metrics()

    def define_loss_metrics(self) -> None:
        if self.output_dim == 1:
            self.__loss_ce = nn.BCELoss(reduction="mean")
        else:
            self.__loss_ce = nn.CrossEntropyLoss(reduction="mean")

    @classmethod
    def get_parameters(cls, data_loader: ADataLoader):
        parameters = super().get_parameters(data_loader)
        parameters["output_dim"] = data_loader.number_of_reward_categories
        if data_loader.number_of_reward_categories == 1:
            parameters["output_transformation"] = "sigmoid"
        return parameters

    @classmethod
    def get_inference_parameters(cls):
        inference_parameters = super().get_inference_parameters()
        inference_parameters.update({"regularizers": {"nll": None}})
        inference_parameters.update({"model_eval": "accuracy"})
        inference_parameters.update({"loss_type": "classification"})
        return inference_parameters

    def loss(self, y_predict, y_target):
        """
        nll [batch_size, max_lenght]

        """
        y_predict = y_predict.squeeze(1)
        y_target = y_target.float().to(self.device, non_blocking=True)

        l = self.__loss_ce(y_predict, y_target)
        if self.output_dim == 1:
            accuracy = (y_predict.round() == y_target).float()
        else:
            accuracy = (torch.argmax(y_predict) == y_target).float()
        return {"loss": l, "accuracy": accuracy.mean()}


class NonSequentialRegression(NonSequentialModel):
    name_: str = "nonsequential_regressor"

    def __init__(self, **kwargs):
        NonSequentialModel.__init__(self, model_name=self.name_, **kwargs)
        self.set_parameters(**kwargs)
        self.define_deep_models()
        self.define_loss_metrics()

    def define_loss_metrics(self) -> None:
        self.__loss_mse = nn.MSELoss(reduction="sum")
        self.__loss_mae = nn.L1Loss(reduction="sum")

        if self.regression_dist == "normal":
            self.nll_regression = nn.GaussianNLLLoss(reduction="sum")
        elif self.regression_dist == "exp":
            self.nll_regression = exp_nllloss

    @classmethod
    def get_parameters(cls, data_loader: ADataLoader):
        parameters = super().get_parameters(data_loader)
        parameters["regression_dist"] = "exp"  # exp or normal

        return parameters

    def set_parameters(self, **kwargs):
        super().set_parameters(**kwargs)
        self.regression_dist = kwargs.get("regression_dist", "exp")

    @classmethod
    def get_inference_parameters(cls):
        inference_parameters = super().get_inference_parameters()
        inference_parameters.update({"regularizers": {"nll": None}})
        inference_parameters.update({"model_eval": "RMSE"})
        inference_parameters.update({"loss_type": "regression"})
        return inference_parameters

    def loss(self, y_predict, y_target):
        """
        nll [batch_size, max_lenght]

        """
        y_predict = y_predict.squeeze(1)
        y_target = y_target.float().to(self.device, non_blocking=True)
        batch_size = y_target.size(0)

        mse = self.__loss_mse(y_predict, y_target)
        mae = self.__loss_mae(y_predict, y_target)

        return {"loss": mse, "MAE": mae / batch_size, "RMSE": torch.sqrt(mse / batch_size)}


class SequentialModel(AModel):
    def __init__(self, model_name, **kwargs):
        AModel.__init__(self, model_name, **kwargs)

    @classmethod
    def get_parameters(cls, data_loader: ADataLoader):
        """
        here we provide an example of the minimum set of parameters requiered to instantiate the model
        """
        parameters_sample = {
            "backbone_name": "bert",
            "output_dim": 1,
            "dropout": 0.1,
            "cov_layers_dim": [10, 10],
            "output_layers_dim": [],
            "train_backbone": False,
            "cov_emb_dim": 50,
            "output_transformation": "relu",
            "covariates_dim": data_loader.train.dataset.covariates_size,
        }

        return parameters_sample

    def set_parameters(self, **kwargs):
        self.backbone_name = kwargs.get("backbone_name")
        self.covariates_dim = kwargs.get("covariates_dim")
        self.train_backbone = kwargs.get("train_backbone")
        self.output_dim = kwargs.get("output_dim")
        self.output_transformation = kwargs.get("output_transformation")
        self.dropout = kwargs.get("dropout")
        self.cov_layers_dim = kwargs.get("cov_layers_dim")
        self.output_layers_dim = kwargs.get("output_layers_dim")
        self.cov_emb_dim = kwargs.get("cov_emb_dim")

    @classmethod
    def get_inference_parameters(cls):
        inference_parameters = super().get_inference_parameters()
        inference_parameters.update({"regularizers": {"nll": None}})
        inference_parameters.update({"model_eval": "RMSE"})

        return inference_parameters

    def define_deep_models(self):
        self.backbone: nn.Module = load_backbone(self.backbone_name)
        if not self.train_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.dropout_layer = nn.Dropout(self.dropout)
        if self.covariates_dim == 0:
            self.out_layer = MLP(
                input_dim=768,
                output_dim=self.output_dim,
                layers_dim=self.output_layers_dim,
                dropout=self.dropout,
                output_transformation=self.output_transformation,
            )

        else:
            self.cov2emb = MLP(
                input_dim=self.covariates_dim,
                output_dim=self.cov_emb_dim,
                layers_dim=self.cov_layers_dim,
                dropout=self.dropout,
                output_transformation="relu",
            )
            self.out_layer = MLP(
                input_dim=768 + self.cov_emb_dim,
                output_dim=self.output_dim,
                layers_dim=self.output_layers_dim,
                dropout=self.dropout,
                output_transformation=self.output_transformation,
            )
        self._forward_backbone = self.__forward_bert_albert
        if self.backbone_name == "roberta":
            self._forward_backbone = self.__forward_roberta

    def forward(self, x):
        out_p = self._forward_backbone(x)
        emb = self.dropout_layer(out_p)
        if self.covariates_dim != 0:
            cov_emb = self.cov2emb(x["covariates"].to(self.device, non_blocking=True))
            emb = torch.cat((emb, cov_emb), dim=1)

        out = self.out_layer(emb)
        return out

    def __forward_bert_albert(self, x):
        text = x["text"]

        _, out_p = self.backbone(
            input_ids=text["input_ids"].squeeze(1).to(self.device, non_blocking=True),
            token_type_ids=text["token_type_ids"].squeeze(1).to(self.device, non_blocking=True),
            attention_mask=text["attention_mask"].squeeze(1).to(self.device, non_blocking=True),
            return_dict=False,
        )
        return out_p

    def __forward_roberta(self, x):
        text = x["text"]

        _, out_p = self.backbone(
            input_ids=text["input_ids"].squeeze(1).to(self.device, non_blocking=True),
            attention_mask=text["attention_mask"].squeeze(1).to(self.device, non_blocking=True),
            return_dict=False,
        )
        return out_p

    def train_step(self, minibatch: Tuple[torch.Tensor], optimizer: dict, step: int, scheduler: Any = None) -> dict:
        optimizer["optimizer"]["opt"].zero_grad()

        # Train loss
        y_predict = self.forward(minibatch)
        loss_metrics = self.loss(y_predict, minibatch["reward"])
        loss_metrics["loss"].backward()

        clip_grad_norm(self.parameters(), optimizer["optimizer"])
        optimizer["optimizer"]["opt"].step()

        return loss_metrics

    def validate_step(self, minibatch: Tuple[torch.Tensor]) -> dict:
        y_predict = self.forward(minibatch)
        loss_metrics = self.loss(y_predict, minibatch["reward"])

        return loss_metrics

    def new_stats(self) -> dict:
        stats = {}
        stats["loss"] = torch.tensor(0.0, device=self.device)

        return stats


class SequentialClassifier(SequentialModel):
    name_: str = "sequential_classification"

    def __init__(self, **kwargs):
        SequentialModel.__init__(self, model_name=self.name_, **kwargs)
        self.set_parameters(**kwargs)
        self.define_deep_models()
        self.define_loss_metrics()

    def define_loss_metrics(self) -> None:
        if self.output_dim == 1:
            self.__loss_ce = nn.BCELoss(reduction="mean")
        else:
            self.__loss_ce = nn.CrossEntropyLoss(reduction="mean")

    @classmethod
    def get_parameters(cls, data_loader: ADataLoader):
        parameters = super().get_parameters(data_loader)
        parameters["output_dim"] = data_loader.number_of_reward_categories
        if data_loader.number_of_reward_categories == 1:
            parameters["output_transformation"] = "sigmoid"
        return parameters

    @classmethod
    def get_inference_parameters(cls):
        inference_parameters = super().get_inference_parameters()
        inference_parameters.update({"regularizers": {"nll": None}})
        inference_parameters.update({"model_eval": "accuracy"})
        inference_parameters.update({"loss_type": "classification"})

        return inference_parameters

    def loss(self, y_predict, y_target):
        """
        nll [batch_size, max_lenght]

        """
        y_predict = y_predict.squeeze(1)
        y_target = y_target.float().to(self.device, non_blocking=True)

        l = self.__loss_ce(y_predict, y_target)
        if self.output_dim == 1:
            accuracy = (y_predict.round() == y_target).float()
        else:
            accuracy = (torch.argmax(y_predict) == y_target).float()
        return {"loss": l, "accuracy": accuracy.mean()}


class SequentialRegression(SequentialModel):
    name_: str = "sequential_regression"

    def __init__(self, **kwargs):
        SequentialModel.__init__(self, model_name=self.name_, **kwargs)
        self.set_parameters(**kwargs)
        self.define_deep_models()
        self.define_loss_metrics()

    def define_loss_metrics(self) -> None:
        self.__loss_mse = nn.MSELoss(reduction="sum")
        self.__loss_mae = nn.L1Loss(reduction="sum")

        if self.regression_dist == "normal":
            self.nll_regression = nn.GaussianNLLLoss(reduction="sum")
        elif self.regression_dist == "exp":
            self.nll_regression = exp_nllloss

    @classmethod
    def get_parameters(cls, data_loader: ADataLoader):
        parameters = super().get_parameters(data_loader)
        parameters["regression_dist"] = "exp"  # exp or normal
        return parameters

    def set_parameters(self, **kwargs):
        super().set_parameters(**kwargs)
        self.regression_dist = kwargs.get("regression_dist", "exp")

    @classmethod
    def get_inference_parameters(cls):
        inference_parameters = super().get_inference_parameters()

        inference_parameters.update({"model_eval": "RMSE"})
        inference_parameters.update({"loss_type": "regression"})
        return inference_parameters

    def loss(self, y_predict, y_target):
        """
        nll [batch_size, max_lenght]

        """
        y_predict = y_predict.squeeze(1)
        y_target = y_target.float().to(self.device, non_blocking=True)
        batch_size = y_target.size(0)

        mse = self.__loss_mse(y_predict, y_target)
        mae = self.__loss_mae(y_predict, y_target)

        return {"loss": mse, "MAE": mae / batch_size, "RMSE": torch.sqrt(mse / batch_size)}


if __name__ == "__main__":
    from torch.optim import AdamW

    from supervisedllm import data_path
    from supervisedllm.data.dataloaders import TopicDataLoader

    data_dir = data_path / "preprocessed" / "arxiv"
    dataloader_params = {"root_dir": data_dir, "batch_size": 32}

    data_loader = TopicDataLoader("cpu", **dataloader_params)
    databatch = next(data_loader.train.__iter__())
    model_parameters = NonSequentialRegression.get_parameters(data_loader)
    model = NonSequentialRegression(**model_parameters)
    model_optimizers = defaultdict(dict)
    model_optimizers["optimizer"]["opt"] = AdamW(model.parameters())
    model_optimizers["optimizer"]["scheduler"] = None
    model_optimizers["optimizer"]["grad_norm"] = 1.0
    y_predict = model(databatch)
    y_target = databatch["reward"]
    loss = model.loss(y_predict, y_target)
    print(loss)
    print("=== Train Step ===")
    loss = model.train_step(databatch, model_optimizers, 0)
    print("=== Before Update ===")
    print(loss)
    y_predict = model(databatch)
    y_target = databatch["reward"]
    loss = model.loss(y_predict, y_target)
    print("=== After Update ===")
    print(loss)
    print("=== Validate Step ===")
    loss = model.validate_step(databatch)
    print(loss)
    del y_target
    del loss

    model_parameters = NonSequentialClassifier.get_parameters(data_loader)
    model = NonSequentialClassifier(**model_parameters)
    model_optimizers = defaultdict(dict)
    model_optimizers["optimizer"]["opt"] = AdamW(model.parameters())
    model_optimizers["optimizer"]["scheduler"] = None
    model_optimizers["optimizer"]["grad_norm"] = 1.0
    y_predict = model(databatch)
    y_target = databatch["reward"]
    loss = model.loss(y_predict, y_target)
    print(loss)
    print("=== Train Step ===")
    loss = model.train_step(databatch, model_optimizers, 0)
    print("=== Before Update ===")
    print(loss)
    y_predict = model(databatch)
    y_target = databatch["reward"]
    loss = model.loss(y_predict, y_target)
    print("=== After Update ===")
    print(loss)
    print("=== Validate Step ===")
    loss = model.validate_step(databatch)
    print(loss)
    del y_target
    del loss

    # TEST BERT MODEL

    dataloader_params = {"root_dir": data_dir, "batch_size": 2, "transformer_name": "bert"}
    data_loader = TopicDataLoader("cpu", **dataloader_params)
    databatch = next(data_loader.train.__iter__())

    model_parameters = SequentialRegression.get_parameters(data_loader)
    model = SequentialRegression(**model_parameters)
    model_optimizers = defaultdict(dict)
    model_optimizers["optimizer"]["opt"] = AdamW(model.parameters())
    model_optimizers["optimizer"]["scheduler"] = None
    model_optimizers["optimizer"]["grad_norm"] = 1.0
    y_predict = model(databatch)
    y_target = databatch["reward"]
    loss = model.loss(y_predict, y_target)
    print(loss)
    print("=== Train Step ===")
    loss = model.train_step(databatch, model_optimizers, 0)
    print("=== Before Update ===")
    print(loss)
    y_predict = model(databatch)
    y_target = databatch["reward"]
    loss = model.loss(y_predict, y_target)
    print("=== After Update ===")
    print(loss)
    print("=== Validate Step ===")
    loss = model.validate_step(databatch)
    print(loss)
    del y_target
    del loss

    model_parameters = SequentialClassifier.get_parameters(data_loader)
    model = SequentialClassifier(**model_parameters)
    model_optimizers = defaultdict(dict)
    model_optimizers["optimizer"]["opt"] = AdamW(model.parameters())
    model_optimizers["optimizer"]["scheduler"] = None
    model_optimizers["optimizer"]["grad_norm"] = 1.0
    y_predict = model(databatch)
    y_target = databatch["reward"]
    loss = model.loss(y_predict, y_target)
    print(loss)
    print("=== Train Step ===")
    loss = model.train_step(databatch, model_optimizers, 0)
    print("=== Before Update ===")
    print(loss)
    y_predict = model(databatch)
    y_target = databatch["reward"]
    loss = model.loss(y_predict, y_target)
    print("=== After Update ===")
    print(loss)
    print("=== Validate Step ===")
    loss = model.validate_step(databatch)
    print(loss)
    del y_target
    del loss

    # TEST ROBERTA MODEL

    dataloader_params = {"root_dir": data_dir, "batch_size": 2, "transformer_name": "roberta"}
    data_loader = TopicDataLoader("cpu", **dataloader_params)
    databatch = next(data_loader.train.__iter__())

    model_parameters = SequentialRegression.get_parameters(data_loader)
    model_parameters["backbone_name"] = "roberta"
    model = SequentialRegression(**model_parameters)
    y_predict = model(databatch)
    y_target = databatch["reward"]
    loss = model.loss(y_predict, y_target)
    print(loss)
    del y_target
    del loss

    # TEST ALBERT MODEL

    dataloader_params = {"root_dir": data_dir, "batch_size": 2, "transformer_name": "albert"}
    data_loader = TopicDataLoader("cpu", **dataloader_params)
    databatch = next(data_loader.train.__iter__())

    model_parameters = SequentialRegression.get_parameters(data_loader)
    model_parameters["backbone_name"] = "albert"
    model = SequentialRegression(**model_parameters)
    y_predict = model(databatch)
    y_target = databatch["reward"]
    loss = model.loss(y_predict, y_target)
    print(loss)
    del y_target
    del loss

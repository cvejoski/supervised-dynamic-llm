"""
Main training script.
1. Loads a config file containing all the model's parameters.
2. Sets up training procedures and initializes model, trainer and optimizers.
3. Trains the model.
"""
# !/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List

import click
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from supervisedllm.models import AModel
from supervisedllm.utils.helper import (
    create_instance,
    expand_params,
    get_device,
    get_model_default_parameters,
    load_params,
)

_logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "-c",
    "--config",
    "cfg_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to config file containing the training parameteres",
)
@click.option("--quiet", "log_level", flag_value=logging.WARNING, default=True)
@click.option("-v", "--verbose", "log_level", flag_value=logging.INFO)
@click.option("-vv", "--very-verbose", "log_level", flag_value=logging.DEBUG)
@click.option("-d", "--debug", "debug", is_flag=True, default=False)
@click.option("-nc", "--no-cuda", "no_cuda", is_flag=True, default=False)
@click.option("-r", "--resume-training", "resume", is_flag=True, default=False, help="resume training from the last checkpoint")
@click.option("-rf", "--resume-from", "resume_from", type=click.Path(exists=True), help="path to checkpoint.pth to resume from")
def main(cfg_path: Path, log_level: int, debug: bool, resume: bool, resume_from: str, no_cuda: bool):
    logging.basicConfig(
        stream=sys.stdout, level=log_level, datefmt="%Y-%m-%d %H:%M", format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    params = load_params(cfg_path, _logger)
    gs_params = expand_params(params)
    if resume_from is None and resume:
        resume_from = get_latest_checkpoint(params)
    train(debug, gs_params, params, resume_from, no_cuda)


def train(debug: bool, gs_params: List[dict], params: dict, resume: bool, no_cuda: bool):
    """Train the model given parameters.

    Args:
        debug (bool): Training the model in debugging mode. No usage of parallelization and multiprocessing
        gs_params (List[dict]): Grid search parameters. Expanded from the compact parameters.
        params (dict): Training parameters compact
        resume (bool): If ``True`` resume the training from previous checkpoint.
        no_cuda (bool): If ``True`` no CUDA device is used.
    """
    num_workers = params["num_workers"]
    world_size = params.get("world_size", 1)
    distributed = params.get("distributed", False)
    if debug:
        train_in_debug(gs_params, resume, no_cuda)
    elif distributed:
        train_distributed(world_size, gs_params, resume, no_cuda)
    else:
        train_parallel(num_workers, gs_params, resume, no_cuda)


def train_in_debug(gs_params: List[dict], resume: bool, no_cuda: bool):
    """Train the model in debugging mode. No multiprocessing.

    Args:
        gs_params (List[dict]): Grid search parameters. Expanded from the compact parameters.
        resume (bool): If ``True`` resume the training from previous checkpoint.
        no_cuda (bool): If ``True`` no CUDA device is used.
    """
    for search in gs_params:
        train_params(search, resume, True, no_cuda)


def train_parallel(num_workers: int, gs_params: List[dict], resume: bool, no_cuda: bool):
    """Train the model in parallel.

    In case of grid search this funcitons start in parallel training for different parameters in
    the grid search. The number of the parallel trainings is defined by ``num_workers`` parameter.

    Args:
        num_workers (int): Used for parallel training
        gs_params (List[dict]): Grid search parameters
        resume (bool): If ``True`` resume the training from previous checkpoint.
        no_cuda (bool): If ``True`` no CUDA device is used.
    """
    if num_workers > 0:
        p = Pool(num_workers)
        p.map(partial(train_params, resume=resume, no_cuda=no_cuda), gs_params)
    else:
        for search in gs_params:
            train_params(search, resume, False, no_cuda)


def train_distributed(world_size: int, gs_params: List[dict], resume: bool, no_cuda: bool):
    """Train the model in distributed data parallel mode.

    ..note::
        Using `Distributed training` and `Parallel training` at the same time it is not supported.



    Args:
        world_size (int): Number of devices used for the distributed training
        gs_params (List[dict]): Grid search parameters
        resume (bool): If ``True`` resume the training from previous checkpoint.
        no_cuda (bool): If ``True`` no CUDA device is used.
    """
    for param in gs_params:
        processes = []

        for rank in range(world_size):
            p = mp.Process(target=train_params_distributed, args=(rank, world_size, param, no_cuda))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()


def train_params_distributed(rank, world_size, params, no_cuda):
    if rank == 0:
        _logger.info("Name of the Experiment: %s on rank %s", {params["name"]}, {rank})
    print(f"Name of the Experiment: {params['name']} on rank {rank} and world size {world_size}")
    setup(rank, world_size)
    device = get_device(params, rank, _logger, no_cuda=no_cuda)
    data_loader = create_instance("data_loader", params, device, rank, world_size)

    model = create_instance("model", params)
    # Optimizers
    optimizers = init_optimizer(model, params)

    # Trainer
    is_distributed = True
    resume = False
    trainer = create_instance("trainer", params, model, optimizers, is_distributed, resume, params, data_loader)
    best_model = trainer.train()
    with open(os.path.join(params["trainer"]["logging"]["logging_dir"], "best_models.txt"), "a+") as f:
        f.write(str(best_model) + "\n")
    cleanup()


def train_params(params: dict, resume: bool, debug: bool = False, no_cuda: bool = False):
    if debug:
        torch.manual_seed(params["seed"])
    _logger.info("Name of the Experiment: %s", params["name"])
    device = get_device(params, no_cuda=no_cuda)
    data_loader = create_instance("data_loader", params, device)
    model_parameters = get_model_default_parameters(params["model"], data_loader)
    params["model"]["args"] = model_parameters | params["model"]["args"]
    model = create_instance("model", params)

    # Optimizers
    optimizers = init_optimizer(model, params)

    # Trainer
    trainer = create_instance("trainer", params, model, optimizers, False, resume, params, data_loader)
    best_model = trainer.train()
    with open(os.path.join(params["trainer"]["logging"]["logging_dir"], "best_models.txt"), "a+") as f:
        f.write(str(best_model) + "\n")


def init_optimizer(model: AModel, params: dict):
    """Create the optimizer(s) used during training.

    Args:
        model (AModel): Model to be trained
        params (dict): Optimizer parameters

    Returns:
        dict: Optimizer(s) as optimizer name/optimizer pairs.
    """
    optimizers = dict()

    optimizer = create_instance("optimizer", params, model.parameters())
    optimizers["optimizer"] = {
        "opt": optimizer,
        "grad_norm": params["optimizer"].get("gradient_norm_clipping", None),
        "min_lr_rate": params["optimizer"].get("min_lr_rate", 1e-8),
    }
    return optimizers


def get_latest_checkpoint(params: Dict, best_model: bool = False) -> str:
    save_dir = os.path.join(params["trainer"]["save_dir"], params["name"])
    if not os.path.exists(save_dir):
        raise FileNotFoundError()
    latest_run = os.path.join(save_dir, sorted(os.listdir(save_dir))[-1])
    if best_model and os.path.exists(os.path.join(latest_run, "best_model.pth")):
        return os.path.join(latest_run, "best_model.pth")
    checkpoints = [x for x in os.listdir(latest_run) if x.endswith(".pth")]
    if not checkpoints:
        raise FileNotFoundError(f"No .pth files in directory {latest_run}.")
    latest_checkpoint = sorted(checkpoints)[-1]
    return os.path.join(save_dir, latest_run, latest_checkpoint)


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

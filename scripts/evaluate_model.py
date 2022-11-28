# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""Main evaluation script.

For the evaluation the scripts requres as input:

    1. path to trained model,
    2. path to a dataset on which the model will be evaluated.

As output the script will create directory where it will be stored 4 files:

    1. mode.pth: copy of the model which was used for evaluations,
    2. paramters.yaml: hyperparameters used for thraining the model,
    3. results.json: result file where the model prediction per image are stored, as
        well as the result metrics on the whole dataset (see below the detail structure of the json),
    4. mr-fppi.png: curve for IoU 50% and IoU 75%
"""
import logging
import os
import sys
from collections import defaultdict
from multiprocessing.dummy import Pool
from pathlib import Path
from typing import Dict, List, Union

import click
import torch
import torch.distributed as dist
import tqdm

import kiwissenbase
import kiwissenbase.data as kiwi_data
from kiwissenbase.metrics import (
    bbox_category,
    calc_bbox_metrics,
    get_prediction_with_score,
)
from kiwissenbase.models import AModel
from kiwissenbase.typing import Predictions, Targets
from kiwissenbase.utils.file_operations import copy_files, save_json, save_yaml
from kiwissenbase.utils.helper import (
    dict_of_tensors_2_dict_of_lists,
    dict_of_tensors_2_dict_of_numbers,
    get_device,
    load_model_parameters,
    load_trained_model,
    load_training_dataloader,
)
from kiwissenbase.utils.lamr import compute_LAMR
from kiwissenbase.utils.visualization import plot_fppi_vs_mr

LOGGER = logging.getLogger(__name__)
CATEGORIES = ["reasonable", "small", "occlusion", "all"]


@click.command()
@click.option("-m", "--model-dir", required=True, type=click.Path(exists=True), help="Path to the stored trained model.")
@click.option(
    "-s", "split", required=True, type=click.Choice(["train", "validate", "test"]), default="test", help="Split that is evaluated on."
)
@click.option(
    "-rdir",
    "--data-root-dir",
    required=False,
    default=None,
    type=click.Path(exists=True),
    help="Path to the root directory of the dataset.",
)
@click.option(
    "-o",
    "--output-dir",
    required=True,
    type=click.Path(exists=False, dir_okay=True),
    help="Path to the output json file containing the results.",
)
@click.option(
    "--dataset",
    "dataset",
    required=False,
    type=click.Choice(["caltech", "citypersons", "eurocitypersons"]),
    default=None,
    help="Split that is evaluated on.",
)
@click.option("-bs", "--batch-size", type=int, default=8, help="Batch size.")
@click.option("-d", "--gpus", "gpus", default=[], type=list, help="list of GPUs to be used")
@click.option("--distributed", is_flag=True, help="Generate passwords using multiple GPUs!")
@click.option("--num-workers", type=int, default=1, help="Number of parallel workers for evaluating the results!")
@click.option(
    "--evaluation-custom-name",
    type=str,
    default=None,
    help="Append to the output path a custom name for the evaluation where the results will be stored.",
)
@click.option("-nc", "--no-cuda", "no_cuda", is_flag=True, help="Disable GPU support")
@click.option("--quiet", "log_level", flag_value=logging.WARNING, default=True)
@click.option("-v", "--verbose", "log_level", flag_value=logging.INFO)
@click.option("-vv", "--very-verbose", "log_level", flag_value=logging.DEBUG)
@click.version_option(kiwissenbase.__version__)
def main(
    model_dir: Path,
    split: str,
    data_root_dir: Path,
    output_dir: Path,
    dataset: str,
    batch_size: int,
    gpus: List[int],
    evaluation_custom_name: str,
    no_cuda: bool,
    num_workers: int,
    distributed: bool,
    log_level: int,
):
    logging.basicConfig(
        stream=sys.stdout, level=log_level, datefmt="%Y-%m-%d %H:%M", format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # pylint: disable=global-variable-undefined
    global NUM_WORKERS_
    NUM_WORKERS_ = num_workers

    model_dir = Path(model_dir)
    parameters = load_model_parameters(model_dir)
    experiment_name = f'{parameters["name"]}{"" if dataset is None else "_evaluated_" + dataset}_{split}'
    output_dir = Path(output_dir)
    if evaluation_custom_name is None:
        output_dir /= experiment_name
    else:
        output_dir /= evaluation_custom_name

    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device({"gpus": gpus}, 0, no_cuda=no_cuda)
    model = load_trained_model(model_dir, device)
    if dataset is None:
        loader = load_training_dataloader(model_dir, data_root_dir, batch_size, device, True)
    else:
        loader = kiwi_data.all_dataloaders[dataset](
            device,
            root_dir=data_root_dir,
            batch_size=batch_size,
            num_workers=4,
            different_size_target=True,
            group_pedestrian_classes=True,
            return_image_path=True,
            subset="annotated-pedestrians",
        )

    results = evaluate_single_gpu(model, loader, split, device)

    agg_metrics = results["aggregate_metrics"]
    lamr = compute_LAMR(agg_metrics)
    plot_fppi_vs_mr(
        {"iou_50": {"faster_rcnn": agg_metrics["iou_50"]}, "iou_75": {"faster_rcnn": agg_metrics["iou_75"]}},
        lamr_value=lamr,
        save_path=output_dir / "mr-fppi.png",
    )

    save_json(output_dir / "results.json", results)
    save_yaml(output_dir / "parameters.yaml", parameters)
    copy_files(model_dir, output_dir / "model.pth")


def evaluate_single_gpu(model: AModel, loader: kiwi_data.ADataLoader, split: str, device: torch.DeviceObjType) -> dict:
    is_target_present = getattr(loader, split).dataset.has_targets
    if is_target_present:
        img_paths, predictions, targets = image_inference_with_targets(loader, model, split, device)
    else:
        img_paths, predictions = image_inference_without_targets(loader, model, split, device)
        targets = []

    results = {}
    results["predictions"] = calculate_metrics_image_level(img_paths, predictions, targets)
    if targets:
        results["aggregate_metrics"] = calculate_aggregate_metrics(predictions, targets)

    return results


def image_inference_with_targets(data_loader: kiwi_data.ADataLoader, model: AModel, split: str, device: torch.DeviceObjType):
    with torch.no_grad():
        targets = []
        predictions = []
        img_paths = []
        for img_path, images, y_target in tqdm.tqdm(
            getattr(data_loader, split), desc="Inference on Images", total=data_loader.n_batches(split)
        ):
            x = list(image.to(device, non_blocking=True) for image in images)
            y_prediction = model(x)
            y_prediction = [{k: v.cpu() for k, v in t.items()} for t in y_prediction]
            targets.extend(y_target)
            predictions.extend(y_prediction)
            img_paths.extend(img_path)
    return img_paths, predictions, targets


def image_inference_without_targets(data_loader: kiwi_data.ADataLoader, model: AModel, split: str, device: torch.DeviceObjType):
    with torch.no_grad():
        predictions = []
        img_paths = []
        for img_path, images in tqdm.tqdm(getattr(data_loader, split), desc="Inference on Images", total=data_loader.n_batches(split)):
            x = list(image.to(device, non_blocking=True) for image in images)
            y_prediction = model(x)
            y_prediction = [{k: v.cpu() for k, v in t.items()} for t in y_prediction]
            predictions.extend(y_prediction)
            img_paths.extend(img_path)
    return img_paths, predictions


def calculate_metrics_image_level(img_paths: List[str], predictions: Predictions, targets: Targets) -> Dict[str, Union[float, list]]:
    rows = []
    pool = Pool(NUM_WORKERS_)
    inputs = list(zip(list(range(len(img_paths))), img_paths, predictions, targets))
    images = tqdm.tqdm(
        pool.imap_unordered(lambda arg: _calc_img_metrics(*arg), inputs, 10), total=len(img_paths), desc="Calculate image level metrics"
    )

    for img in images:
        rows.append(img)

    pool.close()
    pool.join()
    return rows


def _calc_img_metrics(_id, p, y_hat, y):
    row = defaultdict(dict)
    row["id"] = _id
    row["img_path"] = p
    row["img_name"] = Path(p).stem
    row["prediction"] = dict_of_tensors_2_dict_of_lists(y_hat)
    row["target"] = dict_of_tensors_2_dict_of_lists(y)
    row["num_targets"] = len(list(filter(lambda x: x > 0, row["target"]["labels"])))
    row["iou_50"] = []
    row["iou_75"] = []
    row["num_predictions"] = []
    iou_50 = defaultdict(list)
    iou_75 = defaultdict(list)
    y_categorized = bbox_category(y)
    for score in range(0, 101, 1):  # FIXME Use dynamic programming to improve the speed
        predictions_filtered = get_prediction_with_score([y_hat], score / 100.0)
        image_metrics_iou50 = calc_bbox_metrics(predictions_filtered, [y_categorized], 0.5)
        image_metrics_iou50 = dict_of_tensors_2_dict_of_numbers(image_metrics_iou50)
        image_metrics_iou75 = calc_bbox_metrics(predictions_filtered, [y_categorized], 0.75)
        image_metrics_iou75 = dict_of_tensors_2_dict_of_numbers(image_metrics_iou75)
        row["num_predictions"].append(len(predictions_filtered[0]["scores"]))
        for c in CATEGORIES:
            iou_50[f"fn_{c}"].append(image_metrics_iou50[f"fn_{c}"])
            iou_75[f"fn_{c}"].append(image_metrics_iou75[f"fn_{c}"])
            iou_50[f"fp_{c}"].append(image_metrics_iou50[f"fp_{c}"])
            iou_75[f"fp_{c}"].append(image_metrics_iou75[f"fp_{c}"])

    row = {**row, **dict_2_list_of_dict(iou_50, iou_75)}
    return row


def calculate_aggregate_metrics(predictions, targets):
    iou_50 = defaultdict(list)
    iou_75 = defaultdict(list)
    y_target_categorized = list(map(bbox_category, targets))
    for score in tqdm.trange(0, 101, 1):  # FIXME this can be parallelized @Kostadin,
        predictions_filtered = get_prediction_with_score(predictions, score / 100.0)
        bbox_metrics_iou50 = calc_bbox_metrics(predictions_filtered, y_target_categorized, 0.5)
        bbox_metrics_iou50 = dict_of_tensors_2_dict_of_numbers(bbox_metrics_iou50)
        bbox_metrics_iou75 = calc_bbox_metrics(predictions_filtered, y_target_categorized, 0.75)
        bbox_metrics_iou75 = dict_of_tensors_2_dict_of_numbers(bbox_metrics_iou75)
        for k, v in bbox_metrics_iou50.items():
            iou_50[k].append(v)
            iou_75[k].append(bbox_metrics_iou75[k])
    return dict_2_list_of_dict(iou_50, iou_75)


def dict_2_list_of_dict(iou_50: dict, iou_75: dict):
    results = defaultdict(list)
    for c in CATEGORIES:
        cat_result_50 = {}
        cat_result_75 = {}
        for k in iou_50.keys():
            splits = k.split("_")
            if c in k:
                if len(splits) > 2:
                    metric = "_".join(splits[:-1])
                else:
                    metric = splits[0]
                cat_result_50["category"] = c
                cat_result_50[metric] = iou_50[k]
                cat_result_75["category"] = c
                cat_result_75[metric] = iou_75[k]
            else:
                if splits[-1] not in CATEGORIES:
                    cat_result_50["category"] = c
                    cat_result_50[k] = iou_50[k]
                    cat_result_75["category"] = c
                    cat_result_75[k] = iou_75[k]

        results["iou_50"].append(cat_result_50)
        results["iou_75"].append(cat_result_75)
    return results


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

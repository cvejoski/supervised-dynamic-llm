from kiwissenbase import test_resources_path
from kiwissenbase.utils.file_operations import load_json
from kiwissenbase.metrics import calc_mr
from torchvision.ops import box_iou
import torch
import numpy as np


def unit_img():
    results_path = test_resources_path / "caltech_fasterrcnn_validate-1011_101531_ep28.json"

    predictions_all = load_json(results_path)["predictions"]

    img = predictions_all[10]
    img_bb_predictions = img["prediction"]["boxes"]
    is_pedestrian = torch.tensor(img["target"]["labels"]) == 1
    print(is_pedestrian)
    img_bb_targets = torch.tensor(img["target"]["boxes"])[is_pedestrian]

    print(img_bb_targets)

    print(img["img_name"])

    n_targets = int(len(img_bb_targets))
    n_predictions = int(len(img_bb_predictions))
    tp, fp, fn = 0, 0, 0
    pairwise_iou = box_iou(img_bb_targets, torch.tensor(img_bb_predictions))
    print(pairwise_iou)

    index_tp = []
    for pair in pairwise_iou:
        check = int(torch.count_nonzero(pair >= 0.5))
        print("check: ", check)
        print(torch.argwhere(pair >= 0.5).tolist())
        if check > 0:
            index_tp.extend(torch.argwhere(pair >= 0.5).tolist())
        else:
            fn += 1

    for index, value in enumerate(index_tp):
        index_tp[index] = value[0]

    print(set(index_tp))
    tp = len(set(index_tp))
    fp = int(n_predictions - tp)
    mr = fn / n_targets

    print("num_target", n_targets)
    print("num_prediction", n_predictions)
    print("TP", tp)
    print("FP", fp)
    print("FN", fn)
    print("MR", mr)


testing = unit_img()

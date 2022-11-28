import json

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from kiwissenbase.metrics import calc_mr_category


@pytest.mark.skip()
def test_plot():
    # load 1 batch (32 images) from the evaluation step => 32 targets and 32 predictions
    with open(r"C:\Users\thuynh\PycharmProjects\ki-wissen-base\tests\plot_MR_fppi\predictions_10.json") as f:
        predictions = json.load(f)
    with open(r"C:\Users\thuynh\PycharmProjects\ki-wissen-base\tests\plot_MR_fppi\targets_10.json") as f:
        targets = json.load(f)

    # change datatype of bboxes in targets from list to tensor
    # for target in targets:
    #     for keys, values in target.items():
    #         if keys == "labels":
    #             target[keys] = torch.tensor(values)
    #         else:
    #             for idx, value in enumerate(values):
    #                 new_value = torch.tensor(value)
    #                 values[idx] = new_value
    #             target[keys] = values

    for target in targets:
        for keys, values in target.items():
            target[keys] = torch.tensor(values)

    # score_index_total is the list of 32 elements (or 32 images of 1 batch), each element is the dict which keys are the
    # scores (from 0.1 to 1.0) and values are the index of the corresponding bboxes from prediction of 1 image.
    score_index_total = []

    # range of confident scores
    _range = list(np.linspace(1.0, 0.0, num=101))

    for prediction in predictions:
        score_index = {}
        for i in _range:
            score_index[str(round(i, 3))] = []
        for key, value in score_index.items():
            for index, score in enumerate(prediction["scores"]):
                key = float(key)
                if score >= key:
                    score_index[str(key)].append(index)
                else:
                    break
        score_index_total.append(score_index)

    fn_perscore_total = {}
    fp_perscore_total = {}
    n_targets_total = [0, 0, 0, 0]
    n_imgs_total = 0

    for i in _range:
        key = str(round(i, 3))
        fn_perscore_total[key] = [0, 0, 0, 0]
        fp_perscore_total[key] = [0, 0, 0, 0]

    for num_img, score_index in enumerate(score_index_total):
        y_element = {}
        for score_thresh, index_bbox in score_index.items():
            if index_bbox == []:
                y_element["boxes"] = [torch.empty((0, 4), dtype=torch.int64)]
                y_element["labels"] = []
                y_element["scores"] = []

            elif index_bbox != []:
                y_element["boxes"] = [torch.tensor(predictions[num_img]["boxes"][i]) for i in index_bbox]
                y_element["labels"] = [predictions[num_img]["labels"][i] for i in index_bbox]
                y_element["scores"] = [predictions[num_img]["scores"][i] for i in index_bbox]

            fn, fp, n_targets, n_imgs = calc_mr_category([y_element], [targets[num_img]])
            fn_perscore_total[score_thresh] = [a + b for a, b in zip(fn_perscore_total[score_thresh], fn)]
            fp_perscore_total[score_thresh] = [a + b for a, b in zip(fp_perscore_total[score_thresh], fp)]

        n_targets_total = [a + b for a, b in zip(n_targets_total, n_targets)]
        n_imgs_total += 1

    print("fn: \n", fn_perscore_total)
    print("fp: \n", fp_perscore_total)
    print("num of imgs: \n", n_imgs_total)
    print("num of targets: \n", n_targets_total)

    fppi_accum_perscore_total = {}
    mr_accum_perscore_total = {}
    for keys, values in fp_perscore_total.items():
        fppi_accum_perscore_total[keys] = [a / n_imgs_total for a in fp_perscore_total[keys]]
        mr_accum_perscore_total[keys] = [a / b for a, b in zip(fn_perscore_total[keys], n_targets_total)]
    print("fppi_accum", fppi_accum_perscore_total)
    print("mr_accum", mr_accum_perscore_total)

    # PLOT PART
    x_axis = []  # fppi
    y_axis = []  # MR
    for key, values in mr_accum_perscore_total.items():
        x_axis.append(fppi_accum_perscore_total[key][0])
        y_axis.append(values[0])

    thresh = np.arange(0, 1, 0.01)
    plt.plot(x_axis, y_axis)
    plt.plot(x_axis[::10], y_axis[::10], ls="", marker="o")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(0.01, 1)
    plt.xlim(0.001, 100)
    plt.grid(True, which="both", linestyle="--")
    plt.xlabel("FPPI")
    plt.ylabel("MR")
    plt.title("MR-FPPI for gap space 0.01")
    plt.show()

import logging
import os
import re
import numpy as np
import scipy.interpolate

from kiwissenbase import test_resources_path
from kiwissenbase.utils.file_operations import load_json

LOGGER = logging.getLogger(__name__)


def compute_LAMR(result_path):
    # results_path = test_resources_path / "caltech_fasterrcnn_validate/faster_rcnn_caltech/results.json"
    # print(results_path)
    # results_path = test_resources_path / "citypersons_fasterrcnn_validate.json"

    results = load_json(results_path)["aggregate_metrics"]

    mr_reasonable = np.array(results["iou_50"][0]["MR"])
    fppi_reasonable = np.array(results["iou_50"][0]["fppi"])
    num_samples = 9

    nine_points = np.logspace(-2.0, 0.0, num_samples)
    print("9 points: ", nine_points)

    interpolated = scipy.interpolate.interp1d(fppi_reasonable, mr_reasonable, kind="slinear", fill_value=(1.0, 0.0), bounds_error=False)(
        nine_points
    )
    print("interpolation: ", interpolated)
    log_mr = np.log(interpolated)
    print("log_mr", log_mr)

    return np.exp(np.average(log_mr))


"x y axis same as paper, add points (scores) on curve, add LAMR into the plotting," "test lamr on diff. epochs"


# lamr = compute_LAMR()
# print(lamr)

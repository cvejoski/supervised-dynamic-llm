import logging
import os

import pytest

from kiwissenbase import test_resources_path
from kiwissenbase.utils.file_operations import load_json
from kiwissenbase.utils.lamr import compute_LAMR
from kiwissenbase.utils.visualization import plot_fppi_vs_mr

LOGGER = logging.getLogger(__name__)


@pytest.mark.skip
def test_fppi_mr_plot():
    results_path = test_resources_path / "caltech_fasterrcnn_validate/faster_rcnn_caltech/results.json"
    # results_path = test_resources_path / "citypersons_fasterrcnn_validate.json"
    print(results_path)
    split = os.path.split(results_path)[-2].split("_")
    print(split)
    save_path = ""
    results = load_json(results_path)["aggregate_metrics"]

    lamr = compute_LAMR(results)

    if "citypersons" in split:
        save_path = "citypersons"
        print("city")
    elif "caltech" in split:
        save_path = "caltech"
        print("cal")
    plot_fppi_vs_mr(
        {"iou_50": {"faster_rcnn": results["iou_50"]}, "iou_75": {"faster_rcnn": results["iou_75"]}}, lamr_value=lamr, save_path=save_path
    )
    assert True

import numpy as np
import scipy.interpolate


def compute_LAMR(results: dict) -> dict:
    """Define the log average miss-rate (LAMR).

    Reference: <https://eurocity-dataset.tudelft.nl/eval/benchmarks/detection>.

    Args:
        results (dict): dictionary with image evaluations

    Returns:
        dict:
    """
    dict_lamr = {}
    for iou, iou_value in results.items():
        mr_reasonable = np.array(results[iou][0]["MR"])
        fppi_reasonable = np.array(results[iou][0]["fppi"])
        num_samples = 9
        nine_points = np.logspace(-2.0, 0.0, num_samples)

        interpolated = scipy.interpolate.interp1d(
            fppi_reasonable, mr_reasonable, kind="slinear", fill_value=(1.0, 0.0), bounds_error=False
        )(nine_points)
        log_mr = np.log(interpolated)
        dict_lamr[iou] = np.round(np.exp(np.average(log_mr)), 3)

    return dict_lamr

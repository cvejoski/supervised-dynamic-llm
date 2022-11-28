import os
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# plt.style.use("_mpl-gallery")
font = {"family": "normal", "size": 22}

matplotlib.rc("font", **font)


def plot_fppi_vs_mr(data: Dict[str, Dict[str, List[float]]], lamr_value, save_path: str = None):
    """Plot fppi vs MR plot.

    Args:
        data (Dict[str, Dict[str, List[float]]]): data to be displayed
        save_path (str, optional): path where the image will be stored. Defaults to None.
    """
    # print(lamr_value)
    fig, axis = plt.subplots(nrows=len(data), ncols=1, figsize=(15, 20))
    for i, (iou, iou_values) in enumerate(data.items()):
        for model, values in iou_values.items():
            axis[i].set_title(f"{iou} - LAMR on [10^-2,1] = {lamr_value[iou]}")
            axis[i].plot(values[0]["fppi"][::-1], values[0]["MR"][::-1], label=model)
            axis[i].plot(values[0]["fppi"][::11], values[0]["MR"][::11], ls="", marker="o", label="confidence score 0.1-1.0")
            axis[i].set_xscale("log")
            axis[i].set_yscale("log")
            axis[i].set_xlabel("fppi")
            axis[i].set_ylabel("miss rate")
            axis[i].set_ylim(0.01, 1)
            axis[i].set_xlim(0.001, 50)
            axis[i].set_yticks([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.64, 0.8, 1.0])
            axis[i].set_yticklabels([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.64, 0.8, 1.0], fontsize=12)
            #     axis[i].text(0.1, 0.1, "LAMR on [10^-2,1] = {}".format(lamr_value), ha='left', va='top', style='italic', bbox={
            # 'facecolor': 'green', 'alpha': 0.5, 'pad': 10})
            axis[i].grid()
            axis[i].legend()
    # plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_fig(fig, save_dir: str, save_name: str):
    """Save figure as image.

    Args:
        fig (plt.Figure): figure to be saved
        save_dir (str): directory where the figure will be stored
        save_name (str): name of the image
    """
    fig.savefig(
        os.path.join(save_dir, f"{save_name}.png"),
        format="png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()


def imshow(img: np.ndarray):
    """Show image as figure.

    Args:
        img (np.ndarray): we want to show.
    """
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

import math
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
import torchvision
from torch.nn.modules.loss import CrossEntropyLoss, _Loss

from .typing import Predictions, Targets
from .utils import param_scheduler as p_scheduler


def calc_bbox_metrics(y: Predictions, y_target: Targets, iou_thresh: float, device=None) -> dict:
    """Calculate bounding box metrics.

    Calculate `MR`, `fppi`, `fn` and `fp` for the different bbox categories, `Reasonable`, `Reasonable_small`,
    `Reasonable_occ` and `all`

    Args:
        y (Predictions): predictions
        y_target (Targets): target predictions
        iou_thresh (float): intersection over union threshold
        device (torch.device, optional): device where the tensors are located. Defaults to None.

    Returns:
        dict: bbox metrics
    """
    _, _, fn, fp, n_targets, n_imgs = calc_mr_category(y, y_target, iou_thresh)

    stats = dict()

    stats["n_imgs"] = torch.tensor(n_imgs, device=device)
    if n_targets["reasonable"] > 0:
        stats["MR"] = torch.tensor(fn["reasonable"] / n_targets["reasonable"], device=device)
    else:
        stats["MR"] = torch.tensor(0.0, device=device)
    # pylint: disable=consider-using-dict-items,consider-iterating-dictionary
    for cat_name in fn.keys():
        stats["fn_" + cat_name] = torch.tensor(fn[cat_name], device=device)
        stats["fp_" + cat_name] = torch.tensor(fp[cat_name], device=device)
        stats["n_targets_" + cat_name] = torch.tensor(n_targets[cat_name], device=device)

        stats["MR_" + cat_name] = torch.tensor(fn[cat_name] / n_targets[cat_name] if n_targets[cat_name] > 0 else 0.0, device=device)

        if n_imgs > 0:
            stats["fppi_" + cat_name] = torch.tensor(fp[cat_name] / n_imgs, device=device)
        else:
            stats["fppi_" + cat_name] = torch.tensor(0.0, device=device)

    return stats


def bbox_category(target_boxes: dict) -> Dict[str, List]:
    """As a numerical measure of the performance, log-average miss rate (MR) is computed by averaging over the precision range of [10e-2; 10e0] FPPI (false positives per image).
    For detailed evaluation, we consider the following 4 subsets:
    1. 'Reasonable': height [50, inf]; visibility [0.65, inf]
    2. 'Reasonable_small': height [50, 75]; visibility [0.65, inf]
    3. 'Reasonable_occ=heavy': height [50, inf]; visibility [0.2, 0.65]
    4. 'All': height [20, inf]; visibility [0.2, inf]
    """

    category_boxes = {"reasonable": [], "small": [], "occlusion": [], "all": []}

    boxes, boxes_vis_ratio, boxes_height = filter_bbox_with_pedestrian(target_boxes)
    for box, box_vis_ratio, box_height in zip(boxes, boxes_vis_ratio, boxes_height):
        for cat_name, values in category_boxes.items():
            if is_bbox_in_cat(cat_name, box_height, box_vis_ratio):
                values.append(box)

    return category_boxes


def filter_bbox_with_pedestrian(target_boxes: dict) -> tuple:
    """Filter bbox that contain pedestrian.

    Args:
        target_boxes (dict): all target boxes

    Returns:
        (dict): target boxes containing only pedestrians
    """
    is_pedestrian = target_boxes["labels"] == 1
    boxes = target_boxes["boxes"][is_pedestrian]
    boxes_vis_ratio = target_boxes["boxesVisRatio"][is_pedestrian]
    boxes_height = target_boxes["boxesHeight"][is_pedestrian]

    return boxes, boxes_vis_ratio, boxes_height


def is_bbox_in_cat(cat_name: str, bbox_height: int, vis_ratio: float) -> bool:
    height_range = {"reasonable": [50, 1e5**2], "small": [50, 75], "occlusion": [50, 1e5**2], "all": [20, 1e5**2]}
    visible_range = {"reasonable": [0.65, 1e5**2], "small": [0.65, 1e5**2], "occlusion": [0.2, 0.65], "all": [0.2, 1e5**2]}
    return not (
        bbox_height < height_range[cat_name][0]
        or bbox_height > height_range[cat_name][1]
        or vis_ratio < visible_range[cat_name][0]
        or vis_ratio > visible_range[cat_name][1]
    )


def calc_vis_ratio(box: Sequence, vis_box: Sequence) -> float:
    """Calculate the visibility ratio of a bounding box.

    Args:
        box (Sequence): _description_
        vis_box (Sequence): _description_

    Returns:
        float: _description_
    """
    fullbox_area = (box[2] - box[0]) * (box[3] - box[1])
    visible_area = (vis_box[2] - vis_box[0]) * (vis_box[3] - vis_box[1])
    vis_ratio = visible_area / fullbox_area
    return vis_ratio


def calc_bbox_height(bbox: Sequence) -> int:
    return abs(bbox[3] - bbox[1])


def calc_mr_category(y: Predictions, y_target: Targets, iou_thresh: float = 0.5):
    iou_imgs = {"reasonable": 0, "small": 0, "occlusion": 0, "all": 0}
    mr_imgs = {"reasonable": 0, "small": 0, "occlusion": 0, "all": 0}
    fn = {"reasonable": 0, "small": 0, "occlusion": 0, "all": 0}
    fp = {"reasonable": 0, "small": 0, "occlusion": 0, "all": 0}
    n_targets = {"reasonable": 0, "small": 0, "occlusion": 0, "all": 0}
    n_imgs = 0
    for target, output in zip(y_target, y):
        pedestrian_output_boxes = output["boxes"][output["labels"] == 1]
        if not "reasonable" in target.keys():
            target = bbox_category(target)

        for cat_name, target_boxes in target.items():
            num_pedestrian_bboxes = pedestrian_output_boxes.size(0)
            if not target_boxes:
                # TODO cal fp here
                fp[cat_name] += num_pedestrian_bboxes
                continue
            elif num_pedestrian_bboxes == 0:
                # TODO cal fn here
                n_target_bboxes = len(target_boxes)
                n_targets[cat_name] += n_target_bboxes
                fn[cat_name] += n_target_bboxes
                mr_imgs[cat_name] += 1  # TODO append list
            else:
                # if not torch.is_tensor(target_boxes[0]):
                # avg_iou = average_iou(torch.stack(target_boxes), torch.stack(pedestrian_output_boxes))
                # iou_imgs[id] += average_iou(torch.stack(target_boxes), torch.stack(pedestrian_output_boxes))
                mr_img, n_targets_img, fn_img, fp_img = calc_mr(torch.stack(target_boxes), pedestrian_output_boxes, iou_thresh)
                mr_imgs[cat_name] += mr_img  # TODO append list
                fn[cat_name] += fn_img
                fp[cat_name] += fp_img
                n_targets[cat_name] += n_targets_img
        n_imgs += 1

    return iou_imgs, mr_imgs, fn, fp, n_targets, n_imgs


def calc_mr(target_boxes: torch.Tensor, predicted_boxes: torch.Tensor, iou_thresh: float = 0.5):
    """Calculates the miss rate of predicted and target pedestrian bbs

    miss rate = fn / (fn + tp) = fn / n_targets = (n_targets - tp) / n_targets

    assumes that the target and predictions refer to the same class
    target_boxes: Nx4 tensor (x1,y1,x2,y2)
    predicted_boxes: Mx4 tensor (x1,y1,x2,y2)
    iou_thresh: float (when overlap of predicted and target bb is larger, then counted as detection)
    """

    pairwise_iou = torchvision.ops.box_iou(target_boxes, predicted_boxes)

    n_targets = int(len(target_boxes))
    n_predictions = int(len(predicted_boxes))
    tp, fp, fn = 0, 0, 0

    # match the target to predicted bbs by max-iou
    while len((pairwise_iou > 0).nonzero()):
        max_iou = torch.max(pairwise_iou)
        max_inds = (pairwise_iou == max_iou).nonzero()
        pairwise_iou[max_inds[0][0], :] = 0.0
        pairwise_iou[:, max_inds[0][1]] = 0.0
        if max_iou >= iou_thresh:
            tp += 1

    # calc miss rate
    fp = int(n_predictions - tp)
    fn = int(n_targets - tp)
    mr = fn / n_targets

    # index_tp = []
    # for pair in pairwise_iou:
    #     check = int(torch.count_nonzero(pair >= iou_thresh))
    #     if check > 0:
    #         index_tp.extend(torch.argwhere(pair >= iou_thresh).tolist())
    #     else:
    #         fn += 1
    #
    # for index, value in enumerate(index_tp):
    #     index_tp[index] = value[0]
    #
    # tp = len(set(index_tp))
    # fp = int(n_predictions - tp)
    # mr = fn / n_targets

    return mr, n_targets, fn, fp


def average_iou(target_boxes: torch.Tensor, predicted_boxes: torch.Tensor):
    """calculates the average iou of predicted and target pedestrian bbs
    assumes that the target and predictions refer to the same class
    target_boxes: Nx4 tensor (x1,y1,x2,y2)
    predicted_boxes: Mx4 tensor (x1,y1,x2,y2)
    no penalty for false positives"""

    all_ious = []

    pairwise_iou = torchvision.ops.box_iou(target_boxes, predicted_boxes)

    # match the target to predicted bbs by max-iou
    while len((pairwise_iou > 0).nonzero()):
        max_iou = torch.max(pairwise_iou)
        max_inds = (pairwise_iou == max_iou).nonzero()
        pairwise_iou[max_inds[0][0], :] = 0.0
        pairwise_iou[:, max_inds[0][1]] = 0.0
        all_ious.append(max_iou.item())
    # add a 0 iou for all non-matched target bbs
    all_ious += [0.0] * (len(target_boxes) - len(all_ious))
    return torch.mean(torch.tensor(all_ious, device=predicted_boxes.device))


def kullback_leibler(mean, sigma, reduction="mean"):
    """
    Kullback-Leibler divergence between Gaussian posterior distr.
    with parameters (mean, sigma) and a fixed Gaussian prior
    with mean = 0 and sigma = 1
    """

    kl = -0.5 * (1 + 2.0 * torch.log(sigma) - mean * mean - sigma * sigma)  # [B, D]
    skl = torch.sum(kl, dim=1)
    if reduction == "mean":
        return torch.mean(skl)
    elif reduction == "sum":
        return torch.sum(skl)
    else:
        return skl


def mim_reg(mean, sigma, reduction="mean"):
    """
    Kullback-Leibler divergence between Gaussian posterior distr.
    with parameters (mean, sigma) and a fixed Gaussian prior
    with mean = 0 and sigma = 1
    """

    D = mean.size(-1)
    dist = 0.25 * (1 + 2.0 * torch.log(sigma) + mean * mean + sigma * sigma)  # [B, D]
    s_dist = torch.sum(dist, dim=1) + 0.5 * D * torch.log(torch.tensor(2 * math.pi))
    if reduction == "mean":
        return torch.mean(s_dist)
    elif reduction == "sum":
        return torch.sum(s_dist)
    else:
        return s_dist


def accuracy(logits, target, reduction="mean"):
    prediciton = torch.argmax(logits, dim=1)
    acc = target.eq(prediciton)
    if reduction == "mean":
        return acc.float().mean()
    elif reduction == "sum":
        return acc.sum()
    else:
        return acc


class ELBO(CrossEntropyLoss):
    r"""This criterion combines :func:`nn.LogSoftmax` and :func:`nn.NLLLoss` in one single class.

    It is useful when training a classification problem with `C` classes.
    If provided, the optional argument :attr:`weight` should be a 1D `Tensor`
    assigning weight to each of the classes.
    This is particularly useful when you have an unbalanced training set.

    The `input` is expected to contain raw, unnormalized scores for each class.

    `input` has to be a Tensor of size either :math:`(minibatch, C)` or
    :math:`(minibatch, C, d_1, d_2, ..., d_K)`
    with :math:`K \geq 1` for the `K`-dimensional case (described later).

    This criterion expects a class index in the range :math:`[0, C-1]` as the
    `target` for each value of a 1D tensor of size `minibatch`; if `ignore_index`
    is specified, this criterion also accepts this class index (this index may not
    necessarily be in the class range).

    The loss can be described as:

    .. math::
        \text{loss}(x, class) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
                       = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    or in the case of the :attr:`weight` argument being specified:

    .. math::
        \text{loss}(x, class) = weight[class] \left(-x[class] + \log\left(\sum_j \exp(x[j])\right)\right)

    The losses are averaged across observations for each minibatch.

    Can also be used for higher dimension inputs, such as 2D images, by providing
    an input of size :math:`(minibatch, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`,
    where :math:`K` is the number of dimensions, and a target of appropriate shape
    (see below).


    Args:
        weight (Tensor, optional): a manual rescaling weight given to each class.
            If given, has to be a Tensor of size `C`
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When :attr:`size_average` is
            ``True``, the loss is averaged over non-ignored targets.
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(N, C)` where `C = number of classes`, or
          :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
          in the case of `K`-dimensional loss.
        - Target: :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`, or
          :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case of
          K-dimensional loss.
        - Output: scalar.
          If :attr:`reduction` is ``'none'``, then the same size as the target:
          :math:`(N)`, or
          :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case
          of K-dimensional loss.

    Examples::

        >>> loss = nn.CrossEntropyLoss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(5)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    __constants__ = ["weight", "ignore_index", "reduction"]

    def __init__(
        self,
        weight=None,
        size_average=None,
        ignore_index=-100,
        reduce=None,
        reduction="mean",
        beta_scheduler=None,
    ):
        super(ELBO, self).__init__(weight, size_average, ignore_index, reduce, reduction)
        if beta_scheduler is None:
            self.b_scheduler = p_scheduler.ConstantScheduler()
        else:
            self.b_scheduler = beta_scheduler

    def forward(self, input, target, mean, sigma, step):
        CE = super(ELBO, self).forward(input, target)
        KL = kullback_leibler(mean, sigma, reduction=self.reduction)
        beta = torch.tensor(self.b_scheduler(step))
        if beta == 0:
            loss = CE
        else:
            loss = CE + beta * KL
        return loss, CE, KL, beta


class Perplexity(CrossEntropyLoss):
    __constants__ = ["weight", "ignore_index", "reduction"]

    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None):
        super(Perplexity, self).__init__(weight, size_average, ignore_index, reduce, "mean")

    def forward(self, input, target):
        loss = super(Perplexity, self).forward(input, target)

        return torch.exp(loss)


class VQ(CrossEntropyLoss):
    def __init__(
        self,
        weight=None,
        size_average=None,
        ignore_index=-100,
        reduce=None,
        reduction="mean",
        beta_scheduler=None,
    ):
        super(VQ, self).__init__(weight, size_average, ignore_index, reduce, reduction)
        if beta_scheduler is None:
            self.b_scheduler = p_scheduler.ConstantScheduler()
        else:
            self.b_scheduler = beta_scheduler

    def forward(self, input, target, z_e_x, z_q_x, step):

        # Reconstruction loss
        loss_rec = super(VQ, self).forward(input, target)
        # Vector quantization objective
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        # Commitment objective
        loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

        loss = loss_rec + loss_vq + loss_commit

        return loss, loss_rec, loss_vq, loss_commit


class WAELoss(CrossEntropyLoss):
    """
    Wasserstein Autoencoder Loss
    """

    __constants__ = ["weight", "ignore_index", "reduction"]

    def __init__(
        self,
        weight=None,
        size_average=None,
        ignore_index=-100,
        reduce=None,
        reduction="mean",
    ):
        super(WAELoss, self).__init__(weight, size_average, ignore_index, reduce, reduction)

    def forward(self, input, target, distance, step):
        CE = super(WAELoss, self).forward(input, target)
        beta = torch.tensor(self.b_scheduler(step))
        if beta == 0:
            loss = CE
        else:
            loss = CE + beta * distance

        return loss, CE, distance, beta


class MSELoss(_Loss):
    r"""Creates a criterion that measures the mean squared error (squared L2 norm) between
    each element in the input :math:`x` and target :math:`y`.

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left( x_n - y_n \right)^2,

    where :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then:

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), &  \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(L),  &  \text{if reduction} = \text{'sum'.}
        \end{cases}

    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
    of :math:`n` elements each.

    The sum operation still operates over all the elements, and divides by :math:`n`.

    The division by :math:`n` can be avoided if one sets ``reduction = 'sum'``.

    Args:
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(N, *)` where :math:`*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input

    Examples::

        >>> loss = tyche.loss.MSELoss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    __constants__ = ["reduction"]

    def __init__(self, size_average=None, reduce=None, reduction="mean", ignore_index=None):
        super(MSELoss, self).__init__(size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        if self.ignore_index is not None:
            ix = target != self.ignore_index
            mse = F.mse_loss(input[ix], target[ix], reduction=self.reduction)
        else:
            mse = F.mse_loss(input, target, reduction=self.reduction)
        return mse


class RMSELoss(MSELoss):
    r"""Creates a criterion that measures the mean squared error (squared L2 norm) between
    each element in the input :math:`x` and target :math:`y`.

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left( x_n - y_n \right)^2,

    where :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then:

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), &  \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(L),  &  \text{if reduction} = \text{'sum'.}
        \end{cases}

    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
    of :math:`n` elements each.

    The sum operation still operates over all the elements, and divides by :math:`n`.

    The division by :math:`n` can be avoided if one sets ``reduction = 'sum'``.

    Args:
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(N, *)` where :math:`*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input

    Examples::

        >>> loss = tyche.loss.RMSELoss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    __constants__ = ["reduction"]

    def __init__(self, size_average=None, reduce=None, reduction="mean", ignore_index=None):
        super(RMSELoss, self).__init__(size_average, reduce, reduction, ignore_index)

    def forward(self, input, target):
        mse = super(RMSELoss, self).forward(input, target)
        return torch.sqrt(mse)


#############################################################################################################################

# TODO Unify IOU functions
# TODO Maybe move Focal Loss to RetinaNet or keep it here?


def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU


def get_prediction_with_score(predictions: List[dict], score_thresh: float) -> List[dict]:
    results: list = []
    for prediction in predictions:
        filter_ix = prediction["scores"] >= score_thresh
        result = dict()
        for k, v in prediction.items():
            result[k] = v[filter_ix]
        results.append(result)
    return results

from collections import namedtuple
from typing import List

import albumentations as A
import pytest
import torch
from albumentations.pytorch import ToTensorV2


@pytest.skip(allow_module_level=True)
def target_transform_func(annotations: List[dict]):
    CaltachPedestrianClass = namedtuple("CaltachPedestrianClass", ["name", "id", "hasInstances", "ignoreInEval", "color"],)
    classes = [
        CaltachPedestrianClass("ignore", 0, False, True, (0, 0, 255)),
        CaltachPedestrianClass("person", 1, True, False, (255, 0, 0)),
    ]

    name2class = {c.name: c for c in classes}

    labels, fixed_boxes, visible_box = [], [], []

    for annotation in annotations:
        obj_type = annotation["label"]
        labels.append(int(name2class[obj_type].hasInstances))
        box = annotation["bbox"]
        fixed_boxes.append([box[0], box[1], box[0] + box[2], box[1] + box[3]])
        vis_box = annotation["bboxVis"]
        visible_box.append([vis_box[0], vis_box[1], vis_box[0] + vis_box[2], vis_box[1] + vis_box[3]])
    fixed_boxes = torch.as_tensor(fixed_boxes, dtype=torch.float32)
    visible_box = torch.as_tensor(visible_box, dtype=torch.float32)
    labels = torch.as_tensor(labels)

    target = {"boxes": fixed_boxes, "vis_boxes": visible_box, "labels": labels}
    # target = {"boxes": torch.as_tensor(fixed_boxes, dtype=torch.float32), "labels": torch.as_tensor(labels)}
    return target


@pytest.skip()
def transform_albumentations():
    albumentations_transform = A.Compose([A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],), ToTensorV2(),])
    return albumentations_transform


def wrap_transform(img, trans):
    return trans(image=img)["image"]


def collate_fn(batch):
    return tuple(zip(*batch))


data_path = "/scratch/ki-wissen-base/data/"
validation_batch_size = 32
world_size = 1
col_fn = collate_fn
num_classes = 2

device = torch.device("cuda:0")

# Get Transform from Albumentation
_transform = partial(wrap_transform, trans=transform_albumentations())

# Get test dataset for evaluation
test_dataset = CaltechPedestrian(
    data_path, transform=_transform, download=True, train=False, subset="annotated-pedestrians", target_transform=target_transform_func,
)

# Create Sampler for dataloader
DistributedSampler = torch.utils.data.distributed.DistributedSampler
test_sampler = DistributedSampler(test_dataset, world_size, rank=0)

# Get dataloader for testing
test_loader = DataLoader(
    test_dataset,
    batch_size=validation_batch_size,
    num_workers=4,
    sampler=test_sampler,
    shuffle=test_sampler is None,
    collate_fn=col_fn,
    pin_memory=True,
)
# for batch_idx, data in enumerate(test_loader):
#     print(batch_idx)

# model = torch.load("/scratch/ki-wissen-base/results/saved/faster_rcnn_caltech_IoU0.5/0712_100533/best_model.pth")
# model.eval()

# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# in_features = model.roi_heads.box_predictor.cls_score.in_features
# model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

model.load_state_dict(
    torch.load("/scratch/ki-wissen-base/results/saved/faster_rcnn_caltech_IoU0.5/0712_100533/best_model.pth", map_location=device,)
)
params = model_state["params"]
model.eval()

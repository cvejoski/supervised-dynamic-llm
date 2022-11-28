# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring
import os

import pytest

from kiwissenbase import data_path
from kiwissenbase.data.datasets import CaltechPedestrian, CityPersons, EuroCityPersons

citypersons_root = data_path / "Cityscapes"
eurocitypersons_root = data_path / "ECP"
caltech_root = data_path / "CaltechPedestrian"


@pytest.mark.skipif(not os.path.exists(caltech_root), reason="The dataset is not downloaded")
class TestCaltech:
    def test_initialize_caltechpedestrian(self):
        train = CaltechPedestrian(data_path, download=True, train=True)
        test = CaltechPedestrian(data_path, download=True, train=False)

        assert len(train) == 4_250
        assert len(test) == 4_024

    def test_initialize_caltechpedestrian_x10(self):
        train = CaltechPedestrian(data_path, download=True, train=True, x10=True)
        test = CaltechPedestrian(data_path, download=True, train=False, x10=True)

        assert len(train) == 42_050
        assert len(test) == 40_240

    @pytest.mark.skipif(not os.path.exists(data_path), reason="The dataset is not downloaded")
    def test_load_annotation_caltech(self):
        train = CaltechPedestrian(data_path, download=False, train=True)
        annotations = train.load_annotation(0, 1, 479)
        assert len(annotations) == 2

    @pytest.mark.skipif(not os.path.exists(data_path), reason="The dataset is not downloaded")
    def test_load_img_caltech(self):
        train = CaltechPedestrian(data_path, download=False, train=True)
        img = train.load_image(0, 0, 419, True)
        # cv2.imshow("Image", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        assert img is not None

    @pytest.mark.skipif(not os.path.exists(data_path), reason="The dataset is not downloaded")
    def test_get_caltech(self):
        train = CaltechPedestrian(data_path, download=False, train=True)
        x, target = train[3]
        assert x is not None
        assert len(target["boxes"]) == 2

    @pytest.mark.skipif(not os.path.exists(data_path), reason="The dataset is not downloaded")
    def test_get_caltech_grouped(self):
        train = CaltechPedestrian(data_path, download=False, train=True, group_pedestrian_classes=True)
        x, target = train[3]
        assert x is not None
        assert len(target["labels"]) == 2
        assert target["labels"][0] == 0

    @pytest.mark.skipif(not os.path.exists(data_path), reason="The dataset is not downloaded")
    def test_get_all_images_with_pedestrians_caltech(self):
        train = CaltechPedestrian(data_path, download=False, train=True, subset="annotated-pedestrians")

        assert len(train) == 1_609

    @pytest.mark.skipif(not os.path.exists(data_path), reason="The dataset is not downloaded")
    def test_crop_pedestrian_caltech(self):
        train = CaltechPedestrian(data_path, download=False, train=True)
        img = train.load_image(1, 3, 119, False)
        anno = train.load_annotation(1, 3, 119)
        crops = train.crop_pedestrian(img, anno)

        assert len(crops) == 12


@pytest.mark.skipif(not os.path.exists(citypersons_root), reason="The dataset is not downloaded")
class TestCityPersons:
    def test_initialize_citypersons(self):
        train = CityPersons(citypersons_root, split="train")
        val = CityPersons(citypersons_root, split="val")
        test = CityPersons(citypersons_root, split="test")

        assert len(train) == 2_975
        assert len(val) == 500
        assert len(test) == 1_525

    @pytest.mark.skipif(not os.path.exists(citypersons_root), reason="The dataset is not downloaded")
    def test_load_annotation_file_citypersons(self):
        train = CityPersons(citypersons_root, split="train")
        annotations = train.load_annotation("aachen", "000166_000019")

        assert len(annotations) == 4

    @pytest.mark.skipif(not os.path.exists(citypersons_root), reason="The dataset is not downloaded")
    def test_load_img_citypersons(self):
        train = CityPersons(citypersons_root, split="train")
        img = train.load_image("aachen", "000146_000019", True)
        # cv2.imshow("Image", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        assert img is not None

    @pytest.mark.skipif(not os.path.exists(citypersons_root), reason="The dataset is not downloaded")
    def test_get_citypersons(self):
        train = CityPersons(citypersons_root, split="train")
        img, target = train[45]

        assert img is not None
        assert len(target["boxes"]) == 15
        assert len(target["labels"]) == 15
        assert len(target["boxesVisRatio"]) == 15

    @pytest.mark.skipif(not os.path.exists(citypersons_root), reason="The dataset is not downloaded")
    def test_get_citypersons_group_class(self):
        train = CityPersons(citypersons_root, split="train", group_pedestrian_classes=True)
        img, target = train[45]

        assert img is not None
        assert len(target["boxes"]) == 15
        assert target["labels"][0] == 1
        assert target["labels"][14] == 0
        assert target["labels"][11] == 1

    @pytest.mark.skipif(not os.path.exists(citypersons_root), reason="The dataset is not downloaded")
    def test_get_citypersons_not_group_class(self):
        train = CityPersons(citypersons_root, split="train", group_pedestrian_classes=False)
        img, target = train[45]

        assert img is not None
        assert len(target["boxes"]) == 15
        assert target["labels"][0] == 1
        assert target["labels"][14] == 0
        assert target["labels"][11] == 3

    @pytest.mark.skipif(not os.path.exists(citypersons_root), reason="The dataset is not downloaded")
    def test_crop_pedestrian_citypersons(self):
        train = CityPersons(citypersons_root, split="train")
        img = train.load_image("aachen", "000146_000019", False)
        anno = train.load_annotation("aachen", "000146_000019")
        crops = train.crop_pedestrian(img, anno)

        # for ix, c in enumerate(crops):
        #     cv2.imshow(f'Figure {ix}', c)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        assert len(crops) == 1

    @pytest.mark.skipif(not os.path.exists(citypersons_root), reason="The dataset is not downloaded")
    def test_get_all_images_with_pedestrians_citypersons(self):
        val = CityPersons(citypersons_root, split="val")
        images = val.get_all_images_with_pedestrians()
        assert len(images) == 3


@pytest.mark.skipif(not os.path.exists(eurocitypersons_root), reason="The dataset is not downloaded")
class TestEuroCityPersons:
    def test_initialize_eurocitypersons(self):
        train = EuroCityPersons(eurocitypersons_root, time="day", split="train")
        val = EuroCityPersons(eurocitypersons_root, time="day", split="val")
        test = EuroCityPersons(eurocitypersons_root, time="day", split="test")

        assert len(train) == 23_892
        assert len(val) == 4_266
        assert len(test) == 12_059

    @pytest.mark.skipif(not os.path.exists(eurocitypersons_root), reason="The dataset is not downloaded")
    def test_load_annotation_eurocitypersons(self):
        val = EuroCityPersons(eurocitypersons_root, split="val")
        annotations = val.load_annotation("basel", 630)

        assert len(annotations) == 16

    @pytest.mark.skipif(not os.path.exists(eurocitypersons_root), reason="The dataset is not downloaded")
    def test_load_img_eurocitypersons(self):
        val = EuroCityPersons(eurocitypersons_root, split="val")
        img = val.load_image("basel", 630, True)
        # cv2.imshow("Image", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        assert img is not None

    @pytest.mark.skipif(not os.path.exists(eurocitypersons_root), reason="The dataset is not downloaded")
    def test_get_eurocitypersons(self):
        val = EuroCityPersons(eurocitypersons_root, split="val")
        img, target = val[100]

        assert img is not None

        assert len(target["boxes"]) == 14
        assert len(target["labels"]) == 14
        assert len(target["boxesVisRatio"]) == 14
        assert target["boxesVisRatio"][12] == 0.1
        assert target["boxesVisRatio"][13] == 1
        assert target["boxesVisRatio"][3] == 0.4
        assert target["labels"][0] == 5
        assert target["labels"][2] == 1
        assert target["labels"][4] == 2

    @pytest.mark.skipif(not os.path.exists(eurocitypersons_root), reason="The dataset is not downloaded")
    def test_get_eurocitypersons_group_class(self):
        val = EuroCityPersons(eurocitypersons_root, split="val", group_pedestrian_classes=True)
        img, target = val[100]

        assert img is not None
        assert target["labels"][0] == 0
        assert target["labels"][2] == 1
        assert target["labels"][4] == 1

    @pytest.mark.skipif(not os.path.exists(eurocitypersons_root), reason="The dataset is not downloaded")
    def test_crop_pedestrian_eurocitypersons(self):
        val = EuroCityPersons(eurocitypersons_root, split="val")
        img = val.load_image("basel", 630, False)
        anno = val.load_annotation("basel", 630)
        crops = val.crop_pedestrian(img, anno)

        # for ix, c in enumerate(crops):
        #     cv2.imshow(f'Figure {ix}', c)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        assert len(crops) == 12

    @pytest.mark.skipif(not os.path.exists(data_path), reason="The dataset is not downloaded")
    def test_get_all_images_with_pedestrians_eurocitypersons(self):
        val = EuroCityPersons(eurocitypersons_root, split="val")
        images = val.get_all_images_with_pedestrians()
        assert len(images) == 31
        total = sum([len(city) for city in images.values()])

        assert total == 24189

import pytest

from kiwissenbase.utils.helper import get_static_method


@pytest.mark.skip
def test_get_static_method():
    module = "kiwissenbase.models.object_detection.faster_rcnn"
    class_name = "FasterRCNN"
    method_name = "target_transform"
    method = get_static_method(module, class_name, method_name)
    assert method is not None

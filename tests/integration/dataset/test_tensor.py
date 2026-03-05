# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import sys

import pytest

import muller
from muller.util.exceptions import InvalidTensorNameError
from tests.constants import TEST_TENSOR_PATH, SAMPLE_FILES
from tests.utils import official_path, official_creds


@pytest.mark.skipif(("--storage" in sys.argv and "local" not in sys.argv) or
                    (sys.argv[-1].startswith("--storage=") and sys.argv[-1] != "--storage=local"),
                    reason="It should be skipped if not in local")
def test_get_tensors(storage):
    ds = muller.dataset(path=official_path(storage, TEST_TENSOR_PATH), creds=official_creds(storage), overwrite=True)
    ds.create_tensor(name="images", htype="image", sample_compression="jpg")
    ds.images.extend([muller.read(SAMPLE_FILES["jpg_1"])])
    ds.create_tensor(name="labels", htype="class_label")
    ds.labels.extend(["There are 4 dogs in this picture."])

    tensors = ds.tensors
    assert len(tensors) == 2
    assert list(tensors.values())[0].htype == "image"
    assert list(tensors.values())[0].dtype == "uint8"
    assert ds.images[0][0][0][0].numpy() == 243
    assert ds.images[0].shape == (640, 640, 3)

    assert list(tensors.values())[1].htype == "class_label"
    assert list(tensors.values())[1].dtype == "uint32"
    assert len(ds.labels) == 1


def test_invalid_tensor_name(storage):
    ds = muller.dataset(path=official_path(storage, TEST_TENSOR_PATH), creds=official_creds(storage), overwrite=True)
    try:
        # data is an existing attribute of TransformData and must not be redefined or created anew.
        ds.create_tensor(name="data", htype="text")
        assert False, "No exception raises"
    except InvalidTensorNameError as e:
        assert True, f"uid authorization caused exception {e}"

    try:
        # items is a __builtin__ keyword (method of dict and other standard types)
        ds.create_tensor(name="items", htype="text")
        assert False, "No exception raises"
    except InvalidTensorNameError as e:
        assert True, f"uid authorizatioexn caused exception {e}"

    try:
        # None is a Python reserved keyword (a singleton constant representing the absence of a value)
        ds.create_tensor(name="None", htype="text")
        assert False, "No exception raises"
    except InvalidTensorNameError as e:
        assert True, f"uid authorization caused exception {e}"

    try:
        # values is a __builtin__ keyword
        ds.create_tensor(name="values", htype="text")
        assert False, "No exception raises"
    except InvalidTensorNameError as e:
        assert True, f"uid authorization caused exception {e}"


if __name__ == '__main__':
    pytest.main(["-s", "test_tensor.py"])

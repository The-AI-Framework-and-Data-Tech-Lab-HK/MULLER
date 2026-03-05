# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import pytest

import muller

from tests.utils import official_path, official_creds
from tests.constants import TEST_LOAD_PATH, SAMPLE_FILES


def create_dataset(storage):
    muller_ds = muller.dataset(path=official_path(storage, TEST_LOAD_PATH), creds=official_creds(storage), overwrite=True)
    muller_ds.create_tensor(name="images", htype="image", sample_compression="jpg")
    muller_ds.images.extend([muller.read(SAMPLE_FILES["jpg_1"])])
    return muller_ds


def test_load_dataset(storage):
    create_dataset(storage)
    ds = muller.load(path=official_path(storage, TEST_LOAD_PATH), creds=official_creds(storage))
    # ds.summary()
    assert ds.images[0][0][0][0].numpy() == 243
    assert ds.images[0].shape == (640, 640, 3)
    assert ds.images[0].dtype == "uint8"


def create_tensor_and_commit(storage):
    ds = create_dataset(storage)
    first_commit = ds.commit()
    ds.images.extend([muller.read(SAMPLE_FILES["jpg_2"])])
    ds.create_tensor(name="labels", htype="class_label")
    ds.labels.append([1, 2, 3, 4])
    ds.labels.append([5, 6, 7, 8])
    second_commit = ds.commit()
    return first_commit, second_commit


def test_load_dataset_at_commit(storage):
    first_commit, second_commit = create_tensor_and_commit(storage)
    ds = muller.load(f"{official_path(storage, TEST_LOAD_PATH)}@{first_commit}", creds=official_creds(storage))
    ds.summary()
    assert len(ds) == 1
    assert ds.images[0][0][0][0].numpy() == 243
    assert ds.images[0].shape == (640, 640, 3)
    assert ds.images[0].dtype == "uint8"

    # load the second commit dataset
    ds_1 = muller.load(f"{official_path(storage, TEST_LOAD_PATH)}@{second_commit}", creds=official_creds(storage))
    ds_1.summary()
    assert len(ds_1) == 2
    assert ds_1.images[0].shape == (640, 640, 3)
    assert ds_1.images[1].shape == (425, 640, 3)
    assert ds_1.images[1][0][0][0].numpy() == 129
    assert ds_1.images[1].dtype == "uint8"


if __name__ == '__main__':
    pytest.main(["-s", "test_load.py"])

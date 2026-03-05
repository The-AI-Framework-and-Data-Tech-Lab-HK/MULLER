# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import os
import pytest

import muller
from muller.util.exceptions import DatasetNotExistsError
from tests.utils import official_path, official_creds
from tests.constants import TEST_DELETE_PATH, SAMPLE_FILES, TEST_RENAME_PATH


def create_dataset(storage):
    muller_ds = muller.dataset(path=official_path(storage, TEST_DELETE_PATH), creds=official_creds(storage), overwrite=True)
    muller_ds.create_tensor(name="images", htype="image", sample_compression="jpg")
    muller_ds.images.extend([muller.read(SAMPLE_FILES["jpg_1"])])
    return muller_ds


def test_delete_dataset(storage):
    ds = create_dataset(storage)
    ds.summary()
    muller.delete(path=official_path(storage, TEST_DELETE_PATH), creds=official_creds(storage))

    assert not os.path.exists(official_path(storage, TEST_DELETE_PATH))
    with pytest.raises(DatasetNotExistsError) as e:
        muller.load(path=official_path(storage, TEST_DELETE_PATH))
    assert e.type == DatasetNotExistsError


def test_delete_branch(storage):
    ds = create_dataset(storage)
    ds.checkout("test_branch", create=True)
    ds.checkout("dev_branch", create=True)
    ds.delete_branch("test_branch")
    assert len(ds.branches) == 2
    assert ds.branch == "dev_branch"
    assert "test_branch" not in ds.branches


def test_rename_dataset(storage):
    if storage != "local":
        return  # We only support rename operation in LocalProvider.
    ds = create_dataset(storage)
    ds.rename(official_path(storage, TEST_RENAME_PATH))
    muller.load(official_path(storage, TEST_RENAME_PATH), creds=official_creds(storage))
    assert ds.images[0][0][0][0].numpy() == 243
    assert ds.images[0].shape == (640, 640, 3)
    assert ds.images[0].dtype == "uint8"

    with pytest.raises(DatasetNotExistsError) as e:
        muller.load(path=official_path(storage, TEST_DELETE_PATH), creds=official_creds(storage))
    assert e.type == DatasetNotExistsError

    # Finally we delete the TEST_RENAME_PATH in case we met PathNotEmptyException in the next-round testing.
    muller.delete(path=official_path(storage, TEST_RENAME_PATH), creds=official_creds(storage))


if __name__ == '__main__':
    pytest.main(["-s", "test_delete.py"])

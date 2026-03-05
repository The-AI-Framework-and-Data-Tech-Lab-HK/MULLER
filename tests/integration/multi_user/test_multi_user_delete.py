# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import pytest

import muller
from muller.util.exceptions import UnAuthorizationError
from muller.util.sensitive_config import SensitiveConfig
from tests.constants import TEST_MULTI_USER_PATH
from tests.utils import official_path, official_creds


def test_delete_tensor(storage):
    """ create multi branches by multi users."""
    SensitiveConfig().uid = "A"
    ds = muller.dataset(path=official_path(storage, TEST_MULTI_USER_PATH),
                       creds=official_creds(storage), overwrite=True)
    ds.create_tensor('labels', htype='generic', dtype='int')
    ds.create_tensor('labels_2', htype='generic', dtype='float')
    ds.create_tensor('categories', htype='text')
    ds.labels.extend([10, 2, 30, 4, 50, 6, 70, 8, 90, 100] * 2)
    ds.labels_2.extend([10, 2, 30, 4, 50, 6, 70, 8, 90, 100] * 2)
    ds.categories.extend(['agent', '情感', '生成', '写作', '情感', 'agent', '生成', '写作', '情感', '写作'] * 2)
    ds.commit()

    # Change to user B, checkout a new branch dev, and delete a column
    SensitiveConfig().uid = "B"
    ds.checkout("dev", create=True)
    ds.delete_tensor('labels')
    assert len(ds.tensors) == 2

    # Checkout to user A, and delete columns in branch dev
    SensitiveConfig().uid = "A"
    ds.checkout("dev")
    ds.delete_tensor('labels_2')
    assert len(ds.tensors) == 1

    # Checkout to user B, go back to the main branch. User B cannot delete columns in the main branch.
    SensitiveConfig().uid = "B"
    ds.checkout("main")
    try:
        ds.delete_tensor('labels')  # Raise exception. But it runs smoothly in the test case.
        assert False, "No exception raises"
    except UnAuthorizationError as e:
        assert True, f"exception: {e}"


def test_delete_dataset(storage):
    """ test delete dataset of multiple users."""

    SensitiveConfig().uid = "A"
    ds = muller.dataset(path=official_path(storage, TEST_MULTI_USER_PATH),
                       creds=official_creds(storage), overwrite=True)
    ds.create_tensor('labels', htype='generic', dtype='int')
    ds.create_tensor('categories', htype='text')
    ds.labels.extend([10, 2, 30, 4, 50, 6, 70, 8, 90, 100] * 2)
    ds.categories.extend(['agent', '情感', '生成', '写作', '情感', 'agent', '生成', '写作', '情感', '写作'] * 2)
    ds.commit()

    SensitiveConfig().uid = "B"
    ds.checkout("dev", create=True)
    try:
        ds.delete()
        assert False, "No exception raises"
    except UnAuthorizationError as e:
        assert True, f"exception: {e}"
    assert len(ds) == 20

    SensitiveConfig().uid = "A"
    ds.delete()


if __name__ == '__main__':
    pytest.main(["-s", "test_multi_user_delete.py"])

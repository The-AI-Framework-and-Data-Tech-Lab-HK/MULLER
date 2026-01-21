# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import numpy as np
import pytest

import muller
from muller.util.authorization import obtain_current_user
from muller.util.exceptions import EmptyTensorError, UnAuthorizationError
from muller.util.sensitive_config import SensitiveConfig
from tests.constants import VIEW_TEST_PATH
from tests.utils import official_path, official_creds


def populate(local_path, creds):
    """
    Returns the dataset with given path.
    """
    ds = muller.dataset(local_path, creds=creds, overwrite=True)
    return ds


def test_view_with_empty_tensor(storage):
    """
    Tests dataset load view with empty tensor.
    """
    with populate(official_path(storage, VIEW_TEST_PATH), official_creds(storage)) as ds:
        ds.create_tensor("images")
        ds.images.extend([1, 2, 3, 4, 5])

        ds.create_tensor("labels")
        ds.labels.extend([None, None, None, None, None])
        ds.commit()

        ds[:3].save_view(view_id="save1", optimize=True)

    view = ds.load_view("save1")

    assert len(view) == 3

    with pytest.raises(EmptyTensorError):
        view.labels.numpy()

    np.testing.assert_array_equal(
        view.images.numpy(), np.array([1, 2, 3]).reshape(3, 1)
    )


def test_vds_read_only(storage):
    """
    Tests view read_only property.
    """
    with populate(official_path(storage, VIEW_TEST_PATH), official_creds(storage)) as ds:
        ds.create_tensor("abc")
        ds.abc.extend([1, 2, 3, 4, 5])
        ds.commit()

    ds[:3].save_view(view_id="first_3")

    ds = muller.load(official_path(storage, VIEW_TEST_PATH), creds=official_creds(storage), read_only=True)

    view = ds.load_view("first_3")

    assert view.base_storage.read_only is True


def test_view_from_different_commit(storage):
    """
    Tests save view from different commits.
    """
    with populate(official_path(storage, VIEW_TEST_PATH), official_creds(storage)) as ds:
        ds.create_tensor("x")
        ds.x.extend(list(range(10)))
        cid = ds.commit()
        view = ds[4:9]
        view.save_view(view_id="abcd")
        ds.x.extend(list(range(10, 20)))
        cid2 = ds.commit()
        view2 = ds.load_view("abcd")
        assert view2.commit_id == cid
        assert ds.commit_id == cid2
        assert not view2.is_optimized
        view2.save_view(view_id="efg", optimize=True)
        view3 = ds.load_view("efg")
        assert ds.commit_id == cid2
        assert view3.is_optimized


def test_save_view_with_multi_users(storage):
    """
    Tests save view with multi users.
    """
    # The user public create a dataset
    SensitiveConfig().uid = "public"
    ds = muller.dataset(official_path(storage, VIEW_TEST_PATH), official_creds(storage), overwrite=True)
    with ds:
        ds.create_tensor('labels', htype='generic', dtype='int')
        ds.create_tensor('mul_values', htype='text')
        ds.create_tensor('categories', htype='text')
        ds.labels.extend([10, 2, 30, 4, 50, 6, 70, 8, 90, 100] * 2)
        ds.mul_values.extend(['A000', 'A001', 'A002', 'A003', 'A004', 'A100', 'B000', 'B001', 'C000', 'C100'] * 2)
        ds.categories.extend(['agent', '情感', '生成', '写作', '情感', 'agent', '生成', '写作', '情感', '写作'] * 2)

    # Append data and commit
    ds.labels.extend([100, 150])
    ds.mul_values.extend(['A000', 'A001'])
    ds.categories.extend(['情感', '情感'])
    commit_1 = ds.commit()

    # User A query data and save view
    SensitiveConfig().uid = "A"
    ds = muller.load(official_path(storage, VIEW_TEST_PATH), official_creds(storage))
    v1 = ds.filter_vectorized([("categories", "==", '情感')])
    v1.save_view(view_id="first_11")

    # User A pop data, but there is UnautthorizationError
    try:
        ds.pop([1, 11, 14])
    except UnAuthorizationError:
        pass

    # User A checkout to branch A, pop data, and commit
    ds = muller.load(official_path(storage, VIEW_TEST_PATH), official_creds(storage))
    ds.checkout("branchA", create=True)
    ds.pop([1, 11, 14])
    commit_2 = ds.commit()

    # User B checkout to branch A, query data, and save view
    SensitiveConfig().uid = "B"
    ds = muller.load(official_path(storage, VIEW_TEST_PATH), official_creds(storage))
    ds.checkout("branchA")
    ds_tmp3 = ds.filter_vectorized([("categories", "==", '生成')])
    ds_tmp3.save_view(view_id="second")

    # User B load the view saved by user A
    view_1 = ds.load_view("first_11")
    assert len(view_1) == 8
    assert view_1.commit_id == commit_1

    # User A load the view saved by user B
    SensitiveConfig().uid = "A"
    view_2 = ds.load_view("second")
    assert len(view_2) == 4
    assert view_2.commit_id == commit_2


def test_delete_view_with_multi_users(storage):
    """
    Tests delete view with multi users.
    """
    # The user public create a dataset
    SensitiveConfig().uid = "public"
    ds = muller.dataset(official_path(storage, VIEW_TEST_PATH), official_creds(storage), overwrite=True)
    with ds:
        ds.create_tensor('labels', htype='generic', dtype='int')
        ds.create_tensor('mul_values', htype='text')
        ds.create_tensor('categories', htype='text')
        ds.labels.extend([10, 2, 30, 4, 50, 6, 70, 8, 90, 100] * 2)
        ds.mul_values.extend(['A000', 'A001', 'A002', 'A003', 'A004', 'A100', 'B000', 'B001', 'C000', 'C100'] * 2)
        ds.categories.extend(['agent', '情感', '生成', '写作', '情感', 'agent', '生成', '写作', '情感', '写作'] * 2)

    # Append data and commit
    ds.labels.extend([100, 150])
    ds.mul_values.extend(['A000', 'A001'])
    ds.categories.extend(['情感', '情感'])
    ds.commit()

    # User A query data and save view
    SensitiveConfig().uid = "A"
    ds = muller.load(official_path(storage, VIEW_TEST_PATH), official_creds(storage))
    v1 = ds.filter_vectorized([("categories", "==", '情感')])
    v1.save_view(view_id="first_11")
    assert obtain_current_user() == "A"

    # User B delete the view saved by user A
    SensitiveConfig().uid = "B"
    ds = muller.load(official_path(storage, VIEW_TEST_PATH), official_creds(storage))
    try:
        ds.delete_view("first_11")
        assert False, "No exception raises"
    except UnAuthorizationError as e:
        assert True, f"uid authorization caused exception {e}"

    # User A delete the view saved by herself/himself
    SensitiveConfig().uid = "A"
    ds = muller.load(official_path(storage, VIEW_TEST_PATH), official_creds(storage))
    ds.delete_view("first_11")


if __name__ == '__main__':
    pytest.main(["-s", "test_views.py"])

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


def test_get_views(storage):
    """
    Tests get_views function to retrieve all views.
    """
    SensitiveConfig().uid = "test_user"
    with populate(official_path(storage, VIEW_TEST_PATH), official_creds(storage)) as ds:
        ds.create_tensor("samples")
        ds.samples.extend([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        commit_1 = ds.commit("First commit")
        
        # Save first view
        ds[:3].save_view(view_id="view1", message="First three samples")
        
        # Save second view
        ds[5:8].save_view(view_id="view2", message="Middle samples")
        
        # Get all views
        all_views = ds.get_views()
        assert len(all_views) == 2
        view_ids = [v.id for v in all_views]
        assert "view1" in view_ids
        assert "view2" in view_ids
        
        # Make another commit
        ds.samples.extend([11, 12, 13])
        commit_2 = ds.commit("Second commit")
        
        # Save view from second commit
        ds[8:].save_view(view_id="view3", message="Last samples")
        
        # Get all views
        all_views = ds.get_views()
        assert len(all_views) == 3
        
        # Get views from specific commit
        commit_1_views = ds.get_views(commit_id=commit_1)
        assert len(commit_1_views) == 2
        
        commit_2_views = ds.get_views(commit_id=commit_2)
        assert len(commit_2_views) == 1
        assert commit_2_views[0].id == "view3"


def test_get_view(storage):
    """
    Tests get_view function to retrieve a specific view by id.
    """
    SensitiveConfig().uid = "test_user"
    with populate(official_path(storage, VIEW_TEST_PATH), official_creds(storage)) as ds:
        ds.create_tensor("scores")
        ds.scores.extend([10, 20, 30, 40, 50])
        ds.commit()
        
        # Save a view
        ds[:3].save_view(view_id="test_view", message="Test view")
        
        # Get the view by id
        view_entry = ds.get_view("test_view")
        assert view_entry.id == "test_view"
        assert view_entry.message == "Test view"
        
        # Load the view and verify
        loaded_view = view_entry.load()
        assert len(loaded_view) == 3
        np.testing.assert_array_equal(
            loaded_view.scores.numpy(), 
            np.array([10, 20, 30]).reshape(3, 1)
        )
        
        # Test KeyError for non-existent view
        with pytest.raises(KeyError):
            ds.get_view("non_existent_view")


def test_save_view_to_external_path(storage):
    """
    Tests save_view with external path parameter and optimize=True.
    """
    SensitiveConfig().uid = "test_user"
    external_path = official_path(storage, VIEW_TEST_PATH + "_external")
    
    with populate(official_path(storage, VIEW_TEST_PATH), official_creds(storage)) as ds:
        ds.create_tensor("numbers")
        ds.numbers.extend([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ds.commit()
        
        # Save view to external path with optimize=True to copy data
        view_path = ds[:5].save_view(
            path=external_path,
            view_id="external_view",
            message="External view test",
            optimize=True
        )
        
        assert view_path == external_path
        
        # Load the external VDS directly
        external_ds = muller.load(external_path, creds=official_creds(storage))
        assert len(external_ds) == 5
        np.testing.assert_array_equal(
            external_ds.numbers.numpy(),
            np.array([1, 2, 3, 4, 5]).reshape(5, 1)
        )
        
        # Verify external view is not in parent dataset's get_views
        all_views = ds.get_views()
        view_ids = [v.id for v in all_views]
        assert "external_view" not in view_ids


def test_save_view_to_external_path_vds(storage):
    """
    Tests save_view to external path as VDS (virtual dataset, optimize=False).
    """
    SensitiveConfig().uid = "test_user"
    external_path = official_path(storage, VIEW_TEST_PATH + "_external_vds")
    
    with populate(official_path(storage, VIEW_TEST_PATH), official_creds(storage)) as ds:
        ds.create_tensor("numbers")
        ds.numbers.extend([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ds.commit()
        
        # Save view to external path without optimization (creates VDS)
        view_path = ds[:5].save_view(
            path=external_path,
            view_id="external_vds",
            message="External VDS test"
        )
        
        assert view_path == external_path
        
        # Load the external VDS
        external_vds = muller.load(external_path, creds=official_creds(storage))
        assert len(external_vds) == 5
        
        # Access data through get_view_for_vds (since it's a VDS, not optimized)
        view_from_vds = external_vds.get_view_for_vds()
        assert len(view_from_vds) == 5
        np.testing.assert_array_equal(
            view_from_vds.numbers.numpy(),
            np.array([1, 2, 3, 4, 5]).reshape(5, 1)
        )


def test_get_view_for_vds(storage):
    """
    Tests get_view_for_vds function to get view from VDS.
    """
    SensitiveConfig().uid = "test_user"
    with populate(official_path(storage, VIEW_TEST_PATH), official_creds(storage)) as ds:
        ds.create_tensor("records")
        ds.records.extend([100, 200, 300, 400, 500])
        commit_id = ds.commit()
        
        # Save a view
        ds[1:4].save_view(view_id="vds_test")
        
        # Get the ViewEntry to access VDS path
        view_entry = ds.get_view("vds_test")
        vds_path = f".queries/{view_entry.id}"
        
        # Load the VDS dataset object directly (not through load_view)
        vds = ds.sub_ds(vds_path, verbose=False, read_only=True)
        
        # Now call get_view_for_vds on the VDS object
        view_from_vds = vds.get_view_for_vds()
        
        # Verify the view
        assert len(view_from_vds) == 3
        np.testing.assert_array_equal(
            view_from_vds.records.numpy(),
            np.array([200, 300, 400]).reshape(3, 1)
        )
        assert view_from_vds.commit_id == commit_id


def test_save_view_with_optimize(storage):
    """
    Tests save_view with optimize=True parameter.
    """
    SensitiveConfig().uid = "test_user"
    with populate(official_path(storage, VIEW_TEST_PATH), official_creds(storage)) as ds:
        ds.create_tensor("entries")
        ds.entries.extend(list(range(20)))
        ds.commit()
        
        # Save optimized view
        ds[5:15].save_view(view_id="optimized_view", optimize=True)
        
        # Load the optimized view
        optimized = ds.load_view("optimized_view")
        
        # Verify it's optimized
        assert optimized.is_optimized
        assert len(optimized) == 10
        np.testing.assert_array_equal(
            optimized.entries.numpy(),
            np.array(list(range(5, 15))).reshape(10, 1)
        )


if __name__ == '__main__':
    pytest.main(["-s", "test_views.py"])

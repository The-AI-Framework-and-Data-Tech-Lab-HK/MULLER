# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import logging

import pytest

import muller
from muller.util.exceptions import UnAuthorizationError, UnsupportedMethod
from muller.util.sensitive_config import SensitiveConfig
from tests.constants import TEST_MULTI_USER_PATH
from tests.utils import official_path, official_creds

logging.basicConfig(level=logging.INFO)


def test_multi_branches(storage):
    """ create multi branches by multi users."""
    SensitiveConfig().uid = "A"
    ds = muller.dataset(path=official_path(storage, TEST_MULTI_USER_PATH),
                       creds=official_creds(storage), overwrite=True)
    ds.create_tensor(name="test", htype="text")
    ds.test.append("hi")
    ds.commit()

    logging.info(f'cur user is {SensitiveConfig().uid}, cur branch is {ds.branch}')
    assert len(ds) == 1

    SensitiveConfig().uid = "B"
    ds.checkout("dev_B", create=True)
    ds.test.append("bye")
    ds.commit()

    logging.info(f'cur user is {SensitiveConfig().uid}, cur branch is {ds.branch}')
    assert len(ds) == 2


def test_multi_branches_2(storage):
    """ create multi branches by muti users."""
    ds = muller.dataset(path=official_path(storage, TEST_MULTI_USER_PATH),
                       creds=official_creds(storage), overwrite=True)
    ds.checkout("dev", create=True)
    ds.create_tensor("arr")
    ds.arr.extend([1])
    assert ds.has_head_changes
    ds.commit()
    assert not ds.has_head_changes
    assert len(ds) == 1

    SensitiveConfig().uid = "B"
    ds.checkout("B_branch", create=True)
    ds.arr.extend([2])
    assert ds.has_head_changes
    assert len(ds) == 2

    SensitiveConfig().uid = "C"
    ds.checkout("C_branch", create=True)
    assert len(ds) == 2
    assert not ds.has_head_changes


def test_multi_user_modify_data(storage):
    """ different users modify data on different branches.."""
    # 1.1 User A create the dataset and append data
    SensitiveConfig().uid = "A"
    ds = muller.dataset(path=official_path(storage, TEST_MULTI_USER_PATH),
                       creds=official_creds(storage), overwrite=True)

    ds.create_tensor(name="test", htype="text")
    count = 0
    with ds:
        while count < 20:
            ds.test.append("hi")
            ds.test.append("bye")
            count += 1
    ds.commit()

    # 1.2 User B create a branch and append data
    SensitiveConfig().uid = "B"
    ds.checkout("dev_B", create=True)
    ds.test.append("bye")
    ds.commit()

    # 2.1 User B try to modify the main branch
    ds.checkout("main")

    try:
        ds.pop([1, 8])
        assert False, "No exception raises"
    except UnAuthorizationError as e:
        assert True, f"uid authorization caused exception {e}"

    ds = muller.load(path=official_path(storage, TEST_MULTI_USER_PATH))
    try:
        ds[0].update({"test": "hello"})
        assert False, "No exception raises"
    except UnAuthorizationError as e:
        assert True, f"uid authorization caused exception {e}"

    ds = muller.load(path=official_path(storage, TEST_MULTI_USER_PATH))
    try:
        ds.test[0] = "heyhey"
        assert False, "No exception raises"
    except UnAuthorizationError as e:
        assert True, f"uid authorization caused exception {e}"

    ds = muller.load(path=official_path(storage, TEST_MULTI_USER_PATH))
    try:
        ds.test.append("hi")
        assert False, "No exception raises"
    except UnAuthorizationError as e:
        assert True, f"uid authorization caused exception {e}"


def test_filter_and_merge_data(storage):
    """ different users modify data on different branches.."""
    # 1.1 User A create the dataset and append data
    SensitiveConfig().uid = "A"
    ds = muller.dataset(path=official_path(storage, TEST_MULTI_USER_PATH),
                       creds=official_creds(storage), overwrite=True)

    ds.create_tensor(name="test", htype="text")
    count = 0
    with ds:
        while count < 20:
            ds.test.append("hi")
            ds.test.append("bye")
            count += 1
    ds.commit()

    # 1.2 User B checkout a new branch and append data
    SensitiveConfig().uid = "B"
    ds.checkout("dev_B", create=True)
    ds.test.append("hello")
    ds.test.append("world")
    ds.commit()

    # 1.2 User C checkout a new branch and append data
    SensitiveConfig().uid = "C"
    ds.checkout("dev_C", create=True)
    ds.test.append("good")
    ds.test.append("morning")
    ds.commit()

    # 2.1 User C conduct query and merge on Branch dev_B
    ds.checkout("dev_B")
    assert len(ds) == 42

    # filter vectorized with index
    ds_1 = ds.filter_vectorized([("test", "LIKE", "ye", True)])
    assert len(ds_1) == 20

    # merge
    try:
        ds.merge("dev_C")
        assert False, "No exception raises"
    except UnAuthorizationError as e:
        assert True, f"uid authorization caused exception {e}"

    # 2.2 User A conduct merge on the main branch
    SensitiveConfig().uid = "A"
    ds = muller.load(path=official_path(storage, TEST_MULTI_USER_PATH))

    ds.merge("dev_B")
    assert len(ds) == 42


def test_checkout_authentication(storage):
    """Function to test checkout authentication."""
    SensitiveConfig().uid = "A"
    ds = muller.dataset(path=official_path(storage, TEST_MULTI_USER_PATH),
                       creds=official_creds(storage), overwrite=True)
    ds.create_tensor('labels', htype='generic', dtype='int')
    ds.labels.extend([10, 2, 30, 4, 50, 6, 70, 8, 90, 100] * 2)
    first_commit_id = ds.commit()

    SensitiveConfig().uid = "B"
    ds.checkout("dev-B", create=True)
    
    # Check that the main branch's first commit still belongs to User A
    # Use the actual first commit ID after User A's commit
    try:
        owner_name = ds.version_state['commit_node_map'][first_commit_id].commit_user_name
    except KeyError:
        assert False, f"Commit {first_commit_id} not found in commit_node_map"
    assert owner_name == 'A', f"Expected owner 'A' but got '{owner_name}'"

    ds.checkout("main", create=False)
    try:
        ds.labels.extend([10, 2, 30, 4, 50, 6, 70, 8, 90, 100] * 2)
        assert False, "No exception raises"
    except UnAuthorizationError as e:
        assert True, f"uid authorization caused exception {e}"


def test_limited_authentication_on_other_branches(storage):
    """Function to test limited_authentication_on_other_branches."""
    SensitiveConfig().uid = "A"
    ds = muller.dataset(path=official_path(storage, TEST_MULTI_USER_PATH),
                       creds=official_creds(storage), overwrite=True)
    ds.create_tensor('labels', htype='generic', dtype='int')
    ds.labels.extend([10, 2, 30, 4, 50, 6, 70, 8, 90, 100] * 2)

    SensitiveConfig().uid = "B"
    ds.checkout("dev-B", create=True)
    ds.labels.extend([10, 2, 30, 4, 50, 6, 70, 8, 90, 100] * 2)

    SensitiveConfig().uid = "C"
    try:
        ds.reset()
        assert False, "No exception raises"
    except UnAuthorizationError as e:
        assert True, f"uid authorization caused exception {e}"

    ds.checkout("dev-C", create=True)
    ds.labels.extend([10, 2, 30, 4, 50, 6, 70, 8, 90, 100] * 2)
    ds.checkout("dev-B")

    try:
        ds.create_tensor('labels2', htype='generic', dtype='int')
        assert False, "No exception raises"
    except UnAuthorizationError as e:
        assert True, f"uid authorization caused exception {e}"

    try:
        ds.create_tensor_like('labels_copy', ds["labels"])
        assert False, "No exception raises"
    except UnAuthorizationError as e:
        assert True, f"uid authorization caused exception {e}"

    try:
        ds.commit()
        assert False, "No exception raises"
    except UnAuthorizationError as e:
        assert True, f"uid authorization caused exception {e}"

    try:
        ds.extend(ds[:5])
        assert False, "No exception raises"
    except UnAuthorizationError as e:
        assert True, f"uid authorization caused exception {e}"

    try:
        ds.merge("dev-C")
        assert False, "No exception raises"
    except UnAuthorizationError as e:
        assert True, f"uid authorization caused exception {e}"

    try:
        ds.delete()
        assert False, "No exception raises"
    except UnAuthorizationError as e:
        assert True, f"uid authorization caused exception {e}"

    SensitiveConfig().uid = "B"
    try:
        ds.delete_branch("dev-C")
        assert False, "No exception raises"
    except UnAuthorizationError as e:
        assert True, f"uid authorization caused exception {e}"

    SensitiveConfig().uid = "C"
    try:
        ds.rechunk()
        assert False, "No exception raises"
    except UnAuthorizationError as e:
        assert True, f"uid authorization caused exception {e}"

    try:
        ds.append(ds[5])
        assert False, "No exception raises"
    except UnAuthorizationError as e:
        assert True, f"uid authorization caused exception {e}"

    try:
        ds[0].update({"labels": 3})
        assert False, "No exception raises"
    except UnAuthorizationError as e:
        assert True, f"uid authorization caused exception {e}"

    try:
        ds.delete()
        assert False, "No exception raises"
    except UnAuthorizationError as e:
        assert True, f"uid authorization caused exception {e}"

    try:
        ds.delete_tensor("labels")
        assert False, "No exception raises"
    except UnAuthorizationError as e:
        assert True, f"uid authorization caused exception {e}"

    try:
        ds.pop([0, 1])
        assert False, "No exception raises"
    except UnAuthorizationError as e:
        assert True, f"uid authorization caused exception {e}"

    try:
        ds.rename("a funny test case")
        assert False, "No exception raises"
    except UnAuthorizationError as e:
        assert True, f"uid authorization caused exception {e}"

    try:
        ds.add_data_from_file()  # Check if raising the error through the decorator is sufficient.
        assert False, "No exception raises"
    except UnAuthorizationError as e:
        assert True, f"uid authorization caused exception {e}"

    try:
        ds.add_data_from_dataframes()  # Check if raising the error through the decorator is sufficient.
        assert False, "No exception raises"
    except UnAuthorizationError as e:
        assert True, f"uid authorization caused exception {e}"

    try:
        ds.labels.clear()
        assert False, "No exception raises"
    except UnAuthorizationError as e:
        assert True, f"uid authorization caused exception {e}"

    try:
        ds.labels.extend([1000, 2000])
        assert False, "No exception raises"
    except UnAuthorizationError as e:
        assert True, f"uid authorization caused exception {e}"

    try:
        ds.labels.append(3000)
        assert False, "No exception raises"
    except UnAuthorizationError as e:
        assert True, f"uid authorization caused exception {e}"

    try:
        ds.labels.pop([0, 1, 2])
        assert False, "No exception raises"
    except UnsupportedMethod as e:
        assert True, f"do not support directly pop from a single tensor column {e}"

    try:
        ds.labels[0] = 1
        assert False, "No exception raises"
    except UnAuthorizationError as e:
        assert True, f"uid authorization caused exception {e}"

    try:
        ds.create_index(['labels'], use_uuid=True)
    except UnAuthorizationError as e:
        assert True, f"uid authorization caused exception {e}"

if __name__ == '__main__':
    pytest.main(["-s", "test_multi_user.py"])

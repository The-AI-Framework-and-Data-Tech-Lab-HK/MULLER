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
from muller.util.exceptions import DatasetCorruptError, MergeConflictError
from tests.constants import TEST_DETECT_MERGE, TEST_DETECT_ERROR
from tests.utils import official_path, official_creds


def test_detect_merge(storage):
    """ test the return values of detect merge api."""
    ds = muller.dataset(path=official_path(storage, TEST_DETECT_MERGE),
                       creds=official_creds(storage), overwrite=True)

    ds.create_tensor(name="labels", htype="generic", dtype="int")
    ds.labels.extend([1, 2, 3, 4, 5])

    ds.commit("first on main.")

    ds.checkout("dev", create=True)

    ds.pop(0) # pop 0 on dev fir
    ds.pop(0) # pop 1 on dev fir
    ds[0].labels = 188 # update 2 on dev first
    ds.labels.extend([6, 7, 8, 9])
    fir_dev = ds.commit("first on dev.") # first_dev

    ds.checkout("main", create=False)
    ds.pop(0) # pop 0 on main sec
    ds.pop(0) # pop 1 on main sec
    ds.pop(0) # pop 2 on main sec
    ds.pop(0) # pop 3 on main sec
    ds[0].labels = 18  # update 4 on main sec
    ds.labels.extend([6, 7, 8, 9, 10, 11])
    ds.commit("sec on main") # sec_main

    conflict_tensors, conflict_records = ds.detect_merge_conflict(target_id=fir_dev, show_value=True)
    assert len(conflict_tensors) == 0
    assert len(conflict_records) == 1
    assert conflict_records['labels']['del_ori_idx'] == [2, 3]
    assert conflict_records['labels']['del_ori_values'] == [[3], [4]]
    assert conflict_records['labels']['del_tar_idx'] == []
    assert conflict_records['labels']['del_tar_values'] == []
    assert conflict_records['labels']['app_ori_idx'] == [1, 2, 3, 4, 5, 6]
    assert conflict_records['labels']['app_ori_values'] == [[6], [7], [8], [9], [10], [11]]
    assert conflict_records['labels']['app_tar_idx'] == [3, 4, 5, 6]
    assert conflict_records['labels']['app_tar_values'] == [[6], [7], [8], [9]]
    assert conflict_records['labels']['update_values']['update_tar'] == [{0: [188]}]

    ds.merge("dev", append_resolution='both', update_resolution='theirs', pop_resolution="both")
    assert ds.numpy(aslist=True) == {'labels': [[18], [6], [7], [8], [9], [10], [11], [188], [6], [7], [8], [9]]}


def test_detect_error(storage):
    """ test the error catch."""
    ds = muller.dataset(path=official_path(storage, TEST_DETECT_ERROR),
                       creds=official_creds(storage), overwrite=True)

    ds.create_tensor(name="labels", htype="generic", dtype="int")
    ds.labels.extend([1, 2, 3, 4, 5])
    try:
        ds.detect_merge_conflict("dev")
        assert False, "No exception raises"
    except KeyError as e:
        assert True, f"The commit/branch {e} does not exist in the dataset."
    fir_main = ds.commit("first on main.")

    ds.checkout("dev", create=True)
    ds.labels.extend([1, 2, 3, 4, 5])
    fir_dev = ds.commit("first on dev.")

    ds.labels.extend([1, 2, 3, 4, 5])
    sev_dev = ds.commit("second on dev.")

    ds.checkout(fir_dev, create=False)

    try:
        ds.detect_merge_conflict("main")
        assert False, "No exception raises"
    except DatasetCorruptError as e:
        assert True, f"current commit node is not the head node of current branch.{e}"

    ds.checkout(fir_main, create=False)
    ds.labels.extend([1, 2, 3, 4, 5])

    again_ds = muller.dataset(f"{official_path(storage, TEST_DETECT_ERROR)}@{sev_dev}",
                       creds=official_creds(storage), overwrite=False)

    try:
        again_ds.detect_merge_conflict("main")
        assert False, "No exception raises"
    except DatasetCorruptError as e:
        assert True, f"The given target state has uncommitted changes.{e}"


def test_add_tensor(storage):
    """ Add tensor: The dev branch add two new tensor columns, while the main branch also add a new tensor column."""
    ds = muller.dataset(path=official_path(storage, TEST_DETECT_MERGE),
                       creds=official_creds(storage), overwrite=True)
    ds.create_tensor(name="labels", htype="generic", dtype="int")
    ds.labels.extend([0, 1, 2, 3, 4])
    ds.create_tensor(name="categories", htype="text")
    ds.categories.extend(["a", "b", "c", "d", "e"])

    ds.checkout("dev", create=True)
    ds.create_tensor(name="test1", htype="text")
    ds.test1.extend(["aa", "bb", "cc", "dd", "ee"])
    ds.create_tensor(name="test2", htype="text")
    ds.test1.extend(["aaa", "bbb", "ccc", "ddd", "eee"])
    commit_id_1 = ds.commit()

    ds.checkout("main")
    ds.create_tensor(name="test3", htype="text")
    ds.test3.extend(["aaaa", "bbbb", "cccc", "dddd", "eeee"])

    conflict_tensors, _ = ds.detect_merge_conflict(target_id=commit_id_1, show_value=True)
    assert conflict_tensors == {}

    ds.merge("dev")
    assert len(ds.tensors) == 5


def test_rename_tensor(storage):
    """ Rename tensor: The dev branch rename one tensor columns from the main branch. """
    ds = muller.dataset(path="temp_test", overwrite=True)
    ds.create_tensor(name="labels", htype="generic", dtype="int")
    ds.labels.extend([0, 1, 2, 3, 4])
    ds.create_tensor(name="categories", htype="text")
    ds.categories.extend(["a", "b", "c", "d", "e"])

    ds.checkout("dev", create=True)
    ds.rename_tensor("labels", "new_labels")
    ds.new_labels[2] = 20
    ds.new_labels.extend([500, 600, 700, 800])
    ds.categories.extend(["aa", "bb", "cc", "dd"])
    ds.commit()

    ds.checkout("main")
    ds.merge("dev")
    assert list(ds.tensors.keys()) == ["categories", "new_labels"]
    assert len(ds) == 9


def test_delete_tensor(storage):
    """ Delete tensor: The dev branch delete one tensor columns from the main branch."""
    ds = muller.dataset(path=official_path(storage, TEST_DETECT_MERGE),
                       creds=official_creds(storage), overwrite=True)
    ds.create_tensor(name="labels", htype="generic", dtype="int")
    ds.labels.extend([0, 1, 2, 3, 4])
    ds.create_tensor(name="categories", htype="text")
    ds.categories.extend(["a", "b", "c", "d", "e"])

    ds.checkout("dev", create=True)
    ds.delete_tensor("labels")
    ds.commit()

    ds.checkout("main")
    conflict_tensors, _ = ds.detect_merge_conflict(target_id="dev", show_value=True)
    assert conflict_tensors == {}
    ds.merge("dev", delete_removed_tensors=True)
    assert len(ds.tensors) == 1


def test_conflict_records_append_both(storage):
    """ Test conflict records: append """
    ds = muller.dataset(path=official_path(storage, TEST_DETECT_MERGE),
                       creds=official_creds(storage), overwrite=True)
    ds.create_tensor(name="labels", htype="generic", dtype="int")
    ds.labels.extend([0, 1, 2, 3, 4])
    ds.create_tensor(name="categories", htype="text")
    ds.categories.extend(["a", "b", "c", "d", "e"])

    ds.checkout("dev-1", create=True)
    ds.labels.extend([10, 20, 30])
    ds.categories.extend(["aa", "bb", "cc"])
    ds.commit()

    ds.checkout("main")
    ds.checkout("dev-2", create=True)
    ds.labels.extend([100, 200, 300, 400])
    ds.categories.extend(["aaa", "bbb", "ccc", "ddd"])
    ds.commit()

    ds.checkout("main")
    ds.merge("dev-1")

    _, conflict_records = ds.detect_merge_conflict(target_id="dev-2", show_value=True)
    assert conflict_records["categories"]['app_ori_idx'] == [5, 6, 7]
    assert conflict_records["categories"]['app_ori_values'] == ["aa", "bb", "cc"]
    assert conflict_records["categories"]['app_tar_idx'] == [5, 6, 7, 8]
    assert conflict_records["categories"]['app_tar_values'] == ["aaa", "bbb", "ccc", "ddd"]
    return ds


def test_merge_append_both(storage):
    """ Test conflict records: append both """
    ds = test_conflict_records_append_both(storage)
    ds.merge("dev-2", append_resolution="both")
    assert len(ds) == 12


def test_merge_append_ours(storage):
    """ Test conflict records: append ours """
    ds = test_conflict_records_append_both(storage)
    ds.merge("dev-2", append_resolution="ours")
    assert len(ds) == 8


def test_merge_append_theirs(storage):
    """ Test conflict records: append theirs """
    ds = test_conflict_records_append_both(storage)
    ds.merge("dev-2", append_resolution="theirs")
    assert len(ds) == 9


def test_merge_append_none(storage):
    """ Test conflict records: append None """
    ds = test_conflict_records_append_both(storage)
    try:
        ds.merge("dev-2", append_resolution=None)
        assert False, "No exception raises"
    except MergeConflictError as e:
        assert True, f"Raises MergeConflictError {e}"


def test_conflict_records_update_both(storage):
    """ Test conflict records: update """
    ds = muller.dataset(path=official_path(storage, TEST_DETECT_MERGE),
                       creds=official_creds(storage), overwrite=True)
    ds.create_tensor(name="labels", htype="generic", dtype="int")
    ds.labels.extend([0, 1, 2, 3, 4])
    ds.create_tensor(name="categories", htype="text")
    ds.categories.extend(["a", "b", "c", "d", "e"])

    ds.checkout("dev-1", create=True)
    ds.labels[1] = 10
    ds.categories[1] = "aa"
    ds.labels[2] = 20
    ds.categories[2] = "bb"
    ds.commit()

    ds.checkout("main")
    ds.checkout("dev-2", create=True)
    ds.labels[1] = 100
    ds.labels[2] = 200
    ds.labels[3] = 300
    ds.categories[1] = "aaa"
    ds.categories[2] = "bbb"
    ds.categories[3] = "ccc"
    ds.commit()

    ds.checkout("main")
    ds.merge("dev-1")
    _, conflict_records = ds.detect_merge_conflict(target_id="dev-2", show_value=True)
    assert conflict_records["categories"]["update_values"]["update_ori"][0][1] == ["aa"]
    assert conflict_records["categories"]["update_values"]["update_ori"][1][2] == ["bb"]
    assert conflict_records["categories"]["update_values"]["update_tar"][0][1] == ["aaa"]
    assert conflict_records["categories"]["update_values"]["update_tar"][1][2] == ["bbb"]
    assert conflict_records["labels"]["update_values"]["update_ori"][0][1] == [10]
    assert conflict_records["labels"]["update_values"]["update_ori"][1][2] == [20]
    assert conflict_records["labels"]["update_values"]["update_tar"][0][1] == [100]
    assert conflict_records["labels"]["update_values"]["update_tar"][1][2] == [200]
    return ds


def test_merge_update_ours(storage):
    """ Test conflict records: update ours """
    ds = test_conflict_records_update_both(storage)
    ds.merge("dev-2", update_resolution="ours")
    assert ds.labels.numpy(aslist=True) == [0, 10, 20, 300, 4]


def test_merge_update_theirs(storage):
    """ Test conflict records: update theirs """
    ds = test_conflict_records_update_both(storage)
    ds.merge("dev-2", update_resolution="theirs")
    assert ds.labels.numpy(aslist=True) == [0, 100, 200, 300, 4]


def test_merge_update_none(storage):
    """ Test conflict records: update none """
    ds = test_conflict_records_update_both(storage)
    try:
        ds.merge("dev-2", update_resolution=None)
        assert False, "No exception raises"
    except MergeConflictError as e:
        assert True, f"Raises MergeConflictError {e}"


def test_conflict_records_pop_both(storage):
    """ Test conflict records: pop """
    ds = muller.dataset(path=official_path(storage, TEST_DETECT_MERGE),
                       creds=official_creds(storage), overwrite=True)
    ds.create_tensor(name="labels", htype="generic", dtype="int")
    ds.labels.extend([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ds.create_tensor(name="categories", htype="text")
    ds.categories.extend(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"])

    ds.checkout("dev-1", create=True)
    ds.pop([1, 2, 8])
    ds.commit()

    ds.checkout("main")
    ds.checkout("dev-2", create=True)
    ds.pop([1, 2, 4, 5])
    ds.commit()

    ds.checkout("main")
    _, conflict_records = ds.detect_merge_conflict(target_id="dev-1", show_value=True)
    assert conflict_records["labels"]["del_tar_idx"] == [1, 2, 8]
    assert conflict_records["labels"]["del_tar_values"] == [1, 2, 8]

    ds.merge("dev-1", pop_resolution="theirs")
    _, conflict_records = ds.detect_merge_conflict(target_id="dev-2", show_value=True)
    assert conflict_records["labels"]["del_tar_idx"] == [4, 5]
    assert conflict_records["labels"]["del_tar_values"] == [4, 5]
    return ds


def test_merge_pop_both(storage):
    """ Test conflict records: pop both """
    ds = test_conflict_records_pop_both(storage)
    ds.merge("dev-2", pop_resolution="both")
    assert ds.labels.numpy(aslist=True) == [0, 3, 6, 7, 9]


def test_merge_pop_ours(storage):
    """ Test conflict records: pop ours """
    ds = test_conflict_records_pop_both(storage)
    ds.merge("dev-2", pop_resolution="ours")
    assert ds.labels.numpy(aslist=True) == [0, 3, 4, 5, 6, 7, 9]


def test_merge_pop_theirs(storage):
    """ Test conflict records: pop theirs """
    ds = test_conflict_records_pop_both(storage)
    ds.merge("dev-2", pop_resolution="theirs")
    assert ds.labels.numpy(aslist=True) == [0, 3, 6, 7, 9, 8]  # Note: it may lead to different sequence.
    assert ds.categories.numpy(aslist=True) == ['a', 'd', 'g', 'h', 'j', 'i']


def test_merge_pop_none(storage):
    """ Test conflict records: pop none """
    ds = test_conflict_records_pop_both(storage)
    try:
        ds.merge("dev-2", pop_resolution=None)
        assert False, "No exception raises"
    except MergeConflictError as e:
        assert True, f"Raises MergeConflictError {e}"


def test_long_merge(storage):
    """ Test long merge with create new tensor."""
    ds = muller.dataset(path=official_path(storage, TEST_DETECT_MERGE),
                       creds=official_creds(storage), overwrite=True)
    ds.create_tensor(name="labels", htype="generic", dtype="int")
    ds.labels.extend([0, 1, 2, 3, 4])
    ds.create_tensor(name="categories", htype="text")
    ds.categories.extend(["a", "b", "c", "d", "e"])
    ds.commit()

    ds.checkout("dev-1", create=True)
    ds.labels.extend([50, 60, 70])
    ds.categories.extend(["ff", "gg", "hh"])
    ds.labels[3] = 30
    ds.pop(1)
    ds.commit('commit on dev-1')

    ds.checkout('main')
    ds.checkout("dev-2", create=True)
    ds.labels.extend([500, 600, 700, 800])
    ds.categories.extend(["fff", "ggg", "hhh", "iii"])
    ds.labels[3] = 300
    ds.labels[4] = 400
    ds.pop([1, 2])
    ds.commit()

    ds.checkout('main')
    ds.merge("dev-1", pop_resolution="theirs")
    assert len(ds.labels.numpy(aslist=True)) == 7
    assert ds.labels.numpy(aslist=True)[-1] == [70]

    conflict_tensors, conflict_records = ds.detect_merge_conflict("dev-2", show_value=True)

    assert conflict_tensors == {}
    assert conflict_records['categories']['app_ori_idx'] == [4, 5, 6]
    assert conflict_records['categories']['app_ori_values'] == ["ff", "gg", "hh"]
    assert conflict_records['labels']['app_ori_values'] == [50, 60, 70]

    ds.merge("dev-2", append_resolution="both", pop_resolution="ours", update_resolution="theirs")
    assert ds.labels.numpy(aslist=True) == [[0], [2], [300], [400], [50], [60], [70], [500], [600], [700], [800]]
    assert ds.categories.numpy(aslist=True) == [['a'], ['c'], ['d'], ['e'], ['ff'], ['gg'], ['hh'], ['fff'],
                                            ['ggg'], ['hhh'], ['iii']]
    with ds:
        ds.checkout("dev-3", create=True)
        ds.create_tensor("features", htype="generic", dtype="float")
        ds.features.extend(np.arange(0, 1.1, 0.1))
        ds.commit()

    # Go back to the main branch, and merge branch dev-3.
    ds.checkout("main")
    conflict_tensors, conflict_records = ds.detect_merge_conflict("dev-3",
                                                                  show_value=True)
    assert conflict_tensors == {}
    assert conflict_records['categories']['app_ori_idx'] == []
    assert conflict_records['categories']['app_ori_values'] == []
    assert conflict_records['labels']['app_ori_idx'] == []
    assert conflict_records['labels']['app_ori_values'] == []

    ds.merge("dev-3")
    assert len(ds.tensors) == 3
    assert 'features' in ds.tensors


def test_uuid_get(storage):
    """test extend hidden tensor's uuid repeated situation."""
    ds = muller.dataset(path=official_path(storage, TEST_DETECT_MERGE),
                       creds=official_creds(storage), overwrite=True)
    ds.create_tensor(name="labels", htype="generic", dtype="int")
    ds.labels.extend([0, 1, 2, 3, 4])
    ds.create_tensor(name="categories", htype="text")
    ds.categories.extend(["a", "b", "c", "d", "e"])

    ds.checkout("dev-1", create=True)
    ds.labels.extend([10, 20, 30])
    ds.categories.extend(["aa", "bb", "cc"]) # ["a", "b", "c", "d", "e", "aa", "bb", "cc"]
    ds.commit()

    ds.checkout("main")
    ds.checkout("dev-2", create=True)
    ds.labels.extend([100, 200, 300, 400])
    ds.categories.extend(["aaa", "bbb", "ccc", "ddd"]) # ["a", "b", "c", "d", "e", "aaa", "bbb", "ccc", "ddd"]
    ds.commit()

    ds.checkout("main")
    ds.merge("dev-1") # ["a", "b", "c", "d", "e", "aa", "bb", "cc"]
    assert ds.labels.numpy(aslist=True) == [[0], [1], [2], [3], [4], [10], [20], [30]]
    assert len(ds.categories.numpy(aslist=True)) == 8
    assert len(ds._uuid.numpy(aslist=True)) == 8

    conflict_tensors, conflict_records = ds.detect_merge_conflict(target_id="dev-2", show_value=True)
    assert conflict_tensors == {}
    assert conflict_records["categories"]['app_ori_idx'] == [5, 6, 7]
    assert conflict_records["categories"]['app_ori_values'] == ["aa", "bb", "cc"]
    assert conflict_records["categories"]['app_tar_idx'] == [5, 6, 7, 8]
    assert conflict_records["categories"]['app_tar_values'] == ["aaa", "bbb", "ccc", "ddd"]


def complicated_merge_1(storage):
    """ Test conflict records: complicated merge - create dataset and conduct some edit """
    ds = muller.dataset(path=official_path(storage, TEST_DETECT_MERGE),
                       creds=official_creds(storage), overwrite=True)
    ds.create_tensor(name="labels", htype="generic", dtype="int")
    ds.labels.extend([0, 1, 2, 3, 4])
    ds.create_tensor(name="categories", htype="text")
    ds.categories.extend(["a", "b", "c", "d", "e"])

    ds.checkout("dev-1", create=True)
    ds.labels.extend([50, 60, 70])
    ds.categories.extend(["ff", "gg", "hh"])
    ds.labels[3] = 30
    ds.pop(1)
    ds.commit()

    ds.checkout("main")
    ds.checkout("dev-2", create=True)
    ds.labels.extend([500, 600, 700, 800])
    ds.categories.extend(["fff", "ggg", "hhh", "iii"])
    ds.labels[3] = 300
    ds.labels[4] = 400
    ds.pop([1, 2])
    ds.commit()

    ds.checkout("main")
    ds.merge("dev-1", pop_resolution="theirs")
    return ds


def test_complicated_merge_1(storage):
    """ Test conflict records: complicated merge 1"""
    ds = complicated_merge_1(storage)
    ds.merge("dev-2", append_resolution="both", pop_resolution="both", update_resolution="theirs")
    assert ds.labels.numpy(aslist=True) == [0, 300, 400, 50, 60, 70, 500, 600, 700, 800]


def test_complicated_merge_2(storage):
    """ Test conflict records: complicated merge 2"""
    ds = complicated_merge_1(storage)
    ds.merge("dev-2", append_resolution="ours", pop_resolution="both", update_resolution="theirs")
    assert ds.labels.numpy(aslist=True) == [0, 300, 400, 50, 60, 70]


def test_complicated_merge_3(storage):
    """ Test conflict records: complicated merge 3"""
    ds = complicated_merge_1(storage)
    ds.merge("dev-2", append_resolution="theirs", pop_resolution="both", update_resolution="theirs")
    assert ds.labels.numpy(aslist=True) == [0, 300, 400, 500, 600, 700, 800]


def test_complicated_merge_4(storage):
    """ Test conflict records: complicated merge 4"""
    ds = complicated_merge_1(storage)
    ds.merge("dev-2", append_resolution="both", pop_resolution="ours", update_resolution="theirs")
    assert ds.labels.numpy(aslist=True) == [0, 2, 300, 400, 50, 60, 70, 500, 600, 700, 800]


def test_complicated_merge_5(storage):
    """ Test conflict records: complicated merge 5"""
    ds = complicated_merge_1(storage)
    ds.merge("dev-2", append_resolution="both", pop_resolution="theirs", update_resolution="theirs")
    assert ds.labels.numpy(aslist=True) == [0, 300, 400, 50, 60, 70, 500, 600, 700, 800]


def test_complicated_merge_6(storage):
    """ Test conflict records: complicated merge 6"""
    ds = complicated_merge_1(storage)
    ds.merge("dev-2", append_resolution="both", pop_resolution="both", update_resolution="ours")
    assert ds.labels.numpy(aslist=True) == [0, 30, 400, 50, 60, 70, 500, 600, 700, 800]


def test_complicated_merge_7(storage):
    """ Test conflict records: complicated merge 7"""
    ds = complicated_merge_1(storage)
    try:
        ds.merge("dev-2", append_resolution="both", pop_resolution="both", update_resolution=None)
        assert False, "No exception raises"
    except MergeConflictError as e:
        assert True, f"Raises MergeConflictError {e}"

    try:
        ds.merge("dev-2", append_resolution="both", pop_resolution=None, update_resolution=None)
        assert False, "No exception raises"
    except MergeConflictError as e:
        assert True, f"Raises MergeConflictError {e}"

    try:
        ds.merge("dev-2", append_resolution=None, pop_resolution="both", update_resolution=None)
        assert False, "No exception raises"
    except MergeConflictError as e:
        assert True, f"Raises MergeConflictError {e}"


def complicated_merge_2(storage):
    """ Test conflict records: complicated merge 2 - create dataset and conduct some edit """
    ds = muller.dataset(path=official_path(storage, TEST_DETECT_MERGE),
                       creds=official_creds(storage), overwrite=True)
    ds.create_tensor(name="labels", htype="generic", dtype="int")
    ds.labels.extend([0, 1, 2, 3, 4])
    ds.create_tensor(name="categories", htype="text")
    ds.categories.extend(["a", "b", "c", "d", "e"])

    ds.checkout("dev-1", create=True)
    ds.pop(1)
    ds.labels.extend([50, 60, 70])
    ds.categories.extend(["ff", "gg", "hh"])
    ds.labels[2] = 30
    ds.commit()

    ds.checkout("main")
    ds.checkout("dev-2", create=True)
    ds.pop([1, 2])
    ds.labels.extend([500, 600, 700, 800])
    ds.categories.extend(["fff", "ggg", "hhh", "iii"])
    ds.labels[1] = 300
    ds.labels[2] = 400
    ds.commit()

    ds.checkout("main")
    ds.merge("dev-1", pop_resolution="theirs")
    return ds


def test_complicated_merge_8(storage):
    """ Test conflict records: complicated merge 8 """
    ds = complicated_merge_2(storage)
    ds.merge("dev-2", append_resolution="both", pop_resolution="both", update_resolution="theirs")
    assert ds.labels.numpy(aslist=True) == [0, 300, 400, 50, 60, 70, 500, 600, 700, 800]


def test_complicated_merge_9(storage):
    """ Test conflict records: complicated merge 9 """
    ds = complicated_merge_2(storage)
    ds.merge("dev-2", append_resolution="ours", pop_resolution="both", update_resolution="theirs")
    assert ds.labels.numpy(aslist=True) == [0, 300, 400, 50, 60, 70]


def test_complicated_merge_10(storage):
    """ Test conflict records: complicated merge 10 """
    ds = complicated_merge_2(storage)
    ds.merge("dev-2", append_resolution="theirs", pop_resolution="both", update_resolution="theirs")
    assert ds.labels.numpy(aslist=True) == [0, 300, 400, 500, 600, 700, 800]  # The sequence may be different


def test_complicated_merge_11(storage):
    """ Test conflict records: complicated merge 11 """
    ds = complicated_merge_2(storage)
    ds.merge("dev-2", append_resolution="both", pop_resolution="ours", update_resolution="theirs")
    assert ds.labels.numpy(aslist=True) == [0, 2, 300, 400, 50, 60, 70, 500, 600, 700, 800]


def test_complicated_merge_12(storage):
    """ Test conflict records: complicated merge 12 """
    ds = complicated_merge_2(storage)
    ds.merge("dev-2", append_resolution="both", pop_resolution="theirs", update_resolution="theirs")
    assert ds.labels.numpy(aslist=True) == [0, 300, 400, 50, 60, 70, 500, 600, 700, 800]


def test_complicated_merge_13(storage):
    """ Test conflict records: complicated merge 13 """
    ds = complicated_merge_2(storage)
    ds.merge("dev-2", append_resolution="both", pop_resolution="both", update_resolution="ours")
    assert ds.labels.numpy(aslist=True) == [0, 30, 400, 50, 60, 70, 500, 600, 700, 800]


if __name__ == '__main__':
    pytest.main(["-s", "test_detect_merge.py"])

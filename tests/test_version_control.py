# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import muller
from muller.constants import FIRST_COMMIT_ID
from muller.core.auth.authorization import obtain_current_user
from muller.core.version_control.commit_node import CommitNode
from muller.core.version_control.functions import (
    _merge_commit_node_maps,
    _version_info_from_json,
    _version_info_to_json,
)
from muller.util.exceptions import (
    ExportDataFrameLimit,
    MergeConflictError,
    VersionControlError,
)

from tests.constants import VERSION_CONTROL_TEST_PATH
from tests.utils import official_path, official_creds

branch_commit_map = {"main": "f"}
branch_info = {"main":{"based_on": None, "create_time": None}}


def load_dataset(storage, branch: str = "main", overwrite: bool = False):
    if branch == "main":
        ds = muller.dataset(path=official_path(storage, VERSION_CONTROL_TEST_PATH),
                           creds=official_creds(storage), overwrite=overwrite)
        return ds
    main_path = official_path(storage, VERSION_CONTROL_TEST_PATH)
    dev_path = main_path.rstrip("/") + "@" + branch
    ds = muller.load(path=dev_path, creds=official_creds(storage))
    return ds


def test_merge_commit_node_map():
    root = CommitNode("main", FIRST_COMMIT_ID)
    a = CommitNode("main", "a")
    b = CommitNode("main", "b")
    c = CommitNode("main", "c")
    e = CommitNode("main", "e")
    root.add_successor(a, "me", "commit a")
    root.add_successor(b, "me", "commit b")
    a.add_successor(c, "me", "commit c")
    c.add_successor(e, "me", "commit e")
    map1 = {
        FIRST_COMMIT_ID: root,
        "a": a,
        "b": b,
        "c": c,
        "e": e,
    }

    root = CommitNode("main", FIRST_COMMIT_ID)
    a = CommitNode("main", "a")
    b = CommitNode("main", "b")
    d = CommitNode("main", "d")
    f = CommitNode("main", "f")
    root.add_successor(a, "commit a")
    root.add_successor(b, "commit b")
    b.add_successor(d, "commit d")
    d.add_successor(f, "commit f")

    map2 = {
        FIRST_COMMIT_ID: root,
        "a": a,
        "b": b,
        "d": d,
        "f": f,
    }

    merged = _merge_commit_node_maps(map1, map2)

    assert set(merged.keys()) == {FIRST_COMMIT_ID, "a", "b", "c", "d", "e", "f"}
    get_children = lambda node: set(c.commit_id for c in node.children)
    assert get_children(merged[FIRST_COMMIT_ID]) == {"a", "b"}
    assert get_children(merged["a"]) == set("c")
    assert get_children(merged["b"]) == set("d")
    assert get_children(merged["c"]) == set("e")
    assert get_children(merged["d"]) == set("f")

    # Test json encoding
    version_info = {"commit_node_map": merged, "branch_commit_map": branch_commit_map, "branch_info": branch_info}
    encoded = _version_info_to_json(version_info)
    decoded = _version_info_from_json(encoded)
    assert decoded["branch_commit_map"] == branch_commit_map
    merged = decoded["commit_node_map"]

    assert set(merged.keys()) == {FIRST_COMMIT_ID, "a", "b", "c", "d", "e", "f"}
    get_children = lambda node: set(c.commit_id for c in node.children)
    assert get_children(merged[FIRST_COMMIT_ID]) == {"a", "b"}
    assert get_children(merged["a"]) == set("c")
    assert get_children(merged["b"]) == set("d")
    assert get_children(merged["c"]) == set("e")
    assert get_children(merged["d"]) == set("f")


def test_commit_details(storage):
    """Function for testing commit details."""
    ds = load_dataset(storage, overwrite=True)

    ds.create_tensor(name="array_1")
    ds.array_1.append(np.array([1]))
    first_commit = ds.commit("This is first commit.")
    assert first_commit == "firstdbf9474d461a19e9333c2fd19b46115348f"

    ds.array_1.append([2])
    second_commit = ds.commit("Append once at second commit.")
    assert len(ds.commits()) == 2
    assert ds.get_commit_details(second_commit)['author'] == obtain_current_user()
    assert ds.get_commit_details(second_commit)['commit'] == second_commit
    assert ds.get_commit_details(second_commit)['message'] == "Append once at second commit."


def test_reset(storage):
    """Function testing reset."""
    ds = load_dataset(storage) # return the latest version on main branch.

    ds.checkout("dev", create=True)
    assert ds.branch == "dev"
    assert list(ds.branches)[0] == "main"
    assert list(ds.branches)[1] == "dev"
    assert len(ds.branches) == 2

    ds.array_1.append([3])
    ds.commit("This is third commit, and first commit on dev.")

    ds.array_1.append([4])
    assert ds.has_head_changes

    ds.reset()
    assert not ds.has_head_changes

    ds.pop(2)
    ds.array_1.info.update(array=1)
    ds.array_1[1] = [5] # update
    ds.array_1.append([6])
    ds.commit("This is fourth commit, and second commit on dev.")


def test_diff_to_prev(storage):
    """Function testing diff to previous version."""
    ds = load_dataset(storage, "dev", False)

    third_commit = ds.commits()[1]['commit']
    fourth_commit = ds.commits()[0]['commit']

    assert ds.diff_to_prev(third_commit, as_dict=True, show_value=True)['tensor'][0]['array_1']['add_value'] == [[3]]
    assert ds.diff_to_prev(third_commit, as_dict=True, show_value=True)['tensor'][0]['array_1']['data_added'] == [2, 3]
    assert len(ds.diff_to_prev(fourth_commit, as_dict=True,
                               show_value=True, limit=2)['tensor'][0]['array_1']['add_value']) == 1
    try:
        update_values_to_prev = ds.diff_to_prev(fourth_commit, as_dict=True,
                        show_value=True, limit=2)['tensor'][0]['array_1']['updated_values']
    except KeyError as e:
        raise ValueError(f"Please check the returned dict.") from e
    assert update_values_to_prev == [[5]]
    try:
        data_updated_to_prev = ds.diff_to_prev(fourth_commit, as_dict=True,
                        show_value=True, limit=2)['tensor'][0]['array_1']['data_updated']
    except KeyError as e:
        raise ValueError(f"Please check the returned dict.") from e
    assert data_updated_to_prev == {1}


def test_diff(storage):
    """Function testing diff."""
    ds = load_dataset(storage, "dev", False)

    third_commit = ds.commits()[1]['commit']
    fourth_commit = ds.commits()[0]['commit']

    assert ds.diff(third_commit, fourth_commit, as_dict=True)['tensor'][1][0]["array_1"]["data_added"] == [2,3]
    assert ds.diff(third_commit, fourth_commit, as_dict=True)['tensor'][1][0]['array_1']['data_deleted'] == {2}
    assert ds.diff(third_commit, fourth_commit, as_dict=True)['tensor'][1][0]['array_1']['data_updated'] == {1}
    assert ds.diff(third_commit, fourth_commit, as_dict=True)['tensor'][1][0]['array_1']['info_updated']

    assert ds.diff(third_commit, fourth_commit, as_dict=True, show_value=True)['tensor'][0] == []
    assert ds.diff(third_commit, fourth_commit,
                   as_dict=True, show_value=True)['tensor'][1][0]['array_1']['add_value'] == [[6]]
    assert ds.diff(third_commit, fourth_commit, as_dict=True,
            show_value=True)['tensor'][1][0]['array_1']['updated_values'] == [[5]]

    assert ds.diff(third_commit, fourth_commit,
                   as_dict=True, show_value=True)['tensor'][1][0]['array_1']['data_deleted'] == {2}

    assert ds.diff(third_commit, fourth_commit,
                   as_dict=True, show_value=True)['tensor'][1][0]['array_1']['data_deleted_values'] == [[3]]

    assert len(ds.diff(third_commit, fourth_commit,
                       as_dict=True, show_value=True)['tensor'][1][0]['array_1']['data_deleted_ids']) == 1

    assert len(ds.diff(third_commit, fourth_commit,
                       as_dict=True, show_value=True)['tensor'][1][0]['array_1']['add_value']) == 1

    assert len(ds.diff(third_commit, fourth_commit,
                       as_dict=True, show_value=True)['tensor'][1][0]['array_1']['updated_values']) == 1

    assert len(ds.diff(third_commit, fourth_commit,
                       as_dict=True, show_value=True)['tensor'][1][0]['array_1']['data_deleted_values']) == 1

    assert len(ds.diff(third_commit, fourth_commit,
                       as_dict=True, show_value=True, offset=2)['tensor'][1][0]['array_1']['add_value']) == 0

    assert len(ds.diff(third_commit, fourth_commit,
                       as_dict=True, show_value=True, offset=2)['tensor'][1][0]['array_1']['updated_values']) == 0

    assert len(ds.diff(third_commit, fourth_commit,
                       as_dict=True, show_value=True, offset=2)['tensor'][1][0]['array_1']['data_deleted_values']) == 1

    assert len(ds.diff(third_commit, fourth_commit,
                       as_dict=True, show_value=True, limit=2)['tensor'][1][0]['array_1']['add_value']) == 1

    assert len(ds.diff(third_commit, fourth_commit,
                       as_dict=True, show_value=True, limit=2)['tensor'][1][0]['array_1']['updated_values']) == 1

    assert len(ds.diff(third_commit, fourth_commit,
                   as_dict=True, show_value=True, limit=2)['tensor'][1][0]['array_1']['data_deleted_values']) == 0

    assert len(ds.diff(third_commit, fourth_commit,
                       as_dict=True, show_value=True, offset=1, limit=1)
               ['tensor'][1][0]['array_1']['add_value']) == 0

    assert len(ds.diff(third_commit, fourth_commit,
                       as_dict=True, show_value=True, offset=1, limit=1)
               ['tensor'][1][0]['array_1']['updated_values']) == 1

    assert len(ds.diff(third_commit, fourth_commit, as_dict=True, show_value=True, offset=1,
                       limit=1)['tensor'][1][0]['array_1']['data_deleted_values']) == 0

    assert 'array_1' not in ds.diff(third_commit, fourth_commit,
                       as_dict=True, asrow=True, show_value=True)['tensor'][1][0]

    assert len(ds.diff(third_commit, fourth_commit,
                       as_dict=True, asrow=True, show_value=True)['tensor'][1][0]['add_value']) == 1
    try:
        add_value_array_1 = ds.diff(third_commit, fourth_commit, as_dict=True, asrow=True,
                show_value=True)['tensor'][1][0]['add_value'][0]['array_1']
    except KeyError as e:
        raise ValueError(f"Please check the returned dict.") from e
    assert add_value_array_1 == np.array([6])


def test_commits_between(storage):
    """Function testing commits between versions."""
    ds = load_dataset(storage, "dev", False)

    ds.checkout("alt", create=True)
    assert ds.branch == "alt"
    assert len(ds.branches) == 3

    assert len(ds.commits_between(as_dict=True)[0]) == 3


def test_delete_branches(storage):
    """Function testing delete branches"""
    ds = load_dataset(storage, "main", False)
    ds.merge("alt")

    with pytest.raises(VersionControlError) as e:
        ds.delete_branch("dev")
    assert e.type == VersionControlError
    assert str(e.value) == "Cannot delete branch dev because it has sub-branches"

    with pytest.raises(VersionControlError) as e:
        ds.delete_branch("main")
    assert e.type == VersionControlError
    assert str(e.value) == "Cannot delete the currently checked out branch: main"
    ds.checkout("dev", create=False)
    with pytest.raises(VersionControlError) as e:
        ds.delete_branch("main")
    assert e.type == VersionControlError
    assert str(e.value) == "Cannot delete the main branch"

    ds.delete_branch("alt")
    assert len(ds.branches) == 2
    assert ds.array_1.numpy(aslist=True)[0] == [1]
    assert ds.array_1.numpy(aslist=True)[1] == [5]


def test_commits_under(storage):
    """Function testing commits under branch."""
    ds = load_dataset(storage, "dev", False)

    assert len(ds.commits_under()) == 4
    assert ds.commits_under()[0].branch == "dev"

    assert len(ds.commits_under("main")) == 5
    order_date = ds.commits_under("main", True)
    assert order_date[3].commit_time >= order_date[4].commit_time


def test_get_children_nodes(storage):
    ds = load_dataset(storage, overwrite=True)
    ds.create_tensor(name="array_1")
    ds.array_1.append(np.array([1]))
    first_commit = ds.commit()
    assert first_commit == "firstdbf9474d461a19e9333c2fd19b46115348f"

    ds.array_1.append([2])
    second_commit = ds.commit("append once")

    ds.checkout("dev", create=True)

    ds.array_1.append([3])
    third_commit = ds.commit("third_commit")
    ds.array_1.append([4])

    ds.reset()
    ds.pop(2)
    fourth_commit = ds.commit("fourth_commit")

    ds.checkout("alt", create=True)
    ds.array_1.append([6])
    fifth_commit = ds.commit("fifth_commit")

    ds.checkout("main")
    ds.array_1.append([7])
    sixth_commit = ds.commit("sixth_commit")

    target_info = third_commit
    result_tree_dict = ds.get_children_nodes(target_commit_id=target_info)
    assert result_tree_dict['children'][0]['commit_id'] == fourth_commit

    target_info_1 = FIRST_COMMIT_ID
    result_tree_dict_1 = ds.get_children_nodes(target_commit_id=target_info_1)

    assert result_tree_dict_1['children'][0]['commit_id'] == second_commit
    assert result_tree_dict_1['children'][0]['children'][1]['commit_id'] == third_commit
    assert result_tree_dict_1['children'][0]['children'][1]['children'][0]['children'][1]['commit_id'] == fifth_commit
    assert result_tree_dict_1['children'][0]['children'][0]['commit_id'] == sixth_commit


def test_get_commits(storage):
    ds = load_dataset(storage, overwrite=True)
    with ds:
        ds.create_tensor("test", htype="text")
        ds.create_tensor("test2", htype="text")
    with ds:
        for _ in range(5):
            ds.test.append("first")
            ds.test2.append("first")
    ds.commit("first commit")
    ds.checkout("exp456", create=True)
    with ds:
        for _ in range(5):
            ds.test.append("exp456")
            ds.test2.append("exp456")
    ds.commit("exp456")
    ds.checkout("exp457", create=True)
    with ds:
        for _ in range(5):
            ds.test.append("exp457")
            ds.test2.append("exp457")
    ds.commit("exp457")
    with ds:
        for _ in range(5):
            ds.test.append("exp457_2")
            ds.test2.append("exp457_2")
    ds.commit("exp457_2")
    ds.checkout("main")
    ds.checkout("sft280", create=True)
    with ds:
        for _ in range(5):
            ds.test.append("sft280")
            ds.test2.append("sft280")
    ds.commit("sft280")
    ds.merge("exp457", append_resolution='both', update_resolution='ours', pop_resolution="ours")
    commit_messages = ds.commits(ordered_by_date=True)
    assert len(commit_messages) == 6
    for i in range(1, len(commit_messages)):
        assert commit_messages[i]["time"] <= commit_messages[i-1]["time"]
    commit_messages = ds.commits(ordered_by_date=False)
    assert len(commit_messages) == 6
    assert commit_messages[-1]["message"] == "first commit"


def test_fast_forward_merge(storage):
    """Function testing fast-forward merge."""
    ds = load_dataset(storage, overwrite=True)

    with ds:
        ds.create_tensor("test", htype="text")
        ds.create_tensor("test2", htype="text")

    with ds:
        for _ in range(5):
            ds.test.append("first")
            ds.test2.append("first")
    ds.commit("first commit on main.")

    ds.checkout("dev", create=True)
    ds.test.append("dev_first")
    ds.test2.append("dev_first")

    ds.test[2] = "update"
    ds.test2[2] = "update"

    ds.pop(0)
    ds.commit("first commit on dev.")

    ds.pop(0)
    ds.commit("sec commit on dev.")

    ds.checkout("main", create=False)

    with pytest.raises(MergeConflictError) as e:
        ds.merge(target_id="dev", append_resolution=None, update_resolution=None, pop_resolution=None)
    assert e.type == MergeConflictError

    ds.merge(target_id="dev", append_resolution='both', update_resolution='theirs', pop_resolution="theirs")
    test_value = ds.test.numpy(aslist=True)
    assert len(test_value) == 4
    assert test_value[0] == ['update']
    assert test_value[1] == ['first']


def test_3_way_merge(storage):
    """Function testing 3 way merge."""
    ds = load_dataset(storage, overwrite=True)

    with ds:
        ds.create_tensor("test", htype="text")
        ds.create_tensor("test2", htype="text")

    with ds:
        for _ in range(5):
            ds.test.append("first")
            ds.test2.append("first")
    ds.commit("first commit on main.")

    ds.checkout("dev", create=True)
    ds.test[2] = "update"
    ds.test2[2] = "update"
    ds.pop(0)
    ds.commit("first commit on dev.")

    ds.checkout("main", create=False)
    ds.test.append("main_sec")
    ds.test2.append("main_sec")
    ds.pop(1)
    ds.commit("sec commit on main.")

    with pytest.raises(MergeConflictError) as e:
        ds.merge(target_id="dev", append_resolution=None, update_resolution=None, pop_resolution=None)
    assert e.type == MergeConflictError

    ds.merge(target_id="dev", append_resolution='both', update_resolution='theirs', pop_resolution="theirs")
    test_value = ds.test.numpy(aslist=True)

    assert len(test_value) == 5
    assert test_value[0] == ['update']
    assert test_value[3] == ['main_sec']


def test_merge_delete_tensor(storage):
    """Function testing merging 2 branches one of which has deleted a common tensor before."""

    # 1、Case when 2 branches both deleted a tensor
    ds = load_dataset(storage, overwrite=True)
    ds.create_tensor("arr")
    ds.checkout("br_1", create=True)
    ds.delete_tensor("arr")
    ds.commit()
    ds.checkout("main")
    ds.delete_tensor("arr")
    ds.merge("br_1")

    assert len(ds.tensors) == 0     # The tensor should be deleted.

    # 2、Case when target branch deleted a tensor, while original did not,
    # (a) when argument delete_removed_tensors equals False
    ds = load_dataset(storage, overwrite=True)
    ds.create_tensor("arr")
    ds.checkout("br_1", create=True)
    ds.delete_tensor("arr")
    ds.commit()
    ds.checkout("main")
    ds.merge('br_1')

    assert len(ds.tensors) == 1     # The tensor should not be deleted

    # (a) when argument delete_removed_tensors equals False
    ds.merge('br_1', delete_removed_tensors=True)
    assert len(ds.tensors) == 0     # The tensor should be deleted now

    # 3、Case when original branch deleted a tensor, while target did not
    ds = load_dataset(storage, overwrite=True)
    ds.create_tensor("arr")
    ds.checkout("br_1", create=True)
    ds.checkout("main")
    ds.delete_tensor("arr")
    ds.merge('br_1')
    assert len(ds.tensors) == 0     # The tensor should be deleted.


def test_filter_with_checkout(storage):
    """Filter function with checkout."""

    ds = load_dataset(storage, overwrite=True)

    ds.create_tensor('labels', htype='generic', dtype='int')
    ds.create_tensor('categories', htype='text')

    ds.labels.extend([10, 2, 30, 4, 50, 6, 70, 8, 90, 100] * 2)
    ds.categories.extend(['agent', '情感', '生成', '写作', '情感', 'agent', '生成', '写作', '情感', '写作'] * 2)

    ds.checkout('dev', create=True)

    ds.labels.extend([10, 2, 30, 4, 50, 6, 70, 8, 90, 100] * 2)
    ds.categories.extend(
        ['agent2', '情感2', '生成2', '写作2', '情感2', 'agent2', '生成2', '写作2', '情感2', '写作2'] * 2)

    ds_1 = ds.filter_vectorized([("labels", ">", 50, False),
                                 ("categories", "==", '写作2', False)], ["OR"])
    assert len(ds_1.filtered_index) == 16


def test_direct_diff(storage):
    """Function testing direct diff."""
    ds = load_dataset(storage, overwrite=True)
    ds.create_tensor(name="labels", htype="generic", dtype="int")
    ds.create_tensor(name="categories", htype="text")
    ds.labels.extend([0, 1, 2, 3, 4])
    ds.categories.extend(["a", "b", "c", "d", "e"])

    ds.checkout("dev", create=True)
    ds.labels[0] = 11
    ds.categories[0] = "haha"
    dev_1 = ds.commit("first on dev")
    ds.checkout("main", create=False)

    ds.checkout("dev_2", create=True)
    ds.labels.extend([5, 6])
    ds.categories.extend(["f", "g"])
    ds.commit("first on dev_2")

    ds.labels[1] = 111
    ds.categories[1] = "xixixi"
    dev_2 = ds.commit("sec on dev_2")

    final_dict = ds.direct_diff(dev_1, dev_2)
    records = final_dict.get("modified_records", {})

    assert len(records['labels']['added_values']) == len(records['categories']['added_values']) == 2
    for key, _ in records['labels']['added_values'].items():
        assert records['categories']['added_values'][key]
    assert len(records['labels']['edited_values_tar1']) == len(records['labels']['edited_values_tar2']) == 2
    for key, _ in records['categories']['edited_values_tar1'].items():
        assert records['categories']['edited_values_tar2'][key]


def test_direct_diff_tensor(storage):
    """Function to test the tensor name of direct diff."""
    ds = load_dataset(storage, overwrite=True)
    ds.create_tensor(name="labels", htype="generic", dtype="int")
    ds.labels.extend([0, 1, 2, 3, 4])

    ds.checkout("dev", create=True)
    ds.create_tensor(name="categories", htype="text")
    ds.categories.extend(["a", "b", "c", "d", "e"])
    dev_1 = ds.commit("first on dev")

    ds.checkout("main", create=False)

    ds.checkout("dev_2", create=True)
    ds.create_tensor(name="annotations", htype="text")
    ds.annotations.extend(["a", "b", "c", "d", "e"])
    ds.commit("first on dev_2")

    ds.create_tensor(name="scores", htype="generic")
    ds.scores.extend([100, 99, 98, 100, 97])
    dev_2 = ds.commit("sec on dev_2")
    final_dict = ds.direct_diff(dev_1, dev_2)
    new_tensors = final_dict["added_columns"]
    removed_tensors = final_dict["removed_columns"]
    records = final_dict.get("modified_records", {})

    assert new_tensors == {'annotations', 'scores'}
    assert removed_tensors == {'categories'}
    assert records['labels']['added_values'] == records['labels']['removed_values'] == {}


def test_direct_diff_as_dataframe(storage):
    """Function to test dataframe return by direct diff."""
    ds = load_dataset(storage, overwrite=True)
    ds.create_tensor(name="labels", htype="generic", dtype="int")
    ds.create_tensor(name="categories", htype="text")
    ds.create_tensor(name="test1", htype="generic", dtype="int")
    ds.labels.extend([0, 1, 2, 3, 4])
    ds.categories.extend(["a", "b", "c", "d", "e"])
    ds.test1.extend([100, 101, 102, 103, 104])

    ds.checkout("dev", create=True)
    ds.labels[0] = 11
    ds.categories[3] = "haha"
    ds.pop(2)
    ds.labels.append(100)
    ds.categories.append("hello")
    ds.test1.append(105)
    dev_1 = ds.commit("first on dev")

    ds.checkout("main", create=False)
    ds.checkout("dev_2", create=True)
    ds.delete_tensor("test1")
    ds.labels.extend([5, 6])
    ds.categories.extend(["f", "g"])
    ds.commit("first on dev_2")
    ds.labels[1] = 111
    ds.categories[1] = "xixixi"
    ds.create_tensor(name="test2", htype="generic", dtype="int")
    ds.test2.extend([100, 101, 102, 103, 104, 105, 106])
    dev_2 = ds.commit("sec on dev_2")

    final_df_dict = ds.direct_diff(dev_1, dev_2, as_dataframe=True)

    assert_frame_equal(pd.DataFrame({"added_columns": ["test2"]}), final_df_dict["added_columns"])

    assert_frame_equal(pd.DataFrame({"removed_columns": ["test1"]}), final_df_dict["removed_columns"])

    assert (len(final_df_dict["added_rows"]["uuid"]) == len(final_df_dict["added_rows"]["labels"])
            == len(final_df_dict["added_rows"]["categories"]) == 3)

    assert (len(final_df_dict["removed_rows"]["uuid"]) == len(final_df_dict["removed_rows"]["labels"])
            == len(final_df_dict["removed_rows"]["categories"]) == 1)

    assert len(final_df_dict["edited_rows"]["uuid"]) == 3


def test_direct_diff_as_large_dataframe(storage):
    """Function to test large dataframe return by direct diff."""
    ds = load_dataset(storage, overwrite=True)
    ds.create_tensor(name="labels", htype="generic", dtype="int")
    ds.labels.extend([0, 1, 2, 3, 4])
    id_1 = ds.commit()

    ds.labels.extend([0, 1, 2, 3, 4] * 20001)
    id_2 = ds.commit()

    try:
        final_df_dict = ds.direct_diff(id_1, id_2, as_dataframe=True)
    except ExportDataFrameLimit as e:
        assert True, f"exception: {e}"

    final_df_dict = ds.direct_diff(id_1, id_2, as_dataframe=True, force=True)
    assert len(final_df_dict["added_rows"]["uuid"]) == len(final_df_dict["added_rows"]["labels"]) == 100005


if __name__ == '__main__':
    pytest.main(["-s", "test_version_control.py"])

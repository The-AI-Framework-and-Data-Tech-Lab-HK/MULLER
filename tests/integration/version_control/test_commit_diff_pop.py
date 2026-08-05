# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

"""Dataset-level regression tests for CommitDiff.pop index translation.

``data_deleted`` in a commit diff must report indices in the coordinate
space at the start of the commit, even when earlier pops in the same
commit have already shifted the live indices.
"""

import numpy as np
import pytest

import muller
from muller.util.exceptions import InvalidOperationError


@pytest.fixture
def ds(tmp_path):
    ds = muller.dataset(str(tmp_path / "pop_diff_ds"))
    ds.create_tensor("labels")
    ds.labels.extend([[0], [1], [2], [3], [4]])
    return ds


def _tensor_change(ds, id_1, id_2, **kwargs):
    return ds.diff(id_1, id_2, as_dict=True, **kwargs)["tensor"][1][0]["labels"]


def test_repeated_pop_records_distinct_deleted_indices(ds):
    first = ds.commit("base")

    ds.pop(1)
    ds.pop(1)  # removes original samples 1 and 2
    second = ds.commit("pop live index 1 twice")

    np.testing.assert_array_equal(ds.labels.numpy(), [[0], [3], [4]])

    change = _tensor_change(ds, first, second)
    assert change["data_deleted"] == {1, 2}
    assert len(change["data_deleted_ids"]) == 2
    assert change["data_added"] == [3, 3]

    deleted_values = _tensor_change(ds, first, second, show_value=True)["data_deleted_values"]
    assert sorted(int(np.asarray(value).reshape(-1)[0]) for value in deleted_values) == [1, 2]


def test_pop_added_and_base_in_same_commit(ds):
    first = ds.commit("base")

    ds.labels.append([5])
    ds.labels.append([6])
    ds.pop(0)  # pre-existing sample 0; live layout becomes [1, 2, 3, 4, 5, 6]
    ds.pop(4)  # live 4 is the sample [5] appended in this commit
    second = ds.commit("mixed pops")

    np.testing.assert_array_equal(ds.labels.numpy(), [[1], [2], [3], [4], [6]])

    change = _tensor_change(ds, first, second)
    assert change["data_deleted"] == {0}
    assert len(change["data_deleted_ids"]) == 1
    # Net added block in live coordinates: one surviving appended sample.
    assert change["data_added"] == [4, 5]

    add_values = _tensor_change(ds, first, second, show_value=True)["add_value"]
    assert [int(np.asarray(value).reshape(-1)[0]) for value in add_values] == [6]


def test_pops_in_consecutive_commits_use_commit_start_coordinates(ds):
    first = ds.commit("base")

    ds.pop(1)  # removes original sample 1
    second = ds.commit("pop 1")

    ds.pop(1)  # live layout is [0, 2, 3, 4]; removes original sample 2
    third = ds.commit("pop 1 again")

    np.testing.assert_array_equal(ds.labels.numpy(), [[0], [3], [4]])

    # Each commit reports deletions relative to its own starting layout.
    assert _tensor_change(ds, first, second)["data_deleted"] == {1}
    assert _tensor_change(ds, second, third)["data_deleted"] == {1}


def test_pop_on_view_raises_and_leaves_diff_untouched(ds):
    ds.commit("base")

    sliced = ds[1:3]
    with pytest.raises(InvalidOperationError):
        sliced.pop(0)

    filtered = ds.filter("labels == 1")
    with pytest.raises(InvalidOperationError):
        filtered.pop(0)

    assert len(ds) == 5
    commit_diff = ds.labels.chunk_engine.commit_diff
    assert list(commit_diff.data_deleted) == []
    assert commit_diff.num_samples_added == 0

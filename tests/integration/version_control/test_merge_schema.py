# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

"""Regression tests for schema compatibility validation before merge.

The merge copies raw chunks between branches, so merging tensors whose
dtype/htype/compression diverged after the fork point used to silently corrupt
data. ``check_common_tensor_mismatches`` must reject such merges with a clear
``MergeMismatchError`` instead.
"""

import numpy as np
import pytest

import muller
from muller.util.exceptions import MergeMismatchError

from tests.constants import TEST_MERGE_SCHEMA_PATH


def _fresh_dataset():
    return muller.dataset(path=TEST_MERGE_SCHEMA_PATH, overwrite=True)


def test_merge_dtype_mismatch_raises():
    """Branches that materialized different dtypes for the same tensor cannot be merged."""
    ds = _fresh_dataset()
    ds.create_tensor("readings")
    ds.commit("create tensor without dtype")

    ds.checkout("dev", create=True)
    ds.readings.append(np.array([1.5], dtype=np.float32))
    ds.commit("float32 data on dev")

    ds.checkout("main")
    ds.readings.append(np.array([1], dtype=np.int64))
    ds.commit("int64 data on main")

    with pytest.raises(MergeMismatchError) as excinfo:
        ds.merge("dev", append_resolution="both", update_resolution="theirs", pop_resolution="theirs")
    assert "dtype" in str(excinfo.value)
    assert "readings" in str(excinfo.value)


def test_merge_dtype_none_is_compatible():
    """A tensor that never received data (dtype=None) merges with any dtype."""
    ds = _fresh_dataset()
    ds.create_tensor("readings")
    ds.commit("create tensor without dtype")

    ds.checkout("dev", create=True)
    ds.readings.append(np.array([1], dtype=np.int64))
    ds.readings.append(np.array([2], dtype=np.int64))
    ds.commit("int64 data on dev")

    ds.checkout("main")
    ds.merge("dev")

    assert len(ds.readings) == 2
    np.testing.assert_array_equal(ds.readings.numpy().reshape(-1), np.array([1, 2], dtype=np.int64))


def test_merge_same_dtype_still_allowed():
    """The new dtype check must not reject merges of branches with matching dtypes."""
    ds = _fresh_dataset()
    ds.create_tensor("readings")
    ds.readings.append(np.array([0], dtype=np.int64))
    ds.commit("base int64 sample")

    ds.checkout("dev", create=True)
    ds.readings.append(np.array([1], dtype=np.int64))
    ds.commit("append on dev")

    ds.checkout("main")
    ds.readings.append(np.array([2], dtype=np.int64))
    ds.commit("append on main")

    ds.merge("dev", append_resolution="both", update_resolution="theirs", pop_resolution="theirs")
    values = sorted(int(v) for v in ds.readings.numpy().reshape(-1))
    assert values == [0, 1, 2]

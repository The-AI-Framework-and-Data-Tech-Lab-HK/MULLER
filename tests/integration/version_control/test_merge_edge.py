# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

"""Merge edge cases: schema conflicts beyond dtype, and UUID integrity.

``test_merge_schema.py`` covers dtype divergence. This module covers the other
fields checked by ``check_common_tensor_mismatches`` (htype / compression) and
asserts that merges which append or pop samples do not drop or duplicate
dataset UUIDs.
"""

import numpy as np
import pytest

import muller
from muller.constants import DATASET_UUID_NAME
from muller.util.exceptions import MergeMismatchError

from tests.constants import SAMPLE_FILES, TEST_MERGE_EDGE_PATH


def _fresh_dataset():
    return muller.dataset(path=TEST_MERGE_EDGE_PATH, overwrite=True)


def _uuid_list(ds):
    commit_id = ds.version_state["commit_id"]
    return ds.get_tensor_uuids(DATASET_UUID_NAME, commit_id)


class TestMergeSchemaConflicts:
    def test_htype_mismatch_raises(self):
        ds = _fresh_dataset()
        ds.commit("empty base")

        ds.checkout("dev", create=True)
        ds.create_tensor("payload", htype="text")
        ds.payload.append("hello")
        ds.commit("text on dev")

        ds.checkout("main")
        ds.create_tensor("payload", htype="generic", dtype="int64")
        ds.payload.append(np.array([1], dtype=np.int64))
        ds.commit("generic on main")

        with pytest.raises(MergeMismatchError) as excinfo:
            ds.merge("dev", append_resolution="both", update_resolution="theirs", pop_resolution="theirs")
        assert "htype" in str(excinfo.value)
        assert "payload" in str(excinfo.value)

    def test_sample_compression_mismatch_raises(self):
        ds = _fresh_dataset()
        ds.commit("empty base")

        ds.checkout("dev", create=True)
        ds.create_tensor("images", htype="image", sample_compression="jpeg")
        ds.images.append(muller.read(SAMPLE_FILES["jpg_1"]))
        ds.commit("jpeg on dev")

        ds.checkout("main")
        ds.create_tensor("images", htype="image", sample_compression="png")
        ds.images.append(muller.read(SAMPLE_FILES["jpg_1"]))
        ds.commit("png on main")

        with pytest.raises(MergeMismatchError) as excinfo:
            ds.merge("dev", append_resolution="both", update_resolution="theirs", pop_resolution="theirs")
        assert "sample_compression" in str(excinfo.value)
        assert "images" in str(excinfo.value)

    def test_chunk_compression_mismatch_raises(self):
        ds = _fresh_dataset()
        ds.commit("empty base")

        ds.checkout("dev", create=True)
        ds.create_tensor("readings", dtype="float64", chunk_compression="lz4")
        ds.readings.append(np.array([1.0], dtype=np.float64))
        ds.commit("lz4 on dev")

        ds.checkout("main")
        ds.create_tensor("readings", dtype="float64", chunk_compression=None)
        ds.readings.append(np.array([2.0], dtype=np.float64))
        ds.commit("uncompressed on main")

        with pytest.raises(MergeMismatchError) as excinfo:
            ds.merge("dev", append_resolution="both", update_resolution="theirs", pop_resolution="theirs")
        assert "chunk_compression" in str(excinfo.value)
        assert "readings" in str(excinfo.value)


class TestMergeUuidIntegrity:
    def test_merge_append_both_preserves_unique_uuids(self):
        """Both-branch appends must keep one UUID per sample, all unique."""
        ds = _fresh_dataset()
        ds.create_tensor("labels", dtype="int64")
        ds.labels.extend([np.array([i], dtype=np.int64) for i in range(5)])
        ds.commit("base")
        base_uuids = set(_uuid_list(ds))

        ds.checkout("dev", create=True)
        ds.labels.extend([np.array([10], dtype=np.int64), np.array([11], dtype=np.int64)])
        ds.commit("append on dev")

        ds.checkout("main")
        ds.labels.extend([np.array([20], dtype=np.int64), np.array([21], dtype=np.int64), np.array([22], dtype=np.int64)])
        ds.commit("append on main")

        ds.merge("dev", append_resolution="both", update_resolution="theirs", pop_resolution="theirs")

        uuids = _uuid_list(ds)
        assert len(uuids) == len(ds.labels) == 10
        assert len(set(uuids)) == 10
        assert base_uuids.issubset(set(uuids))
        assert len(ds[DATASET_UUID_NAME]) == len(ds.labels)

    def test_merge_with_pops_keeps_uuid_aligned_with_rows(self):
        """After pops on both sides, surviving rows still have distinct UUIDs."""
        ds = _fresh_dataset()
        ds.create_tensor("labels", dtype="int64")
        ds.labels.extend([np.array([i], dtype=np.int64) for i in range(6)])
        ds.commit("base")

        ds.checkout("dev", create=True)
        ds.pop([1])
        ds.labels.append(np.array([100], dtype=np.int64))
        ds.commit("pop+append on dev")

        ds.checkout("main")
        ds.pop([2])
        ds.labels.append(np.array([200], dtype=np.int64))
        ds.commit("pop+append on main")

        ds.merge(
            "dev",
            append_resolution="both",
            update_resolution="theirs",
            pop_resolution="both",
        )

        uuids = _uuid_list(ds)
        n = len(ds.labels)
        assert n == len(uuids) == len(ds[DATASET_UUID_NAME])
        assert len(set(uuids)) == n
        assert None not in uuids

# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

"""End-to-end tests for Dataset.create_uuid_index / load_uuid_index.

These APIs used to crash unconditionally because shard persistence relied on
the never-shipped ``cykhash_ext`` extension.
"""

import numpy as np
import pytest

import muller
from muller.constants import DATASET_UUID_NAME, FIRST_COMMIT_ID

from tests.constants import TEST_UUID_INDEX_PATH


def test_create_and_load_uuid_index_round_trip():
    ds = muller.dataset(path=TEST_UUID_INDEX_PATH, overwrite=True)
    ds.create_tensor("readings")
    with ds:
        ds.readings.extend([np.array([i]) for i in range(20)])

    ds.create_uuid_index()
    table = ds.load_uuid_index()

    uuids = ds.get_tensor_uuids(DATASET_UUID_NAME, ds.version_state["commit_id"])
    assert len(table) == len(uuids) == 20

    signed_keys = np.array(uuids, dtype=np.uint64).view(np.int64)
    for expected_index, key in enumerate(signed_keys):
        assert table.get(np.int64(key)) == expected_index


def test_create_uuid_index_requires_first_commit():
    ds = muller.dataset(path=TEST_UUID_INDEX_PATH, overwrite=True)
    ds.create_tensor("readings")
    ds.readings.append(np.array([1]))
    ds.commit("move off the first commit")
    assert ds.version_state["commit_id"] != FIRST_COMMIT_ID
    with pytest.raises(ValueError):
        ds.create_uuid_index()

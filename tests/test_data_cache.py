# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import time
import pytest

import muller
from muller.core.storage.data_cache import DataCache
from tests.constants import TEST_CACHE_PATH
from tests.utils import official_path, official_creds

CACHE = DataCache(capacity=5)


def test_dataset_cache(storage):
    """test cache to store dataset"""
    ds = muller.empty(official_path(storage, TEST_CACHE_PATH), creds=official_creds(storage), overwrite=True)
    with ds:
        ds.create_tensor("col1", htype="text")
        ds.create_tensor("col2", htype="text")
    with ds:
        for _ in range(100):
            ds.col1.append("data1")
            ds.col2.append("data2")

    def load_muller_dataset(storage, muller_path):
        ds = CACHE.get(muller_path)
        if ds is None:
            ds = muller.load(official_path(storage, muller_path), creds=official_creds(storage))
            CACHE.put(muller_path, ds)
        return ds

    first_load_start = time.time()
    ds = load_muller_dataset(storage, TEST_CACHE_PATH)
    first_load_end = time.time()
    assert len(ds) == 100

    second_load_start = time.time()
    ds = load_muller_dataset(storage, TEST_CACHE_PATH)
    second_load_end = time.time()
    assert len(ds) == 100

    assert second_load_end - second_load_start < first_load_end - first_load_start


if __name__ == '__main__':
    pytest.main(["-s", "test_data_cache.py"])

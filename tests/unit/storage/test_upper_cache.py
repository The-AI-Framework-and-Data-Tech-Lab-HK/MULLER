# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

"""Unit tests for the bounded upper caches of LRUCache.

``upper_cache`` (uuids / filter / filter_vectorized) and ``upper_cache_merge``
hold derived, recomputable data. They must stay bounded (LRU eviction) so that
long-lived storages don't accumulate uuid lists or merge records of old
commits indefinitely.
"""

import pickle

import pytest

from muller.constants import (
    FILTER_CACHE_SIZE,
    MERGE_RECORDS_CACHE_COMMITS,
    UUID_CACHE_COMMITS,
)
from muller.core.storage.lru_cache import BoundedLRUDict, LRUCache
from muller.core.storage.memory import MemoryProvider
from muller.core.version_control.operations.merge import RecordsCache


def _make_lru_cache() -> LRUCache:
    return LRUCache(MemoryProvider(), MemoryProvider(), cache_size=1024)


def _make_records(original_id: str, target_id: str = "t0", tensor_name: str = "abc") -> RecordsCache:
    return RecordsCache(
        target_commit_id=target_id,
        original_commit_id=original_id,
        tensor_name=tensor_name,
        app_ori_idx=[1],
        app_tar_idx=[2],
        delete_ori=[],
        delete_tar=[],
        original_id_to_index_map={"id1": 0},
        target_id_to_index_map={"id1": 1},
        updated_indexes=[(0, 1)],
        detect_conflicts=[],
    )


class TestBoundedLRUDict:
    def test_rejects_non_positive_maxsize(self):
        with pytest.raises(ValueError):
            BoundedLRUDict(maxsize=0)

    def test_evicts_least_recently_used_on_insert(self):
        d = BoundedLRUDict(maxsize=3)
        d["a"], d["b"], d["c"] = 1, 2, 3
        d["d"] = 4  # evicts "a" (the oldest), never the new entry
        assert "a" not in d
        assert list(d) == ["b", "c", "d"]

    def test_getitem_refreshes_recency(self):
        d = BoundedLRUDict(maxsize=3)
        d["a"], d["b"], d["c"] = 1, 2, 3
        assert d["a"] == 1  # "a" becomes most recently used
        d["d"] = 4  # now "b" is the oldest
        assert "a" in d and "b" not in d

    def test_get_refreshes_recency_and_returns_default(self):
        d = BoundedLRUDict(maxsize=2)
        d["a"], d["b"] = 1, 2
        assert d.get("missing") is None
        assert d.get("missing", "fallback") == "fallback"
        assert d.get("a") == 1
        d["c"] = 3
        assert "a" in d and "b" not in d

    def test_overwrite_refreshes_recency_without_eviction(self):
        d = BoundedLRUDict(maxsize=2)
        d["a"], d["b"] = 1, 2
        d["a"] = 10  # overwrite must not evict, and refreshes "a"
        assert dict(d) == {"a": 10, "b": 2}
        d["c"] = 3
        assert "a" in d and "b" not in d

    def test_contains_does_not_refresh_recency(self):
        d = BoundedLRUDict(maxsize=2)
        d["a"], d["b"] = 1, 2
        assert "a" in d  # membership check alone must not promote "a"
        d["c"] = 3
        assert "a" not in d


class TestUpperCacheBounds:
    def test_sub_caches_are_pre_created_and_bounded(self):
        cache = _make_lru_cache()
        for name, maxsize in (
            ("uuids", UUID_CACHE_COMMITS),
            ("filter", FILTER_CACHE_SIZE),
            ("filter_vectorized", FILTER_CACHE_SIZE),
        ):
            assert isinstance(cache.upper_cache[name], BoundedLRUDict)
            assert cache.upper_cache[name].maxsize == maxsize
        assert isinstance(cache.upper_cache_merge, BoundedLRUDict)
        assert cache.upper_cache_merge.maxsize == MERGE_RECORDS_CACHE_COMMITS

    def test_uuid_cache_evicts_oldest_commit(self):
        cache = _make_lru_cache()
        uuids = cache.upper_cache["uuids"]
        for i in range(UUID_CACHE_COMMITS + 1):
            uuids[f"commit_{i}"] = {"abc": [i] * 10}
        assert len(uuids) == UUID_CACHE_COMMITS
        assert "commit_0" not in uuids
        assert uuids[f"commit_{UUID_CACHE_COMMITS}"]["abc"] == [UUID_CACHE_COMMITS] * 10

    def test_clear_target_upper_cache_drops_tensor_entry(self):
        cache = _make_lru_cache()
        cache.upper_cache["uuids"]["commit_a"] = {"abc": [1, 2], "xyz": [3, 4]}
        cache.clear_target_upper_cache("uuids", "commit_a", "abc")
        assert "abc" not in cache.upper_cache["uuids"]["commit_a"]
        assert cache.upper_cache["uuids"]["commit_a"]["xyz"] == [3, 4]
        # unknown commit / tensor is a no-op
        cache.clear_target_upper_cache("uuids", "missing_commit", "abc")
        cache.clear_target_upper_cache("uuids", "commit_a", "missing_tensor")

    def test_merge_records_roundtrip(self):
        cache = _make_lru_cache()
        records = _make_records("ori_0")
        cache.add_records_cache_merge(records)
        fetched = cache.get_records_cache_merge("t0", "abc", "ori_0")
        assert fetched["updated_indexes"] == [(0, 1)]
        assert fetched["original_id_to_index_map"] == {"id1": 0}

    def test_merge_records_evicted_beyond_bound_and_miss_returns_empty(self):
        cache = _make_lru_cache()
        for i in range(MERGE_RECORDS_CACHE_COMMITS + 1):
            cache.add_records_cache_merge(_make_records(f"ori_{i}"))
        assert len(cache.upper_cache_merge) == MERGE_RECORDS_CACHE_COMMITS
        # the oldest original commit was evicted; merge falls back to recomputing
        assert cache.get_records_cache_merge("t0", "abc", "ori_0") == {}
        newest = f"ori_{MERGE_RECORDS_CACHE_COMMITS}"
        assert cache.get_records_cache_merge("t0", "abc", newest) != {}

    def test_merge_records_same_commit_pair_multiple_tensors_not_evicted(self):
        """All tensors of one merge share a single original-commit slot."""
        cache = _make_lru_cache()
        n_tensors = MERGE_RECORDS_CACHE_COMMITS * 5
        for i in range(n_tensors):
            cache.add_records_cache_merge(_make_records("ori_0", tensor_name=f"tensor_{i}"))
        for i in range(n_tensors):
            assert cache.get_records_cache_merge("t0", f"tensor_{i}", "ori_0") != {}

    def test_unpickled_cache_keeps_bounded_sub_caches(self):
        cache = _make_lru_cache()
        cache.upper_cache["uuids"]["commit_a"] = {"abc": [1]}
        restored = pickle.loads(pickle.dumps(cache))
        # contents are reset on unpickle, but the bounded structure must survive
        assert isinstance(restored.upper_cache["uuids"], BoundedLRUDict)
        assert isinstance(restored.upper_cache["filter"], BoundedLRUDict)
        assert isinstance(restored.upper_cache["filter_vectorized"], BoundedLRUDict)
        assert isinstance(restored.upper_cache_merge, BoundedLRUDict)
        assert len(restored.upper_cache["uuids"]) == 0


if __name__ == "__main__":
    pytest.main(["-s", __file__])

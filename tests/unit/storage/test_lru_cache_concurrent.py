# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

"""Concurrency tests for LRUCache and BoundedLRUDict.

``get_items_no_update`` is documented as thread-safe (read-only). Cache hits via
``__getitem__`` / ``get_item_from_cache`` tolerate concurrent LRU bookkeeping
with a try/except guard. These tests pin that contract so regressions surface
as failures rather than silent data races in filter prefetch paths.
"""

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from muller.core.storage.lru_cache import BoundedLRUDict, LRUCache
from muller.core.storage.memory import MemoryProvider


def _make_cache_with_payload(n_keys: int = 32) -> LRUCache:
    next_storage = MemoryProvider()
    for i in range(n_keys):
        next_storage[f"k{i}"] = f"value-{i}".encode()
    cache = LRUCache(MemoryProvider(), next_storage, cache_size=1024 * 1024)
    # Warm the cache so concurrent readers hit cache_storage / lru_sizes.
    for i in range(n_keys):
        _ = cache[f"k{i}"]
    return cache


class TestGetItemsNoUpdateConcurrent:
    def test_concurrent_readers_see_consistent_values(self):
        cache = _make_cache_with_payload(40)
        keys = {f"k{i}" for i in range(40)}
        expected = {f"k{i}": f"value-{i}".encode() for i in range(40)}
        errors = []

        def reader():
            try:
                got = cache.get_items_no_update(keys)
                assert got == expected
            except Exception as exc:  # noqa: BLE001 — collect for the main thread
                errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


class TestCacheHitConcurrent:
    def test_concurrent_getitem_on_warm_cache(self):
        cache = _make_cache_with_payload(20)
        errors = []
        results = []

        def reader(key: str):
            try:
                results.append((key, cache[key]))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(reader, f"k{i % 20}") for i in range(200)]
            for fut in as_completed(futures):
                fut.result()

        assert errors == []
        assert len(results) == 200
        for key, value in results:
            assert value == f"value-{key[1:]}".encode()


class TestBoundedLRUDictConcurrent:
    def test_concurrent_reads_after_fill(self):
        d = BoundedLRUDict(maxsize=64)
        for i in range(64):
            d[f"c{i}"] = i

        errors = []

        def reader():
            try:
                for i in range(64):
                    assert d[f"c{i}"] == i
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []

    def test_concurrent_writers_on_distinct_keys_do_not_crash(self):
        """Distinct-key inserts under contention must not raise.

        Final membership may be smaller than the write set because of LRU
        eviction and races on OrderedDict; we only require that every surviving
        key maps to the value that was written for it.
        """
        d = BoundedLRUDict(maxsize=32)
        n_keys = 64

        def writer(i: int):
            d[f"w{i}"] = i

        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(writer, range(n_keys)))

        assert 1 <= len(d) <= 32
        for key, value in list(d.items()):
            assert key.startswith("w")
            assert value == int(key[1:])


class TestUpperCacheConcurrent:
    def test_concurrent_uuid_cache_reads(self):
        cache = LRUCache(MemoryProvider(), MemoryProvider(), cache_size=1024)
        uuids = cache.upper_cache["uuids"]
        for i in range(4):
            uuids[f"commit_{i}"] = {"tensor": list(range(i * 10, i * 10 + 10))}

        errors = []

        def reader():
            try:
                for i in range(4):
                    assert uuids[f"commit_{i}"]["tensor"][0] == i * 10
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []

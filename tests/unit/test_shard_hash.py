# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

"""Tests for UUID shard table persistence (muller/core/dataset/uuid/shard_hash.py).

Persistence used to depend on a ``cykhash_ext`` C extension that was never
shipped, and ``HashBuilder.put()`` raised ``TypeError`` even on success. These
tests pin the pure-Python replacement.
"""

import numpy as np
import pytest

from muller.core.dataset.uuid.shard_hash import HashBuilder, divide_to_shard, load_all_shards
from muller.util.exceptions import CykhashLoadError, FileAtPathException


def test_put_and_get(tmp_path):
    builder = HashBuilder(shard_dir=str(tmp_path / "shards"))
    builder.put(np.int64(7), np.int64(42))
    assert builder.get(np.int64(7)) == 42
    assert builder.size == 1


def test_put_rejects_wrong_types(tmp_path):
    builder = HashBuilder(shard_dir=str(tmp_path / "shards"))
    with pytest.raises(TypeError):
        builder.put(7, 42)  # plain ints are not np.int64
    with pytest.raises(TypeError):
        builder.put(np.int64(7), "42")
    assert builder.size == 0


def test_put_all_populates_table(tmp_path):
    builder = HashBuilder(shard_dir=str(tmp_path / "shards"))
    keys = np.arange(50, dtype=np.int64)
    values = keys * 3
    builder.put_all(keys, values)
    assert builder.size == 50
    assert builder.get(np.int64(10)) == 30


def test_save_and_load_round_trip(tmp_path):
    shard_dir = str(tmp_path / "shards")
    builder = HashBuilder(shard_dir=shard_dir, shard_idx=0)
    keys = np.arange(100, dtype=np.int64)
    builder.put_all(keys, keys * 10)
    builder.save_table(overwrite=True)

    loader = HashBuilder(shard_dir=shard_dir)
    loader.load_table()
    assert loader.size == 100
    for k in (0, 5, 99):
        assert loader.get(np.int64(k)) == k * 10


def test_save_without_overwrite_raises(tmp_path):
    shard_dir = str(tmp_path / "shards")
    builder = HashBuilder(shard_dir=shard_dir, shard_idx=0)
    builder.put(np.int64(1), np.int64(2))
    builder.save_table(overwrite=True)
    with pytest.raises(FileAtPathException):
        builder.save_table(overwrite=False)


def test_load_from_empty_dir_raises(tmp_path):
    loader = HashBuilder(shard_dir=str(tmp_path / "shards"))
    with pytest.raises(FileNotFoundError):
        loader.load_table()


def test_load_corrupt_shard_raises(tmp_path):
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    (shard_dir / "shard_0.bin").write_bytes(b"this is not a npy payload")
    loader = HashBuilder(shard_dir=str(shard_dir))
    with pytest.raises(CykhashLoadError):
        loader.load_table()


def test_load_wrong_layout_raises(tmp_path):
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    with open(shard_dir / "shard_0.bin", "wb") as f:
        np.save(f, np.arange(10, dtype=np.int64))  # 1-D, not (2, n)
    loader = HashBuilder(shard_dir=str(shard_dir))
    with pytest.raises(CykhashLoadError):
        loader.load_table()


def test_divide_to_shard_and_load_all_shards(tmp_path):
    # Includes uuids above 2**63 to exercise the uint64 -> int64 view.
    uuids = [2**64 - 1, 2**63, 5, 123456789, 42, 7, 99, 1000, 2**40, 13, 17, 19, 23]
    divide_to_shard(path=str(tmp_path), uuids=uuids, num_shards=8)

    shard_files = sorted((tmp_path / "shards").glob("shard_*.bin"))
    assert len(shard_files) == 8

    table = load_all_shards(path=str(tmp_path))
    assert len(table) == len(uuids)
    signed_keys = np.array(uuids, dtype=np.uint64).view(np.int64)
    for index, key in enumerate(signed_keys):
        assert table.get(np.int64(key)) == index


def test_divide_to_shard_empty_uuids(tmp_path):
    divide_to_shard(path=str(tmp_path), uuids=[], num_shards=4)
    table = load_all_shards(path=str(tmp_path))
    assert len(table) == 0

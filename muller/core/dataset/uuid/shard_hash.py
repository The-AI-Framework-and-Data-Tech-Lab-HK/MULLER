# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import glob
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import numpy as np
from cykhash import Int64toInt64Map, Int64toInt64Map_from_buffers

from muller.util.exceptions import FileAtPathException, CykhashPutError, CykhashGetError, CykhashLoadError

# Shard tables are persisted as ``.npy`` payloads holding a (2, n) int64 array
# (row 0: keys, row 1: values). This pure-Python serialization replaces the
# former ``cykhash_ext`` C extension, which was never shipped with the
# repository (its save_map/load_map made create/load_uuid_index crash).
_SHARD_FILE_GLOB = "shard_*.bin"


def process_shard(shard_index: int, shard_dir: str, sharded_uuid: np.ndarray, sharded_idx: np.ndarray):
    """Handle single process: build hashmap and save table to disk."""
    hashmap = HashBuilder(shard_dir=shard_dir, shard_idx=shard_index)
    hashmap.put_all(sharded_uuid, sharded_idx)
    hashmap.save_table(overwrite=True)
    hashmap.clear()


def divide_to_shard(path: str, uuids: List[int], num_shards: int = 8, shard_dir: str = 'shards'):
    """Sharded uuids and indexes to the disk by multi-thread."""
    full_shard_dir = os.path.join(path, shard_dir)
    Path(full_shard_dir).mkdir(parents=True, exist_ok=True)

    uuid_arr = np.array(uuids, dtype=np.uint64).view(np.int64)

    shard_size = len(uuid_arr) // num_shards
    remainder = len(uuid_arr) % num_shards

    with ThreadPoolExecutor(max_workers=num_shards) as executor:
        futures = []
        start = 0

        for shard_index in range(num_shards):
            # Each subarray has a size of shard_size, and the first remainder subarrays contain one additional element.
            end = start + shard_size + (1 if shard_index < remainder else 0)
            sharded_uuid = uuid_arr[start:end]

            future = executor.submit(
                process_shard,
                shard_index,
                full_shard_dir,
                sharded_uuid,
                np.arange(start, end, dtype=np.int64)
            )
            futures.append(future)

            start = end

        for future in futures:
            future.result()


def load_all_shards(path:str, shard_dir:str = 'shards') -> Int64toInt64Map:
    """Function to load all shards from the given dir."""
    hashmap = HashBuilder(shard_dir=f"{path}/{shard_dir}")
    hashmap.load_table()
    return hashmap.table


class HashBuilder:
    def __init__(self, shard_dir: str, shard_idx: int = 0):
        self.size = 0
        self.shard_dir = shard_dir
        self.shard_idx = shard_idx
        self.table = Int64toInt64Map()
        if not os.path.isdir(shard_dir):
            os.makedirs(shard_dir)

    def put_all(self, keys_arr, values_arr):
        """Bulk-load key/value arrays into the (still empty) table."""
        if not self.table:
            try:
                self.table = Int64toInt64Map_from_buffers(keys_arr, values_arr)
                self.size += len(keys_arr)
            except (TypeError, ValueError):
                # Inputs that are not contiguous int64 buffers fall back to
                # element-wise insertion (which validates each pair).
                for key, value in zip(keys_arr, values_arr):
                    self.put(key, value)


    def put(self, key, value):
        """Put a single key/value pair into the table."""
        if not (isinstance(key, np.int64) and isinstance(value, np.int64)):
            raise TypeError(f"Expected key and value to be of type np.int64, but got {type(key)} and {type(value)}")
        try:
            self.table.cput(key, value)
        except Exception as e:
            raise CykhashPutError from e
        self.size += 1


    def get(self, key):
        """Get a shard."""
        if not isinstance(key, np.int64):
            raise TypeError(f"Expected key to be of type np.int64, but got {type(key)}")
        if not self.table:
            raise RuntimeError("Hash table is not initialized.")
        try:
            value = self.table.cget(key)
            return value
        except Exception as e:
            raise CykhashGetError(f"An error occurred while retrieving the key: {key}") from e


    def save_table(self, overwrite=True):
        """Persist the table to ``shard_<idx>.bin`` inside the shard directory."""
        shard_path = os.path.join(self.shard_dir, f"shard_{self.shard_idx}.bin")
        if os.path.exists(shard_path) and not overwrite:
            raise FileAtPathException(shard_path)
        pairs = np.empty((2, len(self.table)), dtype=np.int64)
        for i, (key, value) in enumerate(self.table.items()):
            pairs[0, i] = key
            pairs[1, i] = value
        # np.save gets a file object so the ".bin" name is kept as-is
        # (given a plain path it would append ".npy").
        with open(shard_path, "wb") as f:
            np.save(f, pairs, allow_pickle=False)


    def load_table(self):
        """Load and merge every shard file found in the shard directory."""
        shard_paths = sorted(glob.glob(os.path.join(self.shard_dir, _SHARD_FILE_GLOB)))
        if not shard_paths:
            raise FileNotFoundError(f"No shard files found in: {self.shard_dir}")

        self.table.clear()
        for shard_path in shard_paths:
            try:
                with open(shard_path, "rb") as f:
                    pairs = np.load(f, allow_pickle=False)
            except (OSError, ValueError) as e:
                raise CykhashLoadError(f"Unable to read shard file: {shard_path}") from e
            if pairs.ndim != 2 or pairs.shape[0] != 2 or pairs.dtype != np.int64:
                raise CykhashLoadError(
                    f"Shard file {shard_path} has an unexpected layout: "
                    f"shape={pairs.shape}, dtype={pairs.dtype} (expected (2, n) int64)."
                )
            self.table.update(Int64toInt64Map_from_buffers(pairs[0], pairs[1]))

        self.size = len(self.table)
        logging.info(f"Table loaded from {self.shard_dir}, size: {self.size}")


    def clear(self):
        """Clear table."""
        self.table.clear()

# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import numpy as np
from cykhash import Int64toInt64Map, Int64toInt64Map_from_buffers

# from muller.util.cykhash import cykhash_ext
from muller.util.exceptions import FileAtPathException, CykhashPutError, CykhashGetError, CykhashLoadError


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
        """Put all shards."""
        if not self.table:
            try:
                self.table = Int64toInt64Map_from_buffers(keys_arr, values_arr)
                self.size += len(keys_arr)
            except Exception:
                for key, value in zip(keys_arr, values_arr):
                    self.put(key, value)


    def put(self, key, value):
        """Put a shard."""
        if isinstance(key, np.int64) and isinstance(value, np.int64):
            try:
                self.table.cput(key, value)
            except Exception as e:
                raise CykhashPutError from e
            self.size += 1
        raise TypeError(f"Expected key and value to be of type np.int64, but got {type(key)} and {type(value)}")


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
        """Save table."""
        shard_path = f"{self.shard_dir}/shard_{self.shard_idx}.bin"
        if os.path.exists(shard_path) and not overwrite:
            raise FileAtPathException(shard_path)
        cykhash_ext.save_map(self.table, shard_path)


    def load_table(self):
        """Load table."""
        if not os.path.exists(self.shard_dir):
            raise FileNotFoundError(f"Shard file not found: {self.shard_dir}")

        self.table.clear()
        try:
            cykhash_ext.load_map(self.shard_dir, self.table)
        except Exception as e:
            raise CykhashLoadError from e

        self.size = len(self.table)
        logging.info(f"Table loaded from {self.shard_dir}, size: {self.size}")


    def clear(self):
        """Clear table."""
        self.table.clear()

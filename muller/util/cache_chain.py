# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/util/cache_chain.py
#
# Modifications Copyright (c) 2026 Bingyu Liu

import os
from typing import List, Optional
from uuid import uuid1

from muller.constants import LOCAL_CACHE_PREFIX
from muller.core.storage.local import LocalProvider
from muller.core.storage.memory import MemoryProvider
from muller.core.storage.provider import StorageProvider
from muller.util.exceptions import ProviderSizeListMismatch, ProviderListEmptyError
from ..core.storage.lru_cache import LRUCache


def get_cache_chain(storage_list: List[StorageProvider], size_list: List[int]):
    if not storage_list:
        raise ProviderListEmptyError
    if len(storage_list) <= 1:
        return storage_list[0]
    if len(size_list) + 1 != len(storage_list):
        raise ProviderSizeListMismatch
    store = storage_list[-1]
    for size, cache in zip(reversed(size_list), reversed(storage_list[:-1])):
        store = LRUCache(cache, store, size)
    return store


def generate_chain(
    base_storage: StorageProvider,
    memory_cache_size: int,
    local_cache_size: int,
    path: Optional[str] = None,
) -> StorageProvider:
    if path:
        cached_dataset_name = path.replace("://", "_")
        cached_dataset_name = cached_dataset_name.replace("/", "_")
        cached_dataset_name = cached_dataset_name.replace("\\", "_")
    else:
        cached_dataset_name = str(uuid1())

    storage_list: List[StorageProvider] = []
    size_list: List[int] = []

    # Always have a memory cache prefix. Required for support for HubMemoryObjects.
    storage_list.append(MemoryProvider(f"cache/{cached_dataset_name}"))
    size_list.append(memory_cache_size)

    if local_cache_size > 0:
        local_cache_prefix = os.getenv("LOCAL_CACHE_PREFIX", default=LOCAL_CACHE_PREFIX)
        storage_list.append(
            LocalProvider(f"{local_cache_prefix}/{cached_dataset_name}")
        )
        size_list.append(local_cache_size)
    storage_list.append(base_storage)
    return get_cache_chain(storage_list, size_list)

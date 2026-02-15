# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/util/storage.py
#
# Modifications Copyright (c) 2026 Bingyu Liu

import os
from typing import Optional, Union

from muller.constants import MB
from muller.core.storage.huawei_obs import OBSProvider
from muller.core.storage.local import LocalProvider
from muller.core.storage.memory import MemoryProvider
from muller.core.storage.roma import RomaProvider
from muller.core.storage.s3 import S3Provider
from muller.core.storage.cache_chain import generate_chain


def storage_provider_from_path(
    path: str,
    creds: Optional[Union[dict, str]] = None,
    read_only: bool = False,
):
    """Construct a StorageProvider given a path.

        Arguments:
            path (str): The full path to the Dataset.
            creds (dict): A dictionary containing credentials used to access the dataset at the url.
                This takes precedence over credentials present in the environment.
                Only used when url is provided. Currently only works with Roma (Huawei Cloud) urls.
            read_only (bool): Opens dataset in read only mode if this is passed as True. Defaults to False.

        Returns:
            If given a path starting with mem:// returns the MemoryProvider.
            If given a valid local path, returns the LocalProvider.

        Raises:
            ValueError: If the given path is a local path to a file.
    """
    if creds is None:
        creds = {}
    if path.startswith("mem://"):
        storage = MemoryProvider(path)
    elif path.startswith("huawei-obs://"):
        storage = OBSProvider(endpoint=creds.get("endpoint"),
                              ak=creds.get("ak"),
                              sk=creds.get("sk"),
                              bucket_name=creds.get("bucket_name"),
                              root=path[len("huawei-obs://"):])
        storage.creds_used = "PLATFORM"
    elif path.startswith("roma://"):
        storage = RomaProvider(bucket_name=creds.get("bucket_name"),
                               region=creds.get("region"),
                               app_token=creds.get("app_token"),
                               vendor=creds.get("vendor"),
                               root=path[len("roma://"):])
        storage.creds_used = "PLATFORM"
    elif path.startswith("s3://"):
        storage = S3Provider(endpoint=creds.get("endpoint"),
                             ak=creds.get("ak"),
                             sk=creds.get("sk"),
                             bucket_name=creds.get("bucket_name"),
                             root=path[len("s3://"):])
        storage.creds_used = "PLATFORM"
    else:
        if not os.path.exists(path) or os.path.isdir(path):
            storage = LocalProvider(path)
        else:
            raise ValueError(
                f"Local path {path} must be a path to a local directory"
            )

    if read_only:
        storage.enable_readonly()
    return storage


def get_storage_and_cache_chain(path, read_only, creds, memory_cache_size, local_cache_size):
    """
    Returns storage provider and cache chain for a given path, according to arguments passed.

    Args:
        path (str): The full path to the Dataset.
        creds (dict): A dictionary containing credentials used to access the dataset at the url.
            This takes precedence over credentials present in the environment. Only used when url is provided.
            Currently only works with Roma (Huawei Cloud) urls.
        read_only (bool): Opens dataset in read only mode if this is passed as True. Defaults to False.
        memory_cache_size (int): The size of the in-memory cache to use.
        local_cache_size (int): The size of the local cache to use.
    Returns:
        A tuple of the storage provider and the storage chain.
    """

    storage = storage_provider_from_path(
        path=path,
        creds=creds,
        read_only=read_only
    )
    memory_cache_size_bytes = memory_cache_size * MB
    local_cache_size_bytes = local_cache_size * MB
    storage_chain = generate_chain(
        storage, memory_cache_size_bytes, local_cache_size_bytes, path
    )
    if storage.read_only:
        storage_chain.enable_readonly()
    return storage, storage_chain


def get_local_storage_path(path: str, prefix: str):
    """Obtain the local storage path"""
    local_cache_name = path.replace("://", "_")
    local_cache_name = local_cache_name.replace("/", "_")
    local_cache_name = local_cache_name.replace("\\", "_")
    return os.path.join(prefix, local_cache_name)

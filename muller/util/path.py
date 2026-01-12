# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/util/path.py
#
# Modifications Copyright (c) 2026 Bingyu Liu

import os
import pathlib
import re
from typing import Optional, Union, Tuple, Dict, List
import posixpath

from .exceptions import InvalidDatasetNameException

CLOUD_DS_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]*$")
LOCAL_DS_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_ .-]*$")
_relpath_cache: Dict[str, str] = {}
_joinpath_cache: Dict[str, str] = {}
_filepath_cache: List[str] = []


def get_path_type(path: Optional[str]) -> str:
    if not isinstance(path, str):
        path = str(path)
    if path.startswith(("http://", "https://")):
        return "http"
    elif path.startswith("huawei-obs://"):
        return "obs"
    elif path.startswith("obs://"):
        return "huashan-obs"
    elif path.startswith("huashan:"):
        return "huashan-file"
    elif path.startswith("mep://"):
        return "mep"
    elif path.startswith("roma://"):
        return "roma"
    elif path.startswith("s3://"):
        return "s3"
    else:
        return "local"


def is_remote_path(path: str) -> bool:
    return get_path_type(path) != "local"


def verify_dataset_name(path):
    path_type = get_path_type(path)
    ds_name = os.path.split(path)[-1]
    match = True
    if path_type == "local":
        match = bool(LOCAL_DS_NAME_PATTERN.match(ds_name))
    elif "/queries/" not in path:
        match = bool(CLOUD_DS_NAME_PATTERN.match(ds_name))
    if not match:
        raise InvalidDatasetNameException(path_type)


def convert_pathlib_to_string_if_needed(path: Union[str, pathlib.Path]) -> str:
    if isinstance(path, pathlib.Path):
        path = str(path)
    return path


def get_path_from_storage(storage) -> str:
    """Extracts the underlying path from a given storage."""
    from ..core.storage.lru_cache import LRUCache
    from ..core.storage.local import StorageProvider

    if isinstance(storage, LRUCache):
        return get_path_from_storage(storage.next_storage)
    elif isinstance(storage, StorageProvider):
        if hasattr(storage, "hub_path"):
            return storage.hub_path  # type: ignore
        return storage.root
    else:
        raise ValueError("Invalid storage type.")


def process_dataset_path(path: Union[str, pathlib.Path]) -> Tuple[str, Optional[str]]:
    dataset_path, at, address = str(path).partition("@")
    if not address:
        address = None  # type: ignore
    return dataset_path, address


def relpath(path, start):
    """
    Wrapper around posixpath.relpath that caches results to avoid performance overhead
    """
    key = path + "::" + start
    if key not in _relpath_cache:
        if len(_relpath_cache) > 1000:
            # Simple way to keep the cache from growing too large without doing a full LRU cache.
            # There should not be that many files that we deal with, so likely we will never even hit this.
            _relpath_cache.clear()
        _relpath_cache[key] = posixpath.relpath(path, start)
    return _relpath_cache[key]


def joinpath(path, root):
    """
    Wrapper around posixpath.join that caches results to avoid performance overhead
    """
    if path not in _joinpath_cache:
        if len(_joinpath_cache) > 1000:
            _joinpath_cache.clear()
        _joinpath_cache[path] = posixpath.join(root, path)
    return _joinpath_cache[path]


def dirpath(full_path):
    """
    Wrapper around os.path.isdir that caches results to avoid performance overhead
    """
    if full_path not in _filepath_cache:
        if len(_filepath_cache) > 1000:
            _filepath_cache.clear()
        if os.path.isdir(full_path):
            return True
        else:
            _filepath_cache.append(full_path)
            return False
    else:
        return False

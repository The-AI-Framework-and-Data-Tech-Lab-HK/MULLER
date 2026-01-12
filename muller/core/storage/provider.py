# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/core/storage/provider.py
#
# Modifications Copyright (c) 2026 Bingyu Liu

import posixpath
import threading
from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from typing import Optional, Set, Sequence, Dict, Any

from muller.constants import BYTE_PADDING
from muller.util.exceptions import ReadOnlyModeError, InvalidBytesRequestedError
from muller.util.keys import get_dataset_lock_key

_STORAGES: Dict[str, "StorageProvider"] = {}


def storage_factory(cls, root: str = "", *args, **kwargs) -> "StorageProvider":
    if cls.__name__ == "MemoryProvider":
        return cls(root, *args, **kwargs)
    thread_id = threading.get_ident()
    try:
        return _STORAGES[f"{thread_id}_{root}_{args}_{kwargs}"]
    except Exception:
        storage = cls(*args, **kwargs)
        _STORAGES[f"{thread_id}_{root}_{args}_{kwargs}"] = storage
        return storage


class StorageProvider(ABC, MutableMapping):
    autoflush = False
    read_only = False
    root = ""
    _is_hub_path = False

    @abstractmethod
    def __getitem__(self, key: str):
        """Gets the object present at the path within the given byte range.

        Args:
            key (str): The path relative to the root of the provider.

        Returns:
            bytes: The bytes of the object present at the path.

        Raises:
            KeyError: If an object is not found at the path.
        """

    @abstractmethod
    def __setitem__(self, key: str, value: bytes):
        """Sets the object present at the path with the value

        Args:
            path (str): the path relative to the root of the provider.
            value (bytes): the value to be assigned at the path.
        """

    @abstractmethod
    def __delitem__(self, path: str):
        """Function to delete the object present at the path.

        Args:
            path (str): the path to the object relative to the root of the provider.

        Raises:
            KeyError: an object is not found at the path.
        """

    @abstractmethod
    def __len__(self):
        """Function to returns the number of files present inside the root of the provider.
        """

    @abstractmethod
    def __iter__(self):
        """Function to generator function that iterates over the keys of the provider.
        """

    @staticmethod
    def _assert_byte_indexes(start_byte, end_byte):
        start_byte = start_byte or 0
        if start_byte < 0:
            raise InvalidBytesRequestedError()
        if end_byte is not None and (start_byte > end_byte or end_byte < 0):
            raise InvalidBytesRequestedError()

    @abstractmethod
    def get_items(self, keys: Set[str], ignore_key_error: bool = False) -> Dict[str, Any]:
        pass

    @abstractmethod
    def set_items(self, contents: Dict[str, Any]):
        """contents: {key: value} to set."""
        pass

    @abstractmethod
    def del_items(self, keys: Set[str]):
        """Multithreading get_multiple_files, added during fixing multithread bugs with LRU_Cache, with a different name
         to avoid conflicts with other code.
         by zhouzhenyu at 20240929
        """
        pass

    @abstractmethod
    def clear(self, prefix=""):
        """Delete the contents of the provider."""

    @abstractmethod
    def _all_keys(self) -> Set[str]:
        """Generator function that iterates over the keys of the provider.

        Returns:
            set: set of all keys present at the root of the provider.
        """

    def get_bytes(self,
                  path: str,
                  start_byte: Optional[int] = None,
                  end_byte: Optional[int] = None):
        self._assert_byte_indexes(start_byte, end_byte)
        return self[path][start_byte:end_byte]

    def flush(self):
        """Only needs to be implemented for caches. Flushes the data to the next storage provider.
        Should be a no op for Base Storage Providers like local, s3, azure, gcs, etc.
        """
        self.check_readonly()

    def maybe_flush(self):
        """Flush cache if autoflush has been enabled.
        Called at the end of methods which write data, to ensure consistency as a default.
        """
        if self.autoflush:
            self.flush()

    def set_bytes(self,
                  path: str,
                  value: bytes,
                  start_byte: Optional[int] = None,
                  overwrite: Optional[bool] = False):
        self.check_readonly()
        start_byte = start_byte or 0
        end_byte = start_byte + len(value)
        self._assert_byte_indexes(start_byte, end_byte)
        if path in self and not overwrite:
            current_value = bytearray(self[path])
            if end_byte > len(current_value):
                current_value = current_value.ljust(end_byte, BYTE_PADDING)
            current_value[start_byte:end_byte] = value
            self[path] = current_value
        else:
            if start_byte != 0:
                value = value.rjust(end_byte, BYTE_PADDING)
            self[path] = value

    def copy(self):
        """Returns a copy of the provider.

        Returns:
            StorageProvider: A copy of the provider.
        """
        cls = self.__class__
        new_provider = cls.__new__(cls)
        new_provider.__setstate__(self.__getstate__())
        return new_provider

    def enable_readonly(self):
        """Enables read-only mode for the provider."""
        self.read_only = True

    def check_readonly(self):
        """Raises an exception if the provider is in read-only mode."""
        if self.read_only:
            raise ReadOnlyModeError()

    def disable_readonly(self):
        """Disables read-only mode for the provider."""
        self.read_only = False

    def empty(self) -> bool:
        lock_key = get_dataset_lock_key()
        files_partial_count = self.files_partial_count()
        return files_partial_count - int(lock_key in self) <= 0

    def files_partial_count(self) -> int:
        """Fetch the number of files return by _all_keys()"""
        return len(self._all_keys())

    def delete_multiple(self, paths: Sequence[str]):
        for path in paths:
            del self[path]

    def get_presigned_url(self, key: str) -> str:
        return posixpath.join(self.root, key)

    def get_object_size(self, key: str) -> int:
        raise NotImplementedError()

    def get_items_old(self, keys):
        """原来在dataset._load_tensor_metas里调用了这个函数，修改它会导致一些愚蠢的bug, 暂时留着."""
        for key in keys:
            try:
                yield key, self[key]
            except KeyError:
                yield key, KeyError(key)

    def copy_files(self, src_list, dest_list):
        pass

    def rename(self, path: str):
        pass

# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/core/storage/memory.py
#
# Modifications Copyright (c) 2026 Bingyu Liu

import os
from typing import Any, Dict, Union
from typing import Set

from muller.core.storage.muller_memory_object import MULLERMemoryObject
from muller.core.storage.provider import StorageProvider
from muller.core.storage.util import _get_nbytes


class MemoryProvider(StorageProvider):
    """Provider class for using the memory."""

    def __init__(self, root: str = ""):
        self.dict: Dict[str, Any] = {}
        self.root = root

    def __getitem__(self, key: str):
        """Gets the object present at the path within the given byte range.

        Example:

            >>> memory_provider = MemoryProvider("xyz")
            >>> my_data = memory_provider["abc.txt"]

        Args:
            key (str): The path relative to the root of the provider.

        Returns:
            bytes: The bytes of the object present at the path.

        Raises:
            KeyError: If an object is not found at the path.
        """
        return self.dict[key]

    def __setitem__(
            self,
            key: str,
            value: Union[bytes, MULLERMemoryObject, dict],
    ):
        """Sets the object present at the path with the value

        Example:

            >>> memory_provider = MemoryProvider("xyz")
            >>> memory_provider["abc.txt"] = b"abcd"

        Args:
            key (str): the path relative to the root of the provider.
            value (bytes): the value to be assigned at the path.

        Raises:
            ReadOnlyError: If the provider is in read-only mode.
        """
        self.check_readonly()
        self.dict[key] = value

    def __iter__(self):
        """Generator function that iterates over the keys of the provider.

        Example:

            >>> memory_provider = MemoryProvider("xyz")
            >>> for my_data in memory_provider:
            ...    pass

        Yields:
            str: the path of the object that it is iterating over, relative to the root of the provider.
        """
        yield from self.dict

    def __delitem__(self, key: str):
        """Delete the object present at the path.

        Example:

            >>> memory_provider = MemoryProvider("xyz")
            >>> del memory_provider["abc.txt"]

        Args:
            key (str): the path to the object relative to the root of the provider.

        Raises:
            KeyError: If an object is not found at the path.
            ReadOnlyError: If the provider is in read-only mode.
        """
        self.check_readonly()
        del self.dict[key]

    def __len__(self):
        """Returns the number of files present inside the root of the provider.

        Example:

            >>> memory_provider = MemoryProvider("xyz")
            >>> len(memory_provider)

        Returns:
            int: the number of files present inside the root.
        """
        return len(self.dict)

    def __getstate__(self) -> str:
        """Does NOT save the in memory data in state."""
        return self.root

    def __setstate__(self, state: str):
        self.__init__(root=state)  # type: ignore

    def clear(self, prefix=""):
        """Clears the provider."""
        self.check_readonly()
        if prefix:
            self.dict = {k: v for k, v in self.dict.items() if not k.startswith(prefix)}
        else:
            self.dict = {}

    def get_object_size(self, key: str) -> int:
        return _get_nbytes(self[key])

    def subdir(self, key: str, read_only: bool = False):
        sd = self.__class__(os.path.join(self.root, key))
        sd.read_only = read_only
        return sd

    def get_items(self, keys: Set[str], ignore_key_error: bool = False):
        # Sherry: To improve the remote loading of tensor_meta.json files
        content_dict = {}
        for key in keys:
            if ignore_key_error:
                try:
                    val = self[key]
                    content_dict.update({key: val})
                except KeyError:
                    pass
            else:
                val = self[key]
                content_dict.update({key: val})
        return content_dict

    def set_items(self, contents: Dict[str, Any]):
        # Sherry: To improve the remote uploading of multiple meta files
        self.check_readonly()
        for path, value in contents.items():
            self.dict[path] = value

    def del_items(self, keys: Set[str]):
        for key in keys:
            del self[key]

    def _all_keys(self):
        """Lists all the objects present at the root of the Provider.

        Returns:
            set: set of all the objects found at the root of the Provider.
        """
        return set(self.dict.keys())

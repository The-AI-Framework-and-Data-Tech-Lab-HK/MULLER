# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/core/storage/local.py
#
# Modifications Copyright (c) 2026 Bingyu Liu

import os
import pathlib
import posixpath
import shutil
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Set, Dict

from muller.core.storage.provider import StorageProvider
from muller.util.exceptions import (DirectoryAtPathException,
                                   FileAtPathException,
                                   PathNotEmptyException,
                                   ReadOnlyModeError)


class LocalProvider(StorageProvider):
    """Provider class for using the local filesystem."""

    def __init__(self, root: str, files: set = None):  # root is a string variable represent the path
        """Initilizaes the Local Provider.

        Example:

            >>> local_provider = LocalProvider("/home/ubuntu/Documents/")

        Args:
            root (str): The root of the provider. All read/write request keys will be appended to root."
            files (set): The set of file names shared with other threads.

        Raises:
            FileAtPathException: If the root is a file instead of a directory.
        """
        if os.path.isfile(root):
            raise FileAtPathException(root)
        self.root = root
        self.files: Optional[Set[str]] = None
        if files:
            self.files = files

        self.expiration: Optional[str] = None
        self.db_engine: bool = False
        self.repository: Optional[str] = None

    def __getitem__(self, key: str):
        try:
            full_path = self._check_is_file(key)
            with open(full_path, "rb") as file:
                return file.read()
        except DirectoryAtPathException:
            raise
        except FileNotFoundError as e:
            raise KeyError(key) from e

    def __setstate__(self, state):
        self.__init__(root=state[0], files=state[1])

    def __getstate__(self):
        return self.root, self.files

    def __setitem__(self, key: str, value: bytes):
        self.check_readonly()
        full_path = self._check_is_file(key)
        directory = os.path.dirname(full_path)
        if os.path.isfile(directory):
            raise FileAtPathException(directory)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        with open(full_path, 'wb') as file:
            file.write(value)
        if self.files is not None:
            self.files.add(key)

    def __delitem__(self, key: str):
        """Delete the object present at the path.

        Example:

            >>> local_provider = LocalProvider("/home/ubuntu/Documents/")
            >>> del local_provider["abc.txt"]

        Args:
            key (str): the path to the object relative to the root of the provider.

        Raises:
            KeyError: If an object is not found at the path.
            DirectoryAtPathException: If a directory is found at the path.
            Exception: Any other exception encountered while trying to fetch the object.
            ReadOnlyError: If the provider is in read-only mode.
        """
        self.check_readonly()
        try:
            full_path = self._check_is_file(key)
            os.remove(full_path)
            if self.files is not None:
                self.files.discard(key)
        except DirectoryAtPathException:
            raise
        except FileNotFoundError as e:
            raise KeyError from e

    def __iter__(self):
        """Generator function that iterates over the keys of the provider.

        Example:

            >>> local_provider = LocalProvider("/home/ubuntu/Documents/")
            >>> for my_data in local_provider:
            ...    pass

        Yields:
            str: the path of the object that it is iterating over, relative to the root of the provider.
        """
        yield from self._all_keys()

    def __len__(self):
        """Returns the number of files present inside the root of the provider.

        Example:

            >>> local_provider = LocalProvider("/home/ubuntu/Documents/")
            >>> len(local_provider)

        Returns:
            int: the number of files present inside the root.
        """
        return len(self._all_keys())

    def subdir(self, path: str, read_only: bool = False):
        sd = self.__class__(os.path.join(self.root, path))
        sd.read_only = read_only
        return sd

    def get_bytes(self,
                  path: str,
                  start_byte: Optional[int] = None,
                  end_byte: Optional[int] = None):
        try:
            full_path = self._check_is_file(path)
            with open(full_path, 'rb') as file:
                if start_byte is not None:
                    file.seek(start_byte)
                if end_byte is None:
                    return file.read()
                return file.read(end_byte - (start_byte or 0))
        except DirectoryAtPathException:
            raise
        except FileNotFoundError as e:
            raise KeyError(path) from e

    def check_readonly(self):
        """Raises an exception if the provider is in read-only mode."""
        if self.read_only:
            raise ReadOnlyModeError()

    def get_object_size(self, key: str) -> int:
        return int(os.stat(os.path.join(self.root, key)).st_size)

    def clear(self, prefix=""):
        """Deletes ALL data with keys having given prefix on the local machine (under self.root). Exercise caution!"""
        self.check_readonly()
        full_path = os.path.expanduser(self.root)

        if prefix:
            self.files = self._all_keys()
            self.files = set(file for file in self.files if not file.startswith(prefix))
            full_path = os.path.join(full_path, prefix)
        else:
            self.files = set()
        if os.path.exists(full_path):
            shutil.rmtree(full_path)

    def get_items(self, keys: Set[str], ignore_key_error: bool = False, max_workers: Optional[int] = None) -> \
    Dict[str, bytes]:
        """Multithreaded read from local device.
        Child threads call self.__getitem__, any exception that child threads raise will be raised.

        Args:
            keys: paths to read.

            ignore_key_error: If True, if True, keys that raise KeyError be ignored, then the output may not include
            all key in keys.

            max_workers: Used as ThreadPoolExecutor(max_workers) to launch multithread task.

        Raises:
            Any exception that self.__getitem__ raises.

        Returns:
            Dict[str, bytes]: A dictionary as {path: self[path] for path in path_list}.
        """

        def get_file(path: str):
            return path, self[path]

        dic = {}

        if not ignore_key_error:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for result in executor.map(get_file, keys):
                    dic[result[0]] = result[1]
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(get_file, path) for path in keys]
                for future in futures:
                    try:
                        res = future.result()
                        dic[res[0]] = res[1]
                    except KeyError:
                        pass
        return dic

    def set_items(self, contents: Dict[str, bytes], max_workers: Optional[int] = None):
        """Multithreaded write into local disk.
        Child threads call self.__setitem__, any exception that child threads raise will be raised.

        Args:
            contents: paths and contents to write as {path: content, ...}.
            max_workers: Used as ThreadPoolExecutor(max_workers) to launch multithread task.

        Raises:
            Any exception that self.__setitem__ raises.
        """
        self.check_readonly()

        def set_file(tup):
            path, value = tup
            full_path = self._check_is_file(path)
            directory = os.path.dirname(full_path)
            if os.path.isfile(directory):
                raise FileAtPathException(directory)
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            with open(full_path, 'wb') as file:
                file.write(value)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for _ in executor.map(set_file, contents.items()):  # Trigger exception
                pass

        if self.files is not None:
            self.files.update(contents)

    def del_items(self, keys: Set[str], max_workers: Optional[int] = None):
        """Multithreaded delete from local device.
        Child threads delete files, and finally main thread updates self.files.

        Args:
            keys: keys to delete
            max_workers: Used as ThreadPoolExecutor(max_workers) to launch multithread task.

        Raises:
            DirectoryAtPathException, FileNotFoundError
        """
        self.check_readonly()

        def del_file(path: str):
            try:
                full_path = self._check_is_file(path)
                os.remove(full_path)
            except DirectoryAtPathException:
                raise
            except FileNotFoundError as e:
                raise KeyError from e

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for _ in executor.map(del_file, keys):  # Trigger exception
                pass

        if self.files is not None:
            self.files.difference_update(keys)

    def rename(self, path: str):
        # rename the dataset to a new path name
        """Renames root folder"""
        if os.path.isfile(path) or (os.path.isdir(path) and len(os.listdir(path)) > 0):  # and is prior to or
            raise PathNotEmptyException(use_cloud=False)
        os.rename(self.root, path)
        self.root = path

    def _all_keys(self, refresh_tag: bool = False) -> Set[str]:
        """Lists all the objects present at the root of the Provider.

        Args:
            refresh_tag (bool): refresh keys

        Returns:
            set: set of all the objects found at the root of the Provider.
        """
        if self.files is None or refresh_tag:
            key_set = set()
            full_path = os.path.expanduser(self.root)
            for root, _, files in os.walk(full_path):
                for file_name in files:
                    key_set.add(
                        posixpath.relpath(
                            posixpath.join(pathlib.Path(root).as_posix(), file_name),
                            pathlib.Path(full_path).as_posix(),
                        )
                    )
            self.files = key_set
        return self.files

    def _check_is_file(self, key: str):
        """Checks if the path is a file. Returns the full_path to file if True.

        Args:
            key (str): the path to the object relative to the root of the provider.

        Returns:
            str: the full path to the requested file.

        Raises:
            DirectoryAtPathException: If a directory is found at the path.
        """
        fpath = posixpath.join(self.root, key)
        fpath = os.path.expanduser(fpath)
        fpath = str(pathlib.Path(fpath))
        if os.path.isdir(fpath):
            raise DirectoryAtPathException
        return fpath

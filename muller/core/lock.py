# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/core/lock.py
#
# Modifications Copyright (c) 2026 Xueling Lin

import atexit
import logging
import struct
import threading
import time
import uuid
from collections import defaultdict
from os import getpid
from typing import Dict, Callable, Optional, Set

import muller
from ..core.storage import StorageProvider, LocalProvider, MemoryProvider
from ..util.exceptions import LockedException
from ..util.path import get_path_from_storage
from ..util.remove_cache import get_base_storage


class Lock(object):
    def __init__(self, storage: StorageProvider, path: str, duration: int = 10):
        self.storage = storage
        self._lock_verify_interval = (
            0.01
            if isinstance(storage, (LocalProvider, MemoryProvider))
            else muller.constants.LOCK_VERIFY_INTERVAL
        )
        self.path = path
        self.username = None
        self.tag = int.to_bytes(getpid(), 4, "little")
        self.duration = duration
        self._min_sleep = (
            0.01 if isinstance(storage, (LocalProvider, MemoryProvider)) else 1
        )
        self.acquired = False

    def __enter__(self):
        self.acquire()

    def __exit__(self, *args, **kwargs):
        self.release()

    def refresh_lock(self):
        """Function to refresh lock"""
        storage = self.storage
        path = self.path
        byts = storage.get(path)
        if not byts:
            raise LockedException()
        nodeid, _, tag = _parse_lock_bytes(byts)
        if tag != self.tag or nodeid != uuid.getnode():
            raise LockedException()
        self._write_lock()

    def acquire(self, timeout: Optional[int] = 0):
        """Function to acquire lock."""
        storage = self.storage
        path = self.path
        if timeout is not None:
            start_time = time.time()
        while True:
            try:
                byts = storage.get(path)
            except Exception:
                byts = None
            if byts:
                nodeid, timestamp, tag = _parse_lock_bytes(byts)
                locked = tag != self.tag or nodeid != uuid.getnode()
                if not locked:  # Identical lock
                    return
            else:
                locked = False

            if locked:
                rem = timestamp - time.time()
                if rem > 0:
                    if timeout is not None and time.time() - start_time > timeout:
                        raise LockedException()
                    time.sleep(min(rem, self._min_sleep))
                    continue

            self._write_lock()  # We write the version_control_info.lock here in the storage.
            time.sleep(self._lock_verify_interval)
            try:
                byts = storage.get(path)
            except Exception:
                byts = None
            if not byts:
                continue
            try:
                nodeid, timestamp, tag = _parse_lock_bytes(byts)
            except (AssertionError, struct.error, IndexError) as e:
                raise ValueError from e
            if self.tag == tag and nodeid == uuid.getnode():
                self.acquired = True
                return
            rem = timestamp - time.time()
            if rem > 0:
                time.sleep(min(rem, self._min_sleep))
            continue

    def release(self):
        """Function to release lock."""
        if not self.acquired:
            return
        storage = self.storage
        read_only = False
        try:
            read_only = storage.read_only
            storage.disable_readonly()
            del storage[self.path]
        except Exception as e:
            logging.warning(f"Failed to obtain read only, lock may be corrupt. Error: {e}")
        finally:
            if read_only:
                storage.enable_readonly()

    def _write_lock(self):
        storage = self.storage
        read_only = False
        try:
            read_only = storage.read_only
            storage.disable_readonly()
            storage[self.path] = _get_lock_bytes(self.tag, self.duration)
        finally:
            if read_only:
                storage.enable_readonly()


class PersistentLock(Lock):
    """Locks a StorageProvider instance to avoid concurrent writes from multiple machines.

    Example:
        From machine 1:
        s3 = muller.core.storage.S3Provider(S3_URL)
        lock = muller.core.lock.Lock(s3)  # Works

        From machine 2:
        s3 = muller.core.storage.S3Provider(S3_URL)
        lock = muller.core.lock.Lock(s3)  # Raises LockedException

        The lock is updated every 2 mins by an internal thread. The lock is valid for 5 mins after the last update.

    Args:
        storage (StorageProvider): The storage provider to be locked.
        lock_lost_callback (Callable, optional): Called if the lock is lost after acquiring.
        timeout(int, optional): Keep trying to acquire the lock for the given number of seconds before throwing a
        LockedException. Passing None will wait forever
    Raises:
        LockedException: If the storage is already locked by a different machine.
    """

    def __init__(
        self,
        storage: StorageProvider,
        path: Optional[str] = None,
        lock_lost_callback: Optional[Callable] = None,
        timeout: Optional[int] = 0,
    ):
        super().__init__(storage, path, 10)
        self.storage = storage
        self.path = "dataset_lock.lock" if path is None else path
        self.lock_lost_callback = lock_lost_callback
        self.acquired = False
        self._thread_lock = threading.Lock()
        self._previous_update_timestamp = None
        self.lock = Lock(storage, self.path, muller.constants.DATASET_LOCK_VALIDITY)
        self.acquired = False
        self.timeout = timeout
        self.acquire()
        atexit.register(self.release)

    def acquire(self, timeout: Optional[int] = 0):
        if self.acquired:
            return
        self.lock.acquire(timeout=self.timeout)
        self._thread = threading.Thread(target=self._lock_loop, daemon=True)
        self._thread.start()
        self.acquired = True

    def release(self):
        """Function to release persistent."""
        if not self.acquired:
            return
        with self._thread_lock:
            self._acquired = False # terminate_thread(self._thread)
        try:
            self.lock.release()
        except Exception as e:
            logging.warning(f"Failed to release lock, lock may be corrupt. Error: {e}")

    def _lock_loop(self):
        try:
            while True:
                time.sleep(muller.constants.DATASET_LOCK_UPDATE_INTERVAL)
                try:
                    self.lock.refresh_lock(timeout=self.timeout)
                except LockedException:
                    if self.lock_lost_callback:
                        self.lock_lost_callback()
                    return
        except Exception:  # Thread termination
            return


_LOCKS: Dict[str, Lock] = {}
_REFS: Dict[str, Set[int]] = defaultdict(set)


def lock_dataset(
    dataset,
    lock_lost_callback: Optional[Callable] = None,
    version: Optional[str] = None,
):
    """Locks a StorageProvider instance to avoid concurrent writes from multiple machines.

    Args:
        dataset: Dataset instance.
        lock_lost_callback (Callable, Optional): Called if the lock is lost after acquiring.
        version (str, optional): The version to be locked. If None, the current version is locked.

    Raises:
        LockedException: If the storage is already locked by a different machine.
    """
    storage = get_base_storage(dataset.storage)
    version = version or dataset.version_state["commit_id"]
    key = _get_lock_key(get_path_from_storage(storage), version)
    lock = _LOCKS.get(key)
    if lock:
        lock.acquire()
    else:
        lock = PersistentLock(
            storage,
            path=_get_lock_file_path(version),
            lock_lost_callback=lock_lost_callback,
            timeout=dataset._lock_timeout,
        )
        _LOCKS[key] = lock
    _REFS[key].add(id(dataset))


def unlock_dataset(dataset, version: Optional[str] = None):
    """Unlocks a storage provider that was locked by this machine.

    Args:
        dataset: The dataset to be unlocked
        version (str, optional): The version to be unlocked. If None, the current version is unlocked.
    """
    storage = get_base_storage(dataset.storage)
    version = version or dataset.version_state["commit_id"]
    key = _get_lock_key(get_path_from_storage(storage), version)
    try:
        lock = _LOCKS[key]
        ref_set = _REFS[key]
        try:
            ref_set.remove(id(dataset))
        except KeyError:
            pass
        if not ref_set:
            lock.release()
            del _REFS[key]
            del _LOCKS[key]
    except KeyError:
        pass


def _get_lock_key(storage_path: str, commit_id: str):
    return storage_path + ":" + commit_id


def _get_lock_file_path(version: Optional[str] = None) -> str:
    if version in (None, "firstdbf9474d461a19e9333c2fd19b46115348f"):
        return "dataset_lock.lock"
    return "versions/" + version + "/" + "dataset_lock.lock"  # type: ignore


def _get_lock_bytes(tag: Optional[bytes] = None, duration: int = 10) -> bytes:
    byts = uuid.getnode().to_bytes(6, "little") + struct.pack(
        "d", time.time() + duration
    )
    if tag:
        byts += tag
    return byts


def _parse_lock_bytes(byts):
    assert len(byts) >= 14, len(byts)
    byts = memoryview(byts)
    nodeid = int.from_bytes(byts[:6], "little")
    timestamp = struct.unpack("d", byts[6:14])[0]
    tag = byts[14:]
    return nodeid, timestamp, tag

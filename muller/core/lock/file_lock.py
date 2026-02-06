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

"""
File-based distributed lock implementation using storage provider.
"""

import struct
import time
import uuid
from os import getpid
from typing import Optional

import muller
from muller.core.lock.base import BaseLock
from muller.core.storage import StorageProvider, LocalProvider, MemoryProvider
from muller.util.exceptions import LockedException


def _get_lock_bytes(tag: Optional[bytes] = None, duration: int = 10) -> bytes:
    """Generate lock bytes containing node ID, timestamp, and tag.
    
    Args:
        tag: Optional tag bytes for lock identification.
        duration: Lock duration in seconds.
        
    Returns:
        Lock bytes containing node ID (6 bytes), timestamp (8 bytes), and optional tag.
    """
    byts = uuid.getnode().to_bytes(6, "little") + struct.pack(
        "d", time.time() + duration
    )
    if tag:
        byts += tag
    return byts


def _parse_lock_bytes(byts):
    """Parse lock bytes to extract node ID, timestamp, and tag.
    
    Args:
        byts: Lock bytes to parse.
        
    Returns:
        Tuple of (node_id, timestamp, tag).
    """
    assert len(byts) >= 14, len(byts)
    byts = memoryview(byts)
    nodeid = int.from_bytes(byts[:6], "little")
    timestamp = struct.unpack("d", byts[6:14])[0]
    tag = byts[14:]
    return nodeid, timestamp, tag


class FileLock(BaseLock):
    """File-based distributed lock using storage provider.
    
    This lock implementation writes lock information to the storage provider
    to coordinate access across multiple processes or machines.
    
    Example:
        >>> from muller.core.storage import LocalProvider
        >>> storage = LocalProvider("/path/to/storage")
        >>> lock = FileLock(storage, "my_lock.lock")
        >>> with lock:
        ...     # Critical section
        ...     pass
    
    Args:
        storage: The storage provider to use for lock files.
        path: The lock file path within the storage.
        duration: Lock validity duration in seconds (default: 10).
    """
    
    def __init__(self, storage: StorageProvider, path: str, duration: int = 10):
        """Initialize the file lock.
        
        Args:
            storage: The storage provider to use for lock files.
            path: The lock file path within the storage.
            duration: Lock validity duration in seconds (default: 10).
        """
        super().__init__(path, duration)
        self.storage = storage
        self._lock_verify_interval = (
            0.01
            if isinstance(storage, (LocalProvider, MemoryProvider))
            else muller.constants.LOCK_VERIFY_INTERVAL
        )
        self.username = None
        self.tag = int.to_bytes(getpid(), 4, "little")
        self._min_sleep = (
            0.01 if isinstance(storage, (LocalProvider, MemoryProvider)) else 1
        )

    def refresh_lock(self, timeout: Optional[int] = None):
        """Refresh the lock to extend its validity.
        
        Args:
            timeout: Unused parameter for interface compatibility.
        
        Raises:
            LockedException: If the lock is no longer held by this instance.
        """
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
        """Acquire the lock.
        
        Args:
            timeout: Maximum time to wait for the lock in seconds.
                     0 means fail immediately if lock is held.
                     None means wait indefinitely.
        
        Raises:
            LockedException: If the lock cannot be acquired within the timeout.
        """
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

            self._write_lock()  # We write the lock file in the storage.
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
        """Release the lock.
        
        This method is safe to call even if the lock is not held.
        """
        if not self.acquired:
            return
        storage = self.storage
        read_only = False
        try:
            read_only = storage.read_only
            storage.disable_readonly()
            del storage[self.path]
        except Exception:
            pass
        finally:
            if read_only:
                storage.enable_readonly()
        self.acquired = False

    def _write_lock(self):
        """Write lock bytes to storage."""
        storage = self.storage
        read_only = False
        try:
            read_only = storage.read_only
            storage.disable_readonly()
            storage[self.path] = _get_lock_bytes(self.tag, self.duration)
        finally:
            if read_only:
                storage.enable_readonly()

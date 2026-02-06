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
Persistent lock that auto-refreshes to maintain lock validity.
"""

import atexit
import threading
import time
from typing import Callable, Optional

import muller
from muller.core.lock.base import BaseLock
from muller.core.lock.file_lock import FileLock
from muller.core.storage import StorageProvider
from muller.util.exceptions import LockedException


class PersistentLock(BaseLock):
    """Persistent lock that auto-refreshes to maintain lock validity.
    
    This lock wraps any BaseLock implementation and automatically refreshes
    the lock in a background thread to prevent expiration during long operations.
    
    Example:
        From machine 1:
        >>> s3 = muller.core.storage.S3Provider(S3_URL)
        >>> lock = PersistentLock(s3)  # Works

        From machine 2:
        >>> s3 = muller.core.storage.S3Provider(S3_URL)
        >>> lock = PersistentLock(s3)  # Raises LockedException

        The lock is updated every 2 mins by an internal thread. 
        The lock is valid for 5 mins after the last update.
    
    Args:
        storage: The storage provider to be locked (for FileLock).
        path: The lock file path (default: "dataset_lock.lock").
        lock_lost_callback: Called if the lock is lost after acquiring.
        timeout: Keep trying to acquire the lock for the given number of seconds
                 before throwing a LockedException. None waits forever.
        lock: Optional pre-configured lock instance (for dependency injection).
    
    Raises:
        LockedException: If the storage is already locked by a different machine.
    """

    def __init__(
        self,
        storage: Optional[StorageProvider] = None,
        path: Optional[str] = None,
        lock_lost_callback: Optional[Callable] = None,
        timeout: Optional[int] = 0,
        lock: Optional[BaseLock] = None,
    ):
        """Initialize the persistent lock.
        
        Args:
            storage: The storage provider to be locked (for FileLock).
            path: The lock file path (default: "dataset_lock.lock").
            lock_lost_callback: Called if the lock is lost after acquiring.
            timeout: Keep trying to acquire the lock for the given number of seconds
                     before throwing a LockedException. None waits forever.
            lock: Optional pre-configured lock instance (for dependency injection).
        """
        lock_path = "dataset_lock.lock" if path is None else path
        super().__init__(lock_path, 10)
        
        self.storage = storage
        self.lock_lost_callback = lock_lost_callback
        self._thread_lock = threading.Lock()
        self._previous_update_timestamp = None
        self._acquired = False
        self.timeout = timeout
        self._thread = None
        
        # Use provided lock or create a FileLock
        if lock is not None:
            self.lock = lock
        elif storage is not None:
            self.lock = FileLock(storage, lock_path, muller.constants.DATASET_LOCK_VALIDITY)
        else:
            raise ValueError("Either 'storage' or 'lock' must be provided")
        
        self.acquire()
        atexit.register(self.release)

    @property
    def acquired(self):
        """Whether the lock is currently held."""
        return self._acquired
    
    @acquired.setter
    def acquired(self, value):
        """Set the acquired state."""
        self._acquired = value

    def acquire(self, timeout: Optional[int] = None):
        """Acquire the lock and start the auto-refresh thread.
        
        Args:
            timeout: Maximum time to wait for the lock in seconds.
                     Uses instance timeout if not provided.
        """
        if self._acquired:
            return
        
        actual_timeout = timeout if timeout is not None else self.timeout
        self.lock.acquire(timeout=actual_timeout)
        self._thread = threading.Thread(target=self._lock_loop, daemon=True)
        self._thread.start()
        self._acquired = True

    def release(self):
        """Release the lock and stop the auto-refresh thread."""
        if not self._acquired:
            return
        with self._thread_lock:
            self._acquired = False
        try:
            self.lock.release()
        except Exception:
            pass

    def refresh_lock(self, timeout: Optional[int] = None):
        """Refresh the underlying lock.
        
        Args:
            timeout: Unused parameter for interface compatibility.
        
        Raises:
            LockedException: If the lock is no longer held.
        """
        if not self._acquired:
            raise LockedException()
        self.lock.refresh_lock(timeout=timeout)

    def _lock_loop(self):
        """Background thread that periodically refreshes the lock."""
        try:
            while True:
                time.sleep(muller.constants.DATASET_LOCK_UPDATE_INTERVAL)
                with self._thread_lock:
                    if not self._acquired:
                        return
                try:
                    self.lock.refresh_lock(timeout=self.timeout)
                except LockedException:
                    if self.lock_lost_callback:
                        self.lock_lost_callback()
                    return
        except Exception:  # Thread termination
            return


def create_lock(
    storage: Optional[StorageProvider] = None,
    path: Optional[str] = None,
    duration: int = 10,
    lock_type: Optional[str] = None,
    redis_client=None,
) -> BaseLock:
    """Factory function to create appropriate lock based on configuration.
    
    Args:
        storage: The storage provider (required for FileLock).
        path: The lock path/key.
        duration: Lock duration in seconds.
        lock_type: Lock type ("file" or "redis"). Uses config default if None.
        redis_client: Redis client instance (required for RedisLock).
    
    Returns:
        A lock instance (FileLock or RedisLock).
    
    Raises:
        ValueError: If required parameters are missing for the lock type.
    """
    if lock_type is None:
        lock_type = getattr(muller.constants, 'LOCK_TYPE', 'file')
    
    if lock_type == "redis":
        if redis_client is None:
            # Try to create a Redis client from configuration
            import redis
            redis_client = redis.Redis(
                host=getattr(muller.constants, 'REDIS_LOCK_HOST', 'localhost'),
                port=getattr(muller.constants, 'REDIS_LOCK_PORT', 6379),
                db=getattr(muller.constants, 'REDIS_LOCK_DB', 0),
                password=getattr(muller.constants, 'REDIS_LOCK_PASSWORD', None),
            )
        from muller.core.lock.redis_lock import RedisLock
        return RedisLock(redis_client, path or "default", duration)
    else:
        # Default to FileLock
        if storage is None:
            raise ValueError("storage is required for FileLock")
        return FileLock(storage, path or "dataset_lock.lock", duration)

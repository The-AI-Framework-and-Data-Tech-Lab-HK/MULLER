# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

"""
Redis-based distributed lock implementation.
"""

import time
import uuid
from os import getpid
from typing import Optional

from muller.core.lock.base import BaseLock
from muller.util.exceptions import LockedException

# Lua script for atomic release - only delete if we own the lock
RELEASE_SCRIPT = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""

# Lua script for atomic refresh - only extend if we own the lock
REFRESH_SCRIPT = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("pexpire", KEYS[1], ARGV[2])
else
    return 0
end
"""


class RedisLock(BaseLock):
    """Redis-based distributed lock.
    
    This lock implementation uses Redis SET with NX and PX options for
    atomic lock acquisition, and Lua scripts for safe release and refresh.
    
    Example:
        >>> import redis
        >>> client = redis.Redis(host='localhost', port=6379, db=0)
        >>> lock = RedisLock(client, "my_lock")
        >>> with lock:
        ...     # Critical section
        ...     pass
    
    Args:
        redis_client: A Redis client instance.
        path: The lock key name in Redis.
        duration: Lock validity duration in seconds (default: 10).
        prefix: Optional prefix for the lock key (default: "muller:lock:").
    """
    
    def __init__(
        self,
        redis_client,
        path: str,
        duration: int = 10,
        prefix: str = "muller:lock:",
    ):
        """Initialize the Redis lock.
        
        Args:
            redis_client: A Redis client instance.
            path: The lock key name in Redis.
            duration: Lock validity duration in seconds (default: 10).
            prefix: Optional prefix for the lock key (default: "muller:lock:").
        """
        super().__init__(path, duration)
        self.redis_client = redis_client
        self.prefix = prefix
        self._lock_key = f"{prefix}{path}"
        # Generate a unique token for this lock instance
        self._token = f"{uuid.getnode()}:{getpid()}:{uuid.uuid4().hex}"
        self._release_script = None
        self._refresh_script = None
        self._min_sleep = 0.1  # Minimum sleep time between retries
    
    def _get_release_script(self):
        """Get or register the release Lua script."""
        if self._release_script is None:
            self._release_script = self.redis_client.register_script(RELEASE_SCRIPT)
        return self._release_script
    
    def _get_refresh_script(self):
        """Get or register the refresh Lua script."""
        if self._refresh_script is None:
            self._refresh_script = self.redis_client.register_script(REFRESH_SCRIPT)
        return self._refresh_script

    def acquire(self, timeout: Optional[int] = 0):
        """Acquire the lock.
        
        Uses Redis SET with NX (only set if not exists) and PX (expiration in ms)
        for atomic lock acquisition.
        
        Args:
            timeout: Maximum time to wait for the lock in seconds.
                     0 means fail immediately if lock is held.
                     None means wait indefinitely.
        
        Raises:
            LockedException: If the lock cannot be acquired within the timeout.
        """
        duration_ms = self.duration * 1000
        
        if timeout is not None:
            start_time = time.time()
        
        while True:
            # Try to acquire the lock atomically
            # SET key value NX PX milliseconds
            result = self.redis_client.set(
                self._lock_key,
                self._token,
                nx=True,  # Only set if not exists
                px=duration_ms,  # Expiration in milliseconds
            )
            
            if result:
                # Lock acquired successfully
                self.acquired = True
                return
            
            # Check if we already own the lock (re-entrant check)
            current_value = self.redis_client.get(self._lock_key)
            if current_value and current_value.decode('utf-8') == self._token:
                # We already own this lock
                self.acquired = True
                return
            
            # Lock is held by someone else
            if timeout is not None and timeout == 0:
                raise LockedException()
            
            if timeout is not None and time.time() - start_time > timeout:
                raise LockedException()
            
            # Get TTL to know how long to wait
            ttl_ms = self.redis_client.pttl(self._lock_key)
            if ttl_ms > 0:
                # Wait for a portion of the TTL or min_sleep
                wait_time = min(ttl_ms / 1000.0, self._min_sleep)
                time.sleep(wait_time)
            else:
                # No TTL or key doesn't exist, try again soon
                time.sleep(self._min_sleep)

    def release(self):
        """Release the lock.
        
        Uses a Lua script to atomically check ownership and delete the lock.
        This prevents accidentally releasing a lock that was re-acquired by
        another process after our lock expired.
        """
        if not self.acquired:
            return
        
        try:
            release_script = self._get_release_script()
            release_script(keys=[self._lock_key], args=[self._token])
        except Exception:
            pass
        finally:
            self.acquired = False

    def refresh_lock(self, timeout: Optional[int] = None):
        """Refresh the lock to extend its validity.
        
        Uses a Lua script to atomically check ownership and extend the TTL.
        
        Args:
            timeout: Unused parameter for interface compatibility.
        
        Raises:
            LockedException: If the lock is no longer held by this instance.
        """
        if not self.acquired:
            raise LockedException()
        
        duration_ms = self.duration * 1000
        refresh_script = self._get_refresh_script()
        result = refresh_script(
            keys=[self._lock_key],
            args=[self._token, duration_ms]
        )
        
        if not result:
            self.acquired = False
            raise LockedException()

# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

"""
Abstract base class for distributed locks.
"""

from abc import ABC, abstractmethod
from typing import Optional


class BaseLock(ABC):
    """Abstract base class for distributed locks.
    
    This class defines the interface for all lock implementations.
    Subclasses must implement acquire(), release(), and refresh_lock() methods.
    
    Attributes:
        path: The lock identifier/path.
        duration: The lock duration in seconds.
        acquired: Whether the lock is currently held.
    """
    
    def __init__(self, path: str, duration: int = 10):
        """Initialize the lock.
        
        Args:
            path: The lock identifier/path.
            duration: The lock duration in seconds (default: 10).
        """
        self.path = path
        self.duration = duration
        self.acquired = False
    
    def __enter__(self):
        """Context manager entry - acquires the lock."""
        self.acquire()
        return self
    
    def __exit__(self, *args, **kwargs):
        """Context manager exit - releases the lock."""
        self.release()
    
    @abstractmethod
    def acquire(self, timeout: Optional[int] = 0) -> None:
        """Acquire the lock.
        
        Args:
            timeout: Maximum time to wait for the lock in seconds.
                     0 means fail immediately if lock is held.
                     None means wait indefinitely.
        
        Raises:
            LockedException: If the lock cannot be acquired within the timeout.
        """
        pass
    
    @abstractmethod
    def release(self) -> None:
        """Release the lock.
        
        This method should be safe to call even if the lock is not held.
        """
        pass
    
    @abstractmethod
    def refresh_lock(self) -> None:
        """Refresh/extend the lock duration.
        
        Raises:
            LockedException: If the lock is no longer held by this instance.
        """
        pass

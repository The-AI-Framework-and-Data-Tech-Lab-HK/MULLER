# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

"""
Lock module for MULLER.

Provides distributed locking mechanisms for controlling concurrent writes.
Supports both file-based locks (FileLock) and Redis-based locks (RedisLock).
"""

from muller.core.lock.base import BaseLock
from muller.core.lock.file_lock import FileLock
from muller.core.lock.redis_lock import RedisLock
from muller.core.lock.persistent import PersistentLock
from muller.core.lock.utils import (
    lock_dataset,
    unlock_dataset,
    _get_lock_key,
    _get_lock_file_path,
    _get_lock_bytes,
    _parse_lock_bytes,
)

# Backward compatibility: Lock is an alias for FileLock
Lock = FileLock

__all__ = [
    "BaseLock",
    "Lock",
    "FileLock",
    "RedisLock",
    "PersistentLock",
    "lock_dataset",
    "unlock_dataset",
]

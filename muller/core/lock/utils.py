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
Utility functions for dataset locking.
"""

import struct
import time
import uuid
from collections import defaultdict
from typing import Callable, Dict, Optional, Set

from muller.core.lock.base import BaseLock
from muller.core.lock.persistent import PersistentLock
from muller.util.path import get_path_from_storage
from muller.util.remove_cache import get_base_storage


# Global lock registry
_LOCKS: Dict[str, BaseLock] = {}
_REFS: Dict[str, Set[int]] = defaultdict(set)


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


def _get_lock_key(storage_path: str, identifier: str, user: Optional[str] = None) -> str:
    """Generate lock key based on storage path and branch/commit identifier.
    
    Args:
        storage_path: Storage path
        identifier: Branch name or commit ID
        user: Optional user identifier for additional isolation
        
    Returns:
        Lock key string
    """
    key = f"{storage_path}:{identifier}"
    if user:
        key = f"{key}:{user}"
    return key


def _get_lock_file_path(branch: str, commit_id: Optional[str] = None, 
                       is_head_node: bool = True) -> str:
    """Generate lock file path based on branch (HEAD) or commit (non-HEAD).
    
    Strategy:
    - HEAD node: Lock at branch level (branches/{branch}/branch_lock.lock)
    - Non-HEAD node: Lock at commit level (versions/{commit_id}/dataset_lock.lock)
    - Backward compatibility: If LOCK_BY_BRANCH is False, use commit-based locking
    
    Args:
        branch: The branch name
        commit_id: The commit ID (used for non-HEAD nodes or legacy mode)
        is_head_node: Whether this is a HEAD node (default: True)
        
    Returns:
        Lock file path
    """
    from muller.constants import FIRST_COMMIT_ID, LOCK_BY_BRANCH
    
    # Special handling for main branch initial commit
    if branch == "main" and commit_id in (None, FIRST_COMMIT_ID):
        return "dataset_lock.lock"
    
    # Backward compatibility: use old commit-based locking if disabled
    if not LOCK_BY_BRANCH:
        if commit_id:
            return f"versions/{commit_id}/dataset_lock.lock"
        return "dataset_lock.lock"
    
    # New branch-based locking for HEAD nodes
    if is_head_node:
        return f"branches/{branch}/branch_lock.lock"
    
    # Non-HEAD node: commit-level lock (backward compatible)
    if commit_id:
        return f"versions/{commit_id}/dataset_lock.lock"
    
    # Fallback to branch lock
    return f"branches/{branch}/branch_lock.lock"


def lock_dataset(
    dataset,
    lock_lost_callback: Optional[Callable] = None,
    version: Optional[str] = None,
    branch: Optional[str] = None,
):
    """Locks a dataset branch (HEAD) or commit (non-HEAD) to avoid concurrent writes.
    
    Strategy:
    - HEAD node: Lock branch level for multi-user collaboration
    - Non-HEAD node: Lock commit level to prevent history modification conflicts
    
    Args:
        dataset: Dataset instance.
        lock_lost_callback: Called if the lock is lost after acquiring.
        version: Commit ID (for backward compatibility and non-HEAD locking)
        branch: The branch to lock. If None, uses current branch.
        
    Raises:
        LockedException: If the branch/commit is already locked by another process.
    """
    storage = get_base_storage(dataset.storage)
    
    # Get branch and commit info from dataset
    if branch is None:
        branch = dataset.version_state.get("branch", "main")
    
    commit_id = version or dataset.version_state.get("commit_id")
    
    # Determine if this is a HEAD node
    commit_node = dataset.version_state.get("commit_node")
    is_head_node = commit_node.is_head_node if commit_node else True
    
    # Check if branch-based locking is enabled
    from muller.constants import LOCK_BY_BRANCH
    
    # Generate lock key and path based on HEAD status and configuration
    if LOCK_BY_BRANCH and is_head_node:
        # HEAD node with branch locking enabled: lock by branch
        identifier = branch
        lock_path = _get_lock_file_path(branch, commit_id, is_head_node=True)
    else:
        # Non-HEAD node OR legacy commit-based locking: lock by commit
        identifier = commit_id
        lock_path = _get_lock_file_path(branch, commit_id, is_head_node=False)
    
    key = _get_lock_key(get_path_from_storage(storage), identifier)
    lock = _LOCKS.get(key)
    
    if lock:
        lock.acquire()
    else:
        lock = PersistentLock(
            storage,
            path=lock_path,
            lock_lost_callback=lock_lost_callback,
            timeout=dataset._lock_timeout,
        )
        _LOCKS[key] = lock
    _REFS[key].add(id(dataset))


def unlock_dataset(dataset, version: Optional[str] = None, branch: Optional[str] = None):
    """Unlocks a dataset branch (HEAD) or commit (non-HEAD).
    
    Args:
        dataset: The dataset to be unlocked
        version: Commit ID (for backward compatibility)
        branch: The branch to unlock. If None, uses current branch.
    """
    storage = get_base_storage(dataset.storage)
    
    # Get branch and commit info from dataset
    if branch is None:
        branch = dataset.version_state.get("branch", "main")
    
    commit_id = version or dataset.version_state.get("commit_id")
    
    # Determine if this is a HEAD node
    commit_node = dataset.version_state.get("commit_node")
    is_head_node = commit_node.is_head_node if commit_node else True
    
    # Check if branch-based locking is enabled
    from muller.constants import LOCK_BY_BRANCH
    
    # Generate lock key based on HEAD status and configuration
    if LOCK_BY_BRANCH and is_head_node:
        identifier = branch
    else:
        identifier = commit_id
    
    key = _get_lock_key(get_path_from_storage(storage), identifier)
    
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

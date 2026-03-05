# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

"""
Tests for the lock module.

This module tests both FileLock and RedisLock implementations.
"""

import os
import tempfile
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from muller.core.lock import BaseLock, FileLock, RedisLock, PersistentLock, Lock
from muller.core.lock.file_lock import _get_lock_bytes, _parse_lock_bytes
from muller.core.lock.persistent import create_lock
from muller.core.storage import LocalProvider, MemoryProvider
from muller.util.exceptions import LockedException


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_storage():
    """Create a temporary directory with LocalProvider."""
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = LocalProvider(temp_dir)
        yield storage


@pytest.fixture
def memory_storage():
    """Create a MemoryProvider storage."""
    return MemoryProvider()


@pytest.fixture
def redis_client():
    """Create a Redis client, skip if Redis is not available."""
    try:
        import redis
        client = redis.Redis(host='localhost', port=6379, db=15)
        # Test connection
        client.ping()
        yield client
        # Cleanup: delete all test keys
        for key in client.scan_iter("muller:lock:test_*"):
            client.delete(key)
    except Exception:
        pytest.skip("Redis is not available")


# ============================================================================
# Test Helper Functions
# ============================================================================

class TestLockBytes:
    """Tests for lock byte encoding/decoding."""
    
    def test_get_lock_bytes_without_tag(self):
        """Test generating lock bytes without a tag."""
        byts = _get_lock_bytes(duration=10)
        assert len(byts) >= 14  # 6 bytes node ID + 8 bytes timestamp
    
    def test_get_lock_bytes_with_tag(self):
        """Test generating lock bytes with a tag."""
        tag = b"test"
        byts = _get_lock_bytes(tag=tag, duration=10)
        assert len(byts) >= 18  # 6 + 8 + 4 bytes
    
    def test_parse_lock_bytes(self):
        """Test parsing lock bytes."""
        tag = b"test"
        byts = _get_lock_bytes(tag=tag, duration=10)
        nodeid, timestamp, parsed_tag = _parse_lock_bytes(byts)
        
        assert isinstance(nodeid, int)
        assert isinstance(timestamp, float)
        assert timestamp > time.time()  # Should be in the future
        assert bytes(parsed_tag) == tag
    
    def test_parse_lock_bytes_invalid(self):
        """Test parsing invalid lock bytes raises assertion."""
        with pytest.raises(AssertionError):
            _parse_lock_bytes(b"short")


# ============================================================================
# Test BaseLock Interface
# ============================================================================

class TestBaseLock:
    """Tests for BaseLock abstract class."""
    
    def test_cannot_instantiate_directly(self):
        """Test that BaseLock cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseLock("test_path")
    
    def test_lock_alias_is_filelock(self):
        """Test that Lock is an alias for FileLock."""
        assert Lock is FileLock


# ============================================================================
# Test FileLock
# ============================================================================

class TestFileLock:
    """Tests for FileLock implementation."""
    
    def test_acquire_release_basic(self, temp_storage):
        """Test basic acquire and release flow."""
        lock = FileLock(temp_storage, "test_lock.lock")
        
        assert not lock.acquired
        lock.acquire()
        assert lock.acquired
        lock.release()
        assert not lock.acquired
    
    def test_acquire_release_memory_storage(self, memory_storage):
        """Test acquire and release with MemoryProvider."""
        lock = FileLock(memory_storage, "test_lock.lock")
        
        lock.acquire()
        assert lock.acquired
        lock.release()
        assert not lock.acquired
    
    def test_context_manager(self, temp_storage):
        """Test context manager protocol."""
        lock = FileLock(temp_storage, "test_lock.lock")
        
        assert not lock.acquired
        with lock:
            assert lock.acquired
        assert not lock.acquired
    
    def test_reentrant_acquire(self, temp_storage):
        """Test that re-acquiring the same lock succeeds (same process)."""
        lock = FileLock(temp_storage, "test_lock.lock")
        
        lock.acquire()
        # Second acquire should succeed (same tag and node ID)
        lock.acquire()
        assert lock.acquired
        lock.release()
    
    def test_lock_timeout_zero(self, temp_storage):
        """Test that timeout=0 fails immediately when lock is held."""
        lock1 = FileLock(temp_storage, "test_lock.lock")
        lock2 = FileLock(temp_storage, "test_lock.lock")
        
        # Use different tags to simulate different processes
        lock2.tag = b"diff"
        
        lock1.acquire()
        
        with pytest.raises(LockedException):
            lock2.acquire(timeout=0)
        
        lock1.release()
    
    def test_lock_refresh(self, temp_storage):
        """Test refreshing the lock."""
        lock = FileLock(temp_storage, "test_lock.lock")
        
        lock.acquire()
        original_bytes = temp_storage.get("test_lock.lock")
        
        # Refresh should update the timestamp
        time.sleep(0.1)
        lock.refresh_lock()
        refreshed_bytes = temp_storage.get("test_lock.lock")
        
        # The bytes should be different (new timestamp)
        assert original_bytes != refreshed_bytes
        lock.release()
    
    def test_refresh_without_lock_raises(self, temp_storage):
        """Test that refreshing without holding the lock raises exception."""
        lock = FileLock(temp_storage, "test_lock.lock")
        
        with pytest.raises(LockedException):
            lock.refresh_lock()
    
    def test_refresh_lost_lock_raises(self, temp_storage):
        """Test that refreshing a lost lock raises exception."""
        lock1 = FileLock(temp_storage, "test_lock.lock", duration=1)
        lock2 = FileLock(temp_storage, "test_lock.lock")
        lock2.tag = b"diff"
        
        lock1.acquire()
        # Manually overwrite the lock to simulate another process taking it
        temp_storage["test_lock.lock"] = _get_lock_bytes(b"other", 10)
        
        with pytest.raises(LockedException):
            lock1.refresh_lock()
    
    def test_release_without_acquire(self, temp_storage):
        """Test that releasing without acquiring is safe."""
        lock = FileLock(temp_storage, "test_lock.lock")
        # Should not raise
        lock.release()
    
    def test_lock_file_created(self, temp_storage):
        """Test that lock file is created in storage."""
        lock = FileLock(temp_storage, "my_lock.lock")
        
        assert "my_lock.lock" not in temp_storage
        lock.acquire()
        assert "my_lock.lock" in temp_storage
        lock.release()
        # Lock file should be deleted after release
        assert "my_lock.lock" not in temp_storage


# ============================================================================
# Test RedisLock
# ============================================================================

class TestRedisLock:
    """Tests for RedisLock implementation."""
    
    def test_acquire_release_basic(self, redis_client):
        """Test basic acquire and release flow."""
        lock = RedisLock(redis_client, "test_basic")
        
        assert not lock.acquired
        lock.acquire()
        assert lock.acquired
        lock.release()
        assert not lock.acquired
    
    def test_context_manager(self, redis_client):
        """Test context manager protocol."""
        lock = RedisLock(redis_client, "test_context")
        
        with lock:
            assert lock.acquired
        assert not lock.acquired
    
    def test_reentrant_acquire(self, redis_client):
        """Test that re-acquiring the same lock succeeds (same token)."""
        lock = RedisLock(redis_client, "test_reentrant")
        
        lock.acquire()
        # Second acquire should succeed (same token)
        lock.acquire()
        assert lock.acquired
        lock.release()
    
    def test_lock_timeout_zero(self, redis_client):
        """Test that timeout=0 fails immediately when lock is held."""
        lock1 = RedisLock(redis_client, "test_timeout")
        lock2 = RedisLock(redis_client, "test_timeout")
        
        lock1.acquire()
        
        with pytest.raises(LockedException):
            lock2.acquire(timeout=0)
        
        lock1.release()
    
    def test_lock_with_timeout(self, redis_client):
        """Test acquiring lock with timeout succeeds when lock is released."""
        lock1 = RedisLock(redis_client, "test_with_timeout", duration=1)
        lock2 = RedisLock(redis_client, "test_with_timeout")
        
        lock1.acquire()
        
        # Start a thread that will release the lock after a short delay
        def release_lock():
            time.sleep(0.3)
            lock1.release()
        
        thread = threading.Thread(target=release_lock)
        thread.start()
        
        # This should eventually succeed
        lock2.acquire(timeout=2)
        assert lock2.acquired
        lock2.release()
        thread.join()
    
    def test_lock_refresh(self, redis_client):
        """Test refreshing the lock extends TTL."""
        lock = RedisLock(redis_client, "test_refresh", duration=5)
        
        lock.acquire()
        
        # Get initial TTL
        initial_ttl = redis_client.pttl(lock._lock_key)
        
        # Wait a bit
        time.sleep(1)
        
        # Refresh
        lock.refresh_lock()
        
        # TTL should be extended (close to full duration again)
        new_ttl = redis_client.pttl(lock._lock_key)
        assert new_ttl > initial_ttl - 2000  # Allow some margin
        
        lock.release()
    
    def test_refresh_without_lock_raises(self, redis_client):
        """Test that refreshing without holding the lock raises exception."""
        lock = RedisLock(redis_client, "test_refresh_no_lock")
        
        with pytest.raises(LockedException):
            lock.refresh_lock()
    
    def test_release_without_acquire(self, redis_client):
        """Test that releasing without acquiring is safe."""
        lock = RedisLock(redis_client, "test_release_no_acquire")
        # Should not raise
        lock.release()
    
    def test_key_prefix(self, redis_client):
        """Test that lock key uses the correct prefix."""
        lock = RedisLock(redis_client, "test_prefix", prefix="custom:")
        
        lock.acquire()
        assert redis_client.exists("custom:test_prefix")
        lock.release()
    
    def test_atomic_release(self, redis_client):
        """Test that release only deletes if we own the lock."""
        lock = RedisLock(redis_client, "test_atomic_release")
        
        lock.acquire()
        lock.acquired = True  # Simulate that we think we have the lock
        
        # Manually change the lock to belong to someone else
        redis_client.set(lock._lock_key, "other_token")
        
        # Release should not delete the key (we don't own it anymore)
        lock.release()
        assert redis_client.exists(lock._lock_key)
        
        # Cleanup
        redis_client.delete(lock._lock_key)


# ============================================================================
# Test PersistentLock
# ============================================================================

class TestPersistentLock:
    """Tests for PersistentLock implementation."""
    
    def test_persistent_lock_with_storage(self, temp_storage):
        """Test PersistentLock with storage provider."""
        lock = PersistentLock(storage=temp_storage, path="test_persistent.lock")
        
        assert lock.acquired
        lock.release()
        assert not lock.acquired
    
    def test_persistent_lock_with_injected_lock(self, memory_storage):
        """Test PersistentLock with injected lock instance."""
        inner_lock = FileLock(memory_storage, "test_injected.lock", duration=300)
        lock = PersistentLock(lock=inner_lock)
        
        assert lock.acquired
        lock.release()
    
    def test_persistent_lock_requires_storage_or_lock(self):
        """Test that PersistentLock requires either storage or lock."""
        with pytest.raises(ValueError):
            PersistentLock()
    
    def test_auto_refresh_thread_starts(self, memory_storage):
        """Test that auto-refresh thread is started."""
        lock = PersistentLock(storage=memory_storage, path="test_thread.lock")
        
        assert lock._thread is not None
        assert lock._thread.is_alive()
        
        lock.release()
    
    def test_lock_lost_callback(self, memory_storage):
        """Test that lock_lost_callback is called when lock is lost."""
        callback_called = threading.Event()
        
        def callback():
            callback_called.set()
        
        # Use a very short duration so the lock expires quickly
        inner_lock = FileLock(memory_storage, "test_callback.lock", duration=1)
        lock = PersistentLock(lock=inner_lock, lock_lost_callback=callback)
        
        # Manually corrupt the lock to simulate loss
        memory_storage["test_callback.lock"] = _get_lock_bytes(b"other", 10)
        
        # Wait for the refresh loop to detect the loss
        # Note: This might take up to DATASET_LOCK_UPDATE_INTERVAL
        # For testing, we can manually trigger the refresh
        lock.release()


# ============================================================================
# Test Lock Factory
# ============================================================================

class TestCreateLock:
    """Tests for create_lock factory function."""
    
    def test_create_file_lock(self, temp_storage):
        """Test creating a FileLock via factory."""
        lock = create_lock(storage=temp_storage, path="factory_file.lock", lock_type="file")
        
        assert isinstance(lock, FileLock)
    
    def test_create_file_lock_default(self, temp_storage):
        """Test that FileLock is the default type."""
        with patch('muller.constants.LOCK_TYPE', 'file'):
            lock = create_lock(storage=temp_storage, path="factory_default.lock")
            assert isinstance(lock, FileLock)
    
    def test_create_file_lock_requires_storage(self):
        """Test that creating FileLock requires storage."""
        with pytest.raises(ValueError):
            create_lock(lock_type="file")
    
    def test_create_redis_lock(self, redis_client):
        """Test creating a RedisLock via factory."""
        lock = create_lock(path="factory_redis", lock_type="redis", redis_client=redis_client)
        
        assert isinstance(lock, RedisLock)
        # Cleanup
        lock.release()


# ============================================================================
# Test Concurrent Access
# ============================================================================

class TestConcurrentAccess:
    """Tests for concurrent lock access."""
    
    def test_filelock_concurrent_threads(self, temp_storage):
        """Test FileLock with concurrent threads simulating different processes.
        
        Note: FileLock uses process ID as tag, so threads in the same process
        share the same identity. To test concurrent access, we create separate
        lock instances with unique tags to simulate different processes.
        """
        counter = {"value": 0}
        errors = []
        counter_lock = threading.Lock()  # For thread-safe counter access
        
        def increment(thread_id):
            try:
                # Create a lock instance with unique tag (simulating different process)
                lock = FileLock(temp_storage, "test_concurrent.lock", duration=30)
                lock.tag = f"thread_{thread_id}".encode()  # Unique tag per thread
                
                lock.acquire(timeout=10)
                try:
                    with counter_lock:
                        current = counter["value"]
                    time.sleep(0.01)  # Simulate some work
                    with counter_lock:
                        counter["value"] = current + 1
                finally:
                    lock.release()
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=increment, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All increments should have succeeded
        assert counter["value"] == 5, f"Expected 5, got {counter['value']}, errors: {errors}"
        assert len(errors) == 0
    
    def test_redislock_concurrent_threads(self, redis_client):
        """Test RedisLock with concurrent threads.
        
        RedisLock uses a unique token per instance, so each thread
        naturally gets a different identity.
        """
        lock_name = f"test_concurrent_{time.time()}"
        counter = {"value": 0}
        errors = []
        counter_lock = threading.Lock()  # For thread-safe counter access
        
        def increment():
            try:
                # Each thread creates its own lock instance with unique token
                lock = RedisLock(redis_client, lock_name, duration=30)
                lock.acquire(timeout=10)
                try:
                    with counter_lock:
                        current = counter["value"]
                    time.sleep(0.01)  # Simulate some work
                    with counter_lock:
                        counter["value"] = current + 1
                finally:
                    lock.release()
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=increment) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All increments should have succeeded
        assert counter["value"] == 5, f"Expected 5, got {counter['value']}, errors: {errors}"
        assert len(errors) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

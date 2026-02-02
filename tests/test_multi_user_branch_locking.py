# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

"""
Tests for multi-user multi-branch locking mechanism.

This test module verifies that:
1. Different users can work on different branches concurrently without conflicts
2. Branch ownership is enforced correctly
3. Admin mode allows creators to override branch restrictions
4. Auto-commit works before checkout
5. Locks are properly managed during branch switches
"""

import pytest
import tempfile
import shutil
import os
from unittest.mock import patch

import muller
from muller.util.exceptions import UnAuthorizationError, LockedException


@pytest.fixture
def temp_dataset_path():
    """Create a temporary directory for test datasets."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_multi_user_different_branches(temp_dataset_path):
    """Test that two users can work on different branches without conflicts."""
    path = os.path.join(temp_dataset_path, "test_ds")
    
    # User A creates dataset and a branch
    with patch('muller.util.authorization.obtain_current_user', return_value='user_a'):
        ds_a = muller.empty(path)
        ds_a.create_tensor("data")
        ds_a.data.append([1, 2, 3])
        ds_a.commit("Initial commit")
        
        # Create branch for user A
        ds_a.checkout("branch_a", create=True)
        ds_a.data.append([4, 5, 6])
        ds_a.commit("User A changes")
    
    # User B loads dataset and creates their own branch
    with patch('muller.util.authorization.obtain_current_user', return_value='user_b'):
        ds_b = muller.load(path)
        ds_b.checkout("main")
        ds_b.checkout("branch_b", create=True)
        ds_b.data.append([7, 8, 9])
        ds_b.commit("User B changes")
    
    # Verify both branches exist and have correct data
    with patch('muller.util.authorization.obtain_current_user', return_value='user_a'):
        ds_verify = muller.load(path)
        ds_verify.checkout("branch_a")
        assert len(ds_verify.data) == 2  # Should have 2 samples
    
    with patch('muller.util.authorization.obtain_current_user', return_value='user_b'):
        ds_verify = muller.load(path)
        ds_verify.checkout("branch_b")
        assert len(ds_verify.data) == 2  # Should have 2 samples


def test_branch_ownership_enforcement(temp_dataset_path):
    """Test that users cannot modify branches owned by others."""
    path = os.path.join(temp_dataset_path, "test_ds")
    
    # User A creates dataset and branch
    with patch('muller.util.authorization.obtain_current_user', return_value='user_a'):
        ds_a = muller.empty(path)
        ds_a.create_tensor("data")
        ds_a.data.append([1, 2, 3])
        ds_a.commit("Initial commit")
        ds_a.checkout("branch_a", create=True)
        ds_a.commit("Branch A created")
    
    # User B tries to modify User A's branch
    with patch('muller.util.authorization.obtain_current_user', return_value='user_b'):
        ds_b = muller.load(path)
        ds_b.checkout("branch_a")
        
        # Should raise UnAuthorizationError
        with pytest.raises(UnAuthorizationError):
            ds_b.data.append([4, 5, 6])


def test_admin_mode_override(temp_dataset_path):
    """Test that dataset creator can use admin mode to modify any branch."""
    path = os.path.join(temp_dataset_path, "test_ds")
    
    # Creator (user_a) creates dataset
    with patch('muller.util.authorization.obtain_current_user', return_value='user_a'):
        ds_creator = muller.empty(path)
        ds_creator.create_tensor("data")
        ds_creator.data.append([1, 2, 3])
        ds_creator.commit("Initial commit")
    
    # User B creates their branch
    with patch('muller.util.authorization.obtain_current_user', return_value='user_b'):
        ds_b = muller.load(path)
        ds_b.checkout("branch_b", create=True)
        ds_b.data.append([4, 5, 6])
        ds_b.commit("User B changes")
    
    # Creator tries to modify User B's branch without admin mode
    with patch('muller.util.authorization.obtain_current_user', return_value='user_a'):
        ds_creator = muller.load(path)
        ds_creator.checkout("branch_b")
        
        # Should fail without admin mode
        with pytest.raises(UnAuthorizationError):
            ds_creator.data.append([7, 8, 9])
        
        # Enable admin mode
        ds_creator.enable_admin_mode()
        
        # Now should succeed
        ds_creator.data.append([7, 8, 9])
        ds_creator.commit("Admin fix")
        
        # Disable admin mode
        ds_creator.disable_admin_mode()
        
        # Should fail again after disabling
        with pytest.raises(UnAuthorizationError):
            ds_creator.data.append([10, 11, 12])


def test_auto_commit_before_checkout(temp_dataset_path):
    """Test that auto-commit works before checkout."""
    path = os.path.join(temp_dataset_path, "test_ds")
    
    with patch('muller.util.authorization.obtain_current_user', return_value='user_a'):
        ds = muller.empty(path)
        ds.create_tensor("data")
        ds.data.append([1, 2, 3])
        ds.commit("Initial commit")
        
        # Make uncommitted changes
        ds.data.append([4, 5, 6])
        
        # Checkout to new branch (should auto-commit)
        ds.checkout("new_branch", create=True)
        
        # Go back to main and verify the auto-commit was made
        ds.checkout("main")
        assert len(ds.data) == 2  # Should have both samples


def test_same_user_multiple_branches(temp_dataset_path):
    """Test that the same user can work on multiple branches."""
    path = os.path.join(temp_dataset_path, "test_ds")
    
    with patch('muller.util.authorization.obtain_current_user', return_value='user_a'):
        # Create dataset
        ds = muller.empty(path)
        ds.create_tensor("exp_results")
        ds.exp_results.append([1.0])
        ds.commit("Initial")
        
        # Create experiment branch 1
        ds.checkout("exp1", create=True)
        ds.exp_results.append([2.0])
        ds.commit("Exp 1 results")
        
        # Go back to main and create experiment branch 2
        ds.checkout("main")
        ds.checkout("exp2", create=True)
        ds.exp_results.append([3.0])
        ds.commit("Exp 2 results")
        
        # Verify both branches work
        ds.checkout("exp1")
        assert ds.exp_results.numpy()[-1] == [2.0]
        
        ds.checkout("exp2")
        assert ds.exp_results.numpy()[-1] == [3.0]


def test_branch_metadata_storage(temp_dataset_path):
    """Test that branch metadata (owner) is properly stored and retrieved."""
    path = os.path.join(temp_dataset_path, "test_ds")
    
    with patch('muller.util.authorization.obtain_current_user', return_value='user_a'):
        ds = muller.empty(path)
        ds.create_tensor("data")
        ds.commit("Initial")
        ds.checkout("branch_a", create=True)
    
    # Verify metadata can be retrieved
    with patch('muller.util.authorization.obtain_current_user', return_value='user_b'):
        ds = muller.load(path)
        from muller.util.version_control import get_branch_owner
        owner = get_branch_owner(ds, "branch_a")
        assert owner == "user_a"


def test_non_creator_cannot_enable_admin_mode(temp_dataset_path):
    """Test that non-creator users cannot enable admin mode."""
    path = os.path.join(temp_dataset_path, "test_ds")
    
    # Creator creates dataset
    with patch('muller.util.authorization.obtain_current_user', return_value='user_a'):
        ds = muller.empty(path)
        ds.create_tensor("data")
        ds.commit("Initial")
    
    # Non-creator tries to enable admin mode
    with patch('muller.util.authorization.obtain_current_user', return_value='user_b'):
        ds = muller.load(path)
        
        with pytest.raises(UnAuthorizationError):
            ds.enable_admin_mode()


def test_lock_isolation_different_branches(temp_dataset_path):
    """Test that locks on different branches don't conflict."""
    path = os.path.join(temp_dataset_path, "test_ds")
    
    # Setup: Create dataset with two branches
    with patch('muller.util.authorization.obtain_current_user', return_value='user_a'):
        ds = muller.empty(path)
        ds.create_tensor("data")
        ds.commit("Initial")
        ds.checkout("branch_a", create=True)
    
    with patch('muller.util.authorization.obtain_current_user', return_value='user_b'):
        ds = muller.load(path)
        ds.checkout("main")
        ds.checkout("branch_b", create=True)
    
    # Simulate concurrent access (in single-threaded test, just verify no errors)
    with patch('muller.util.authorization.obtain_current_user', return_value='user_a'):
        ds_a = muller.load(path)
        ds_a.checkout("branch_a")
        # Lock is acquired for branch_a
    
    with patch('muller.util.authorization.obtain_current_user', return_value='user_b'):
        ds_b = muller.load(path)
        ds_b.checkout("branch_b")
        # Should not conflict with branch_a lock
        ds_b.data.append([1, 2, 3])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

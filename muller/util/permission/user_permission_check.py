# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import json
from functools import wraps
from typing import Callable

import muller
from muller.constants import VERSION_CONTROL_INFO_FILENAME, DATASET_META_FILENAME, QUERIES_FILENAME
from muller.util.authorization import obtain_current_user
from muller.util.exceptions import UnAuthorizationError


def _get_view_creator_name(dataset, view_id: str = None):
    """ get dataset view creator name. """
    qjson = json.loads(dataset.base_storage[QUERIES_FILENAME].decode("utf-8").replace("'", '"'))
    for _, q in enumerate(qjson):
        if q["id"] == view_id:
            target_view_meta_file = ".queries/" + (
                    q.get("path") or q["id"]) + "/" + DATASET_META_FILENAME
            dataset_view_info = json.loads(
                dataset.storage.next_storage[target_view_meta_file].decode('utf8').replace("'", '"'))
            target_user_name = dataset_view_info["info"]["uid"]
            return target_user_name
        return None


def user_permission_check(func: Callable):
    """Enhanced user permission check with branch-level isolation and admin mode."""
    @wraps(func)
    def inner(x, *args, **kwargs):
        ds = x if isinstance(x, muller.Dataset) else x.dataset
        current_user_name = obtain_current_user()

        # Get dataset creator
        try:
            dataset_creator_name = x.version_state["meta"].dataset_creator
        except (TypeError, KeyError):
            try:
                dataset_creator_name = x.obtain_dataset_creator_name_from_storage()
            except Exception:
                dataset_creator_name = None

        # Check if in admin mode
        is_admin_mode = getattr(ds, '_admin_mode', False)
        is_creator = dataset_creator_name and current_user_name == dataset_creator_name
        
        # Dataset creator in admin mode has full access
        if is_creator and is_admin_mode:
            return func(x, *args, **kwargs)
        
        # If REQUIRE_ADMIN_MODE is False, creator has automatic full access (backward compatibility)
        if is_creator and not muller.constants.REQUIRE_ADMIN_MODE:
            return func(x, *args, **kwargs)
        
        # Dataset creator without admin mode (when REQUIRE_ADMIN_MODE=True): check branch ownership like regular users
        
        def get_target_user_name(dataset, branch_name:str = None):
            if not dataset.version_state:
                version_state = json.loads(ds.storage.next_storage[VERSION_CONTROL_INFO_FILENAME]
                                           .decode('utf8').replace("'", '"'))
                if not branch_name: # not delete_branch situation
                    branch_name = ds.branch
                current_id = version_state['branches'][branch_name]
                target_user_name = version_state['commits'][current_id]['commit_user_name']
            else:
                # version state load from memory
                version_state = dataset.version_state
                if branch_name: # delete_branch situation
                    try:
                        target_id = version_state["branch_commit_map"][branch_name]
                    except Exception as e:
                        raise Exception from e
                    target_user_name = version_state["commit_node_map"][target_id].commit_user_name
                else:
                    current_id = version_state['commit_id']
                    target_user_name = version_state["commit_node_map"][current_id].commit_user_name
            return target_user_name

        # Check branch ownership for write operations
        write_operations = {
            "commit", "protected_commit", "append", "extend", 
            "update", "_update", "pop", "clear", "__setitem__", 
            "create_tensor", "create_tensor_like", "delete_tensor", "rename_tensor",
            "rechunk", "add_data_from_file", "add_data_from_dataframes",
            "create_index", "create_vector_index", "merge", "reset"
        }
        
        if func.__name__ in write_operations:
            current_branch = ds.version_state.get("branch", "main")
            
            # Try to get branch owner from metadata first, fallback to commit history
            try:
                from muller.util.version_control import get_branch_owner
                branch_owner = get_branch_owner(ds, current_branch)
            except Exception:
                branch_owner = get_target_user_name(ds, None)
            
            # If no branch owner (old dataset or single-user), allow the operation
            if branch_owner is None:
                return func(x, *args, **kwargs)
            
            # Allow if user owns the branch
            if current_user_name == branch_owner:
                return func(x, *args, **kwargs)
            
            # Deny if branch owned by someone else
            if is_creator:
                raise UnAuthorizationError(
                    f"User [{current_user_name}] (dataset creator) cannot modify branch "
                    f"[{current_branch}] owned by [{branch_owner}]. "
                    f"Use ds.enable_admin_mode() to override."
                )
            else:
                raise UnAuthorizationError(
                    f"User [{current_user_name}] is not allowed to modify branch "
                    f"[{current_branch}] owned by [{branch_owner}]. "
                    f"Please checkout your own branch."
                )
        
        # Special handling for dataset-level operations
        if func.__name__ in {"delete", "rename"}:
            if not is_creator:
                raise UnAuthorizationError(
                    f"User [{current_user_name}] is not allowed to "
                    f"delete the dataset. Only [{dataset_creator_name}] is allowed to do it."
                )
            return func(x, *args, **kwargs)
        
        if func.__name__ == "delete_view":
            target_user_name = _get_view_creator_name(ds, args[0])
        elif func.__name__ == "delete_branch":
            target_user_name = get_target_user_name(ds, args[0])
        else:
            target_user_name = get_target_user_name(ds, None)

        if target_user_name and current_user_name == target_user_name:
            return func(x, *args, **kwargs)
        
        # Fallback: deny access
        raise UnAuthorizationError(
            f"User [{current_user_name}] is not allowed to access the Func: {func.__name__} "
            f"which is allowed by dataset creator: [{dataset_creator_name}] "
            f"and branch owner [{target_user_name}]."
        )
    return inner

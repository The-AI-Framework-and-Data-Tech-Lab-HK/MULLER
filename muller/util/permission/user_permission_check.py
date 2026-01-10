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
    """Function to check user permission."""
    @wraps(func)
    def inner(x, *args, **kwargs):
        ds = x if isinstance(x, muller.Dataset) else x.dataset
        current_user_name = obtain_current_user()

        try:
            dataset_creator_name = x.version_state["meta"].dataset_creator
        except (TypeError, KeyError):
            dataset_creator_name = x.obtain_dataset_creator_name_from_storage()

        if dataset_creator_name and current_user_name == dataset_creator_name:
            return func(x, *args, **kwargs)

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

        if func.__name__ in {"delete", "rename"}:
            raise UnAuthorizationError(f"User [{current_user_name}] is not allowed to "
                                       f"delete the dataset only [{dataset_creator_name}] is allowed to do it.")
        if func.__name__ == "delete_view":
            target_user_name = _get_view_creator_name(ds, args[0])
        elif func.__name__ == "delete_branch":
            target_user_name = get_target_user_name(ds, args[0])
        else:
            target_user_name = get_target_user_name(ds, None)

        if target_user_name and current_user_name == target_user_name:
            return func(x, *args, **kwargs)

        raise UnAuthorizationError(f"User [{current_user_name}] is not allowed to access the Func: {func.__name__} "
                                   f"which is allowed by dataset creator: [{dataset_creator_name}] "
                                   f"and branch owner [{target_user_name}].")
    return inner

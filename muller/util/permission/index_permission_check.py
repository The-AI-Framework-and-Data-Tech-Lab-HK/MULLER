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

from muller.client.log import logger
from muller.constants import VERSION_CONTROL_INFO_FILENAME
from muller.util.authorization import obtain_current_user


def index_permission_check(func: Callable):
    """Function to check user permission."""
    @wraps(func)
    def inner(x, *args, **kwargs):
        def obtain_target_user_name_from_storage():
            version_state_storage = json.loads(ds.storage.next_storage[VERSION_CONTROL_INFO_FILENAME]
                                               .decode('utf8').replace("'", '"'))
            try:
                commit_id = next(iter(version_state_storage["branches"].values()))
                target_user_name = version_state_storage["commits"][commit_id]["commit_user_name"]
            except (TypeError, KeyError):
                target_user_name = None
            return target_user_name

        current_user_name = obtain_current_user()
        ds = x
        if ds.version_state:
            try:
                branch = ds.version_state["branch"]
                commit_id = ds.version_state["branch_commit_map"][branch]
                ds.version_state["commits"]
                target_user_name = ds.version_state["commit_node_map"][commit_id].commit_user_name
            except (TypeError, KeyError):
                target_user_name = obtain_target_user_name_from_storage()
        else:
            target_user_name = obtain_target_user_name_from_storage()

        if target_user_name != current_user_name:
            logger.info(f"The index was created by {target_user_name}, it might not match for user {current_user_name}")

        return func(x, *args, **kwargs)

    return inner

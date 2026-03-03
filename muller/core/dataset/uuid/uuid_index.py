# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import os
from muller.constants import FIRST_COMMIT_ID, DATASET_UUID_NAME
from muller.core.dataset.uuid.shard_hash import divide_to_shard, load_all_shards


def create_uuid_index(dataset):
    """Create uuid and index pair and stored in the disk.

    Args:
        dataset: The dataset to create UUID index for.

    Raises:
        KeyError: If commit_id is not found in version_state.
        ValueError: If current_id is not FIRST_COMMIT_ID.
    """
    try:
        current_id = dataset.version_state['commit_id']
    except KeyError as e:
        raise KeyError from e
    if current_id != FIRST_COMMIT_ID:
        raise ValueError
    uuids = dataset.get_tensor_uuids(DATASET_UUID_NAME, current_id)
    divide_to_shard(path=os.path.join(dataset.path, DATASET_UUID_NAME), uuids=uuids)


def load_uuid_index(dataset):
    """Load all uuid indexes from shards.

    Args:
        dataset: The dataset to load UUID index from.

    Returns:
        Loaded UUID index data.

    Raises:
        KeyError: If commit_id is not found in version_state.
        ValueError: If current_id is not FIRST_COMMIT_ID.
    """
    try:
        current_id = dataset.version_state['commit_id']
    except KeyError as e:
        raise KeyError from e
    if current_id != FIRST_COMMIT_ID:
        raise ValueError
    return load_all_shards(path=os.path.join(dataset.path, DATASET_UUID_NAME))

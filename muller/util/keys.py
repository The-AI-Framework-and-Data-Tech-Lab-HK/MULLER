# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
Deprecated: This module has been moved to muller.core.storage_keys

This module is kept for backward compatibility and will be removed in a future version.
Please update your imports to use muller.core.storage_keys instead.
"""

import warnings

# Re-export all functions from the new location
from muller.core.storage_keys import (  # noqa: F401
    dataset_exists,
    filter_name,
    get_chunk_id_encoder_key,
    get_chunk_key,
    get_commit_info_key,
    get_creds_encoder_key,
    get_dataset_diff_key,
    get_dataset_lock_key,
    get_dataset_meta_key,
    get_downsampled_tensor_key,
    get_queries_key,
    get_queries_lock_key,
    get_sample_id_tensor_key,
    get_sample_info_tensor_key,
    get_sample_shape_tensor_key,
    get_tensor_commit_chunk_map_key,
    get_tensor_commit_diff_key,
    get_tensor_meta_key,
    get_tensor_tile_encoder_key,
    get_version_control_info_key,
    get_version_control_info_lock_key,
    tensor_exists,
)

warnings.warn(
    "muller.util.keys is deprecated and will be removed in a future version. "
    "Please use muller.core.storage_keys instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    'dataset_exists',
    'filter_name',
    'get_chunk_id_encoder_key',
    'get_chunk_key',
    'get_commit_info_key',
    'get_creds_encoder_key',
    'get_dataset_diff_key',
    'get_dataset_lock_key',
    'get_dataset_meta_key',
    'get_downsampled_tensor_key',
    'get_queries_key',
    'get_queries_lock_key',
    'get_sample_id_tensor_key',
    'get_sample_info_tensor_key',
    'get_sample_shape_tensor_key',
    'get_tensor_commit_chunk_map_key',
    'get_tensor_commit_diff_key',
    'get_tensor_meta_key',
    'get_tensor_tile_encoder_key',
    'get_version_control_info_key',
    'get_version_control_info_lock_key',
    'tensor_exists',
]

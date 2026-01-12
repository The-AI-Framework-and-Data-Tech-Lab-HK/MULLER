# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import keyword

muller_kwlist = [
    'tensors',  # The attributes of TransformDataset
    'data',
    'embedding',
    'all_chunk_engines',
    'cache_size',
    'cache_used',
    'idx',
    'pg_callback',
    'start_input_idx',
    '_client', # The attribute of Dataset
    'path',
    'storage',
    '_read_only_error',
    'base_storage',
    '_read_only',
    '_locked_out',
    'is_iteration',
    'is_first_load',
    '_is_filtered_view',
    'index',
    'version_state',
    '_locking_enabled',
    '_lock_timeout',
    'temp_tensors',
    'public',
    'verbose',
    '_vc_info_updated',
    '_info',
    '_ds_diff',
    'enabled_tensors',
    '_view_id',
    '_view_base',
    '_view_use_parent_commit',
    '_parent_dataset',
    '_query_string',
    '_inverted_index',
    'filtered_index',
    'split_tensor_meta',
    'creds',
    '_vector_index',
    'append_only',
    'initial_autoflush',
    '_indexing_history',
    'read_only',
    'is_first_load'
    ]


def is_muller_keyword(tensor_name):
    """Validate whether a tensor name is muller keyword or attribute."""
    if keyword.iskeyword(tensor_name) or tensor_name in dir(__builtins__) or tensor_name in muller_kwlist:
        return True
    return False

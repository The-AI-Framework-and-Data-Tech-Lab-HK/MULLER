# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

"""MULLER API - organized by functionality."""

# Dataset operations
from muller.api.dataset import (
    dataset,
    load,
    empty,
    delete,
    get_col_info,
    like,
    from_file,
    from_dataframes,
)

# I/O operations
from muller.api.io import read, tiled, Sample

# Transform operations
from muller.api.transform import compute, ComputeFunction, Pipeline

# Legacy compatibility - keep DatasetAPI for backward compatibility
from muller.api.dataset_api import DatasetAPI

__all__ = [
    # Dataset operations
    'dataset',
    'load',
    'empty',
    'delete',
    'get_col_info',
    'like',
    'from_file',
    'from_dataframes',
    # I/O operations
    'read',
    'tiled',
    'Sample',
    # Transform operations
    'compute',
    'ComputeFunction',
    'Pipeline',
    # Legacy
    'DatasetAPI',
]

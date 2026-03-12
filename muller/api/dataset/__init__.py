# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

"""Dataset API module - organized by functionality."""

from muller.api.dataset.core import (
    dataset,
    load,
    empty,
    delete,
    get_col_info,
)
from muller.api.dataset.copy import like
from muller.api.dataset.import_data import (
    from_file,
    from_dataframes,
    from_csv,
)

__all__ = [
    # Core operations
    'dataset',
    'load',
    'empty',
    'delete',
    'get_col_info',
    # Copy operations
    'like',
    # Import operations
    'from_file',
    'from_dataframes',
    'from_csv',
]

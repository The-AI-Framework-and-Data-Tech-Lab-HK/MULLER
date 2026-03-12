# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/__init__.py
#
# Modifications Copyright (c) 2026 Xueling Lin

__all__ = [
    "read",
    "load",
    "dataset",
    "Dataset",
    "__version__",
    "delete",
    "compute",
    "get_col_info",
    "empty",
    "like",
    "tiled",
    "Sample",
    "from_file",
    "from_dataframes",
    "from_csv",
]


import sys

import muller.constants
from muller._version import __version__
from muller.compression import SUPPORTED_COMPRESSIONS
from muller.core.dataset import Dataset
from muller.core.tensor import Tensor
from muller.core.types.htype import HTYPE_CONFIGURATIONS

# Import from new API structure
from muller.api.dataset import (
    dataset,
    load,
    empty,
    like,
    delete,
    get_col_info,
    from_file,
    from_dataframes,
    from_csv,
)
from muller.api.io import read, tiled, Sample
from muller.api.transform import compute

compressions = list(SUPPORTED_COMPRESSIONS)

if sys.version_info < (3, 11):
    raise RuntimeError("Python version 3.11 or higher is required for this project.")

# The api of muller tensor
tensor = Tensor

import muller.api

# Deprecated: create_dataset_from_file and create_dataset_from_dataframes
# These are now available as from_file and from_dataframes
create_dataset_from_file = from_file
create_dataset_from_dataframes = from_dataframes

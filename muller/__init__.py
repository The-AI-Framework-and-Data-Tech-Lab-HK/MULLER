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
    "get_col_info"
    ]

__version__ = "0.7.0"


import sys

import muller.constants
from muller.api.read import read
from muller.api.tiled import tiled
from muller.compression import SUPPORTED_COMPRESSIONS
from muller.core.dataset import Dataset
from muller.core.tensor import Tensor
from muller.core.transform import compute
from muller.htype import HTYPE_CONFIGURATIONS

compressions = list(SUPPORTED_COMPRESSIONS)

if sys.version_info < (3, 11):
    raise RuntimeError("Python version 3.11 or higher is required for this project.")

# The api of muller dataset
load = muller.api.DatasetAPI.load
empty = muller.api.DatasetAPI.empty
dataset = muller.api.DatasetAPI.dataset
like = muller.api.DatasetAPI.like
delete = muller.api.DatasetAPI.delete
get_col_info = muller.api.DatasetAPI.get_col_info
create_dataset_from_file = muller.api.DatasetAPI.create_dataset_from_file
create_dataset_from_dataframes = muller.api.DatasetAPI.create_dataset_from_dataframes


# The api of muller tensor
tensor = Tensor

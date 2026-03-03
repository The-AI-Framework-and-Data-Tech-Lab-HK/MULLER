# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

"""Data I/O API - reading files, creating tiled samples, and Sample class."""

from typing import Optional, Tuple, Union

import numpy as np

# Re-export read function
from muller.api.read import read

# Re-export tiled function
from muller.api.tiled import tiled

# Re-export Sample class
from muller.core.sample import Sample

__all__ = ['read', 'tiled', 'Sample']

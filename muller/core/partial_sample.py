# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/core/partial_sample.py
#
# Modifications Copyright (c) 2026 Xueling Lin

from typing import Tuple, Optional, Union
import numpy as np


class PartialSample:
    """Represents a sample that is initialized by just shape and the data is updated later."""

    def __init__(
        self,
        sample_shape: Tuple[int, ...],
        tile_shape: Optional[Tuple[int, ...]] = None,
        dtype: Optional[Union[str, np.dtype]] = np.dtype("uint8"),
    ):
        self.sample_shape = sample_shape
        self.tile_shape = tile_shape
        self.dtype = dtype

    @property
    def shape(self):
        return self.sample_shape

    def astype(self, dtype: Union[str, np.dtype]):
        return self.__class__(self.sample_shape, self.tile_shape, dtype)

    def downsample(self, factor: int):
        shape = (
            self.sample_shape[0] // factor,
            self.sample_shape[1] // factor,
        ) + self.sample_shape[2:]
        return self.__class__(
            shape,
            self.tile_shape,
            self.dtype,
        )

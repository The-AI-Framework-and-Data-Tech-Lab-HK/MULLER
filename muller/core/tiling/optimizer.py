# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/core/tiling/optimizer.py
#
# Modifications Copyright (c) 2026 Bingyu Liu

from typing import Tuple, Union, List, Optional

import numpy as np


def get_tile_shape(
    sample_shape: Tuple[int, ...],
    sample_size: Optional[float] = None,
    chunk_size: int = 16 * 2**20,
    exclude_axes: Optional[Union[int, List[int]]] = None,
) -> Tuple[int, ...]:
    """
    Get tile shape for a given sample shape that will fit in chunk_size
    :param sample_shape: Shape of the sample.
    :param sample_size: Size of the compressed sample in bytes.
    :param chunk_size: Expected size of a compressed tile in bytes.
    :param exclude_axes: Dimensions to be excluded from tiling. (2 for RGB images).
    :return: Tile shape
    """
    ratio = sample_size / chunk_size  # type: ignore
    sample_shape = np.array(sample_shape, dtype=np.float32)  # type: ignore
    if isinstance(exclude_axes, int):
        exclude_axes = [exclude_axes]
    elif exclude_axes is None:
        exclude_axes = []
    elif not isinstance(exclude_axes, list):
        exclude_axes = list(exclude_axes)
    sample_shape_masked = sample_shape.copy()  # type: ignore
    sample_shape_masked[exclude_axes] = 0
    while ratio > 1:
        idx = np.argmax(sample_shape_masked)
        val = sample_shape_masked[idx : idx + 1]  # type: ignore
        if val < 2:
            raise ValueError(f"Chunk size is too small: {chunk_size} bytes")
        val /= 2
        ratio /= 2
    sample_shape_masked[exclude_axes] = sample_shape[exclude_axes]  # type: ignore
    arr = np.ceil(sample_shape_masked)
    return tuple(int(x) for x in arr)

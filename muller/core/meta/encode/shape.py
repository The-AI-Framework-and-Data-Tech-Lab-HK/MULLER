# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/core/meta/encode/shape.py
#
# Modifications Copyright (c) 2026 Bingyu Liu

from typing import Tuple

import numpy as np

from muller.core.meta.encode.base_encoder import Encoder, LAST_SEEN_INDEX_COLUMN


class ShapeEncoder(Encoder):

    @property
    def dimensionality(self) -> int:
        """Function to get the dimensionality."""
        return len(self[0])

    def _combine_condition(
        self, item: Tuple[int], compare_row_index: int = -1
    ) -> bool:
        last_shape = self._derive_value(self._encoded[compare_row_index])

        return item == last_shape

    def _derive_value(self, row: np.ndarray, *_) -> Tuple:  # type: ignore
        return tuple(row[:LAST_SEEN_INDEX_COLUMN])

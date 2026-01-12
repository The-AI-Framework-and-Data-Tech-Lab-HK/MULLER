# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/core/meta/encode/byte_positions.py
#
# Modifications Copyright (c) 2026 Bingyu Liu

from typing import Optional, Sequence

import numpy as np

from muller.core.meta.encode.base_encoder import Encoder, LAST_SEEN_INDEX_COLUMN

NUM_BYTES_COLUMN = 0
START_BYTE_COLUMN = 1


class BytePositionsEncoder(Encoder):
    def __init__(self, *args, **kwargs):
        self._num_columns = 3
        super(BytePositionsEncoder, self).__init__(*args, **kwargs)

    def get_sum_of_bytes(self, until_row_index: int = -1) -> int:
        """
        Get the total number of bytes that are accounted for.
        This operation is O(1).

        Args:
            until_row_index (int): Optionally provide a row index for which the sum will end.
                Gets all bytes accounted for up until `until_row_index`.

        Returns:
            int: Number of bytes encoded at and below `until_row_index`.
        """

        if len(self._encoded) == 0:
            return 0

        if until_row_index < 0:
            until_row_index = len(self._encoded) + until_row_index

        last_last_seen_index = 0
        if until_row_index > 0:
            last_last_seen_index = self._encoded[
                until_row_index - 1, LAST_SEEN_INDEX_COLUMN
            ]

        row = self._encoded[until_row_index]
        start_byte = row[START_BYTE_COLUMN]
        num_bytes = row[NUM_BYTES_COLUMN]
        delta = int(row[LAST_SEEN_INDEX_COLUMN]) - int(last_last_seen_index)

        if until_row_index == 0:
            delta += 1

        return int(start_byte + (num_bytes * delta))

    def post_process_state(self, start_row_index: int):
        """
        Starting at `start_row_index`, move downwards through `self._encoded` and update all start bytes
        for each row if applicable. Used for updating.
        """

        for row_index in range(start_row_index, len(self._encoded)):
            if row_index == 0:
                bytes_under_row = 0
            else:
                bytes_under_row = self.get_sum_of_bytes(row_index - 1)
            self._encoded[row_index, START_BYTE_COLUMN] = bytes_under_row

    def pop(self, index: Optional[int] = None):
        """Pop samples."""
        if index is None:
            index = self.get_last_index_for_pop()
        (sb, eb), row = self.__getitem__(index, return_row_index=True)  # type: ignore
        prev = -1 if row == 0 else self._encoded[row - 1, LAST_SEEN_INDEX_COLUMN]
        num_samples_in_row = self._encoded[row, LAST_SEEN_INDEX_COLUMN] - prev \
            if prev != -1 else self._encoded[row, LAST_SEEN_INDEX_COLUMN] + np.abs(prev)
        num_bytes = eb - sb
        if num_samples_in_row == 0:
            raise ValueError("No samples to pop")
        self._encoded[row:, LAST_SEEN_INDEX_COLUMN] -= 1
        self._encoded[row + 1 :, START_BYTE_COLUMN] -= num_bytes

        # after subtracting 1, the row is now empty
        if num_samples_in_row == 1:
            self._encoded = np.delete(self._encoded, row, axis=0)

    def _combine_condition(self, item: int, compare_row_index: int = -1) -> bool:
        """Checks if `num_bytes` matches the `num_bytes` represented at row with index `compare_row_index`."""

        last_num_bytes = self._encoded[compare_row_index, NUM_BYTES_COLUMN]
        return item == last_num_bytes

    def _derive_value(self, row: np.ndarray, row_index: int, local_sample_index: int):
        """Derives the values of byte position."""
        index_bias = 0
        if row_index >= 1:
            index_bias = self._encoded[row_index - 1][LAST_SEEN_INDEX_COLUMN] + 1

        start_byte = row[START_BYTE_COLUMN] + (local_sample_index - index_bias) * row[NUM_BYTES_COLUMN]
        end_byte = start_byte + row[NUM_BYTES_COLUMN]
        return int(start_byte), int(end_byte)

    def _make_decomposable(
        self, item: int, compare_row_index: int = -1
    ) -> Sequence:
        """
        Used for updating. Return value is a sequence representing the row
        that can be decomposed using the `*` operator.
        """

        start_byte = self.get_sum_of_bytes(compare_row_index)
        return [item, start_byte]

# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/core/partial_reader.py
#
# Modifications Copyright (c) 2026 Xueling Lin

from typing import Dict, Tuple


class PartialReader:
    def __init__(self, cache, path: str, header_offset: int):
        self.cache = cache
        self.path = path
        self.data_fetched: Dict[Tuple[int, int], memoryview] = {}
        self.header_offset = header_offset

    def __getitem__(self, slice_: slice) -> memoryview:
        start = slice_.start + self.header_offset
        stop = slice_.stop + self.header_offset
        step = slice_.step
        assert start is not None and stop is not None
        assert step is None or step == 1
        slice_tuple = (start, stop)
        if slice_tuple not in self.data_fetched:
            self.data_fetched[slice_tuple] = memoryview(
                self.cache.get_bytes(self.path, start, stop)
            )
        return self.data_fetched[slice_tuple]

    def get_all_bytes(self) -> bytes:
        return self.cache.next_storage[self.path]

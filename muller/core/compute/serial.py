# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/core/compute/serial.py
#
# Modifications Copyright (c) 2026 Bingyu Liu

from muller.core.compute.provider import ComputeProvider, get_progress_bar


class SerialProvider(ComputeProvider):
    def __init__(self, workers):
        super().__init__(workers)

    def map(self, func, iterable):
        return list(map(func, iterable))

    def map_with_progress_bar(
        self,
        func,
        iterable,
        total_length: int,
        desc=None,
        pbar=None,
        pqueue=None,
    ):
        progress_bar = pbar or get_progress_bar(total_length, desc)

        def sub_func(*args, **kwargs):
            def pg_callback(value: int):
                progress_bar.update(value)

            return func(pg_callback, *args, **kwargs)

        result = self.map(sub_func, iterable)

        return result

    def create_queue(self):
        return None

    def close(self):
        return

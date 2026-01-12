# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/core/compute/process.py
#
# Modifications Copyright (c) 2026 Bingyu Liu

import ctypes
import ctypes.util
import gc
import warnings

from multiprocess import Manager as _Manager
from multiprocess.pool import Pool as ProcessPool

from muller.core.compute.provider import ComputeProvider


class ProcessProvider(ComputeProvider):
    def __init__(self, workers):
        super().__init__(workers)
        self.pool = ProcessPool(processes=workers)
        self.manager = _Manager()
        self._closed = False

    def __del__(self):
        if not self._closed:
            self.close()
            warnings.warn(
                "process pool thread leak. check compute provider is closed after use"
            )

    def create_queue(self):
        return self.manager.Queue()

    def close(self):
        self.pool.close()
        self.pool.join()
        self.pool.terminate()
        if self.manager:
            self.manager.shutdown()
            self.manager = None
        self.shrink_heap()
        self._closed = True

    def map(self, func, iterable):
        return self.pool.map(func, iterable)

    def shrink_heap(self):
        """Function to shrink heap."""
        gc.collect()
        libc = ctypes.CDLL(ctypes.util.find_library("c"))
        libc.malloc_trim(0)

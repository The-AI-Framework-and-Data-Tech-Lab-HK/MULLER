# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import multiprocessing as mp
from multiprocessing import Manager

from muller.core.compute.provider import ComputeProvider


class MPProcessProvider(ComputeProvider):
    def __init__(self, workers):
        super().__init__(workers)
        self.workers = workers
        self.pool = mp.Pool(processes=workers)
        self.manager = Manager()
        self._closed = False

    def map(self, func, iterable):
        return self.pool.map(func, iterable)

    def create_queue(self):
        return self.manager.Queue()

    def close(self):
        self.pool.terminate()
        self.pool.join()
        if self.manager:
            self.manager.shutdown()
            self.manager = None
        self._closed = True

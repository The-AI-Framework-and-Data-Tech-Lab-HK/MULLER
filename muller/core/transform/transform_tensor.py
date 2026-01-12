# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/core/transform/transform_tensor.py
#
# Modifications Copyright (c) 2026 Bingyu Liu

import bisect
import posixpath
from itertools import chain

import numpy as np

from muller.util.exceptions import SampleAppendError
from muller.constants import DATASET_UUID_NAME


class TransformTensor:
    def __init__(self, dataset, name, is_batch=False):
        self.items = []
        self.dataset = dataset
        self.name = name
        self.idx = slice(None, None, None)
        self.numpy_only = True
        self.cum_sizes = []
        self.is_batch = is_batch

    def __len__(self):
        if self.numpy_only:
            return 0 if not self.cum_sizes else self.cum_sizes[-1]
        return len(self.items)

    def __getattr__(self, item):
        return self.dataset[posixpath.join(self.name, item)][self.idx]

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.__getattr__(item)
        self.idx = item # slice(None, None, None)
        return self

    def numpy_compressed(self):
        """Numpy compressed transform"""
        if self.numpy_only:
            return self._numpy_only_data()

        idx = self.idx
        return self.items[idx]

    def non_numpy_only(self):
        """Non-numpy transform"""
        if self.numpy_only:
            items = list(chain(*self.items[:]))
            self.items.clear()
            self.items += items
            self.cum_sizes.clear()
            self.numpy_only = False

    def append(self, item):
        """Adds an item to the tensor."""
        try:
            # optimization applicable only if extending
            self.non_numpy_only()

            if self.is_batch:
                updated_tensor = []
                try:
                    updated_tensor = 0
                    chunk_engine = self.dataset.all_chunk_engines[self.name]
                    if self.numpy_only:
                        for sample in item:
                            chunk_engine.extend(
                                sample,
                                pg_callback=self.dataset.pg_callback,
                            )
                            updated_tensor += len(sample)
                        self.items.clear()
                    else:
                        chunk_engine.extend(
                            item,
                            pg_callback=self.dataset.pg_callback,
                            is_uuid=(self.name == DATASET_UUID_NAME)
                        )
                        updated_tensor += len(item)
                        self.items.clear()
                except Exception as e:
                    self.dataset.rollback({self.name: updated_tensor}, self.name)
                    e = e.__cause__ if isinstance(e, SampleAppendError) else e  # type: ignore
                    raise SampleAppendError(self.name) from e
            else:
                self.items.append(item)  # only not uuid, self.items=list
                self._item_added(item)  # calculate size
        except Exception as e:
            self.items.clear()
            raise SampleAppendError(self.name, item) from e

    def extend(self, items):
        """Adds items to the tensor."""
        if self.numpy_only:
            if self._extend_numpy(items):
                return

        for item in items:
            self.append(item)

    def _item_added(self, item):
        if self.dataset.all_chunk_engines:
            self.dataset.item_added(item, self.name)

    def _numpy_only_data(self):
        idx = self.idx
        if isinstance(idx, int):
            i = bisect.bisect_right(self.cum_sizes, idx)
            if i == 0:
                j = idx
            else:
                j = idx - self.cum_sizes[i - 1]
            return self.items[i][j]
        return self.items[idx]

    def _extend_numpy(self, items):
        """Extend tensor with a numpy array in a numpy-only tensor.
        Returns ``True`` if successful, ``False`` otherwise.
        """
        if isinstance(items, np.ndarray):
            self.items.append(items) # self.items: list

            # update cumulative sizes list
            if len(self.cum_sizes) == 0:
                self.cum_sizes.append(len(items))
            else:
                self.cum_sizes.append(self.cum_sizes[-1] + len(items))

            self._item_added(items)
            return True

        self.non_numpy_only()
        return False

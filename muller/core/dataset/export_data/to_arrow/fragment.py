# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import pyarrow.dataset


class MULLERArrowFragment(pyarrow.dataset.Fragment):
    def __init__(self, dataset, fragment_id=-1):
        self._ds = dataset
        self._fragment_id = fragment_id

    def scanner(self, schema=None, columns=None, myfilter=None, batch_size=None, batch_readahead=None,
                fragment_readahead=None, fragment_scan_options=None, use_threads=True,
                memory_pool=None):
        """Scanner."""
        from .arrow_dataset import MULLERArrowDatasetScanner
        return MULLERArrowDatasetScanner(self._ds, columns, myfilter)

    def count_rows(self, myfilter=None, batch_size=None, batch_readahead=None, fragment_readahead=None,
                   fragment_scan_options=None, use_threads=True, memory_pool=None):
        """Count rows."""
        return self.scanner(columns=list(self._ds.tensors.keys()), myfilter=myfilter).count_rows()

    def head(self, num_rows, columns=None, myfilter=None, batch_size=None, batch_readahead=None,
             fragment_readahead=None, fragment_scan_options=None, use_threads=True,
             memory_pool=None):
        """Return head."""
        return self.scanner(columns=columns, myfilter=myfilter).head(num_rows)

    def take(self, indices, columns=None, myfilter=None, batch_size=None, batch_readahead=None,
             fragment_readahead=None, fragment_scan_options=None, use_threads=True,
             memory_pool=None):
        """Take."""
        return self.scanner(columns=columns, myfilter=myfilter).take(indices)

    def to_batches(self, schema=None, columns=None, myfilter=None, batch_size=None, batch_readahead=None,
                   fragment_readahead=None, fragment_scan_options=None, use_threads=True,
                   memory_pool=None):
        """To Batches."""
        return self.scanner(columns=columns, myfilter=myfilter).to_batches()

    def to_table(self, schema=None, columns=None, myfilter=None, batch_size=None, batch_readahead=None,
                 fragment_readahead=None, fragment_scan_options=None, use_threads=True,
                 memory_pool=None):
        """To Arrow Table."""
        return self.scanner(columns=columns, myfilter=myfilter).to_table()

# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import pyarrow as pa
import pyarrow.dataset

from muller.core.dataset.export_data.to_arrow import MULLERArrowDatasetScanner, MULLERArrowFragment
from muller.util.exceptions import UnsupportedArrowConvertError

MULLER2ARROW_TYPE_MAP = {
    ("image", None): pa.binary(),
    ("video", None): pa.binary(),
    ("audio", None): pa.binary(),
    ("text", None): pa.string(),
    ("list", None): pa.list_(pa.string()),
    ("class_label", None): pa.int64(),
    ("generic", "int64"): pa.int64(),
    ("generic", "float64"): pa.float64(),
    ("generic", "float32"): pa.float32(),
}


class MULLERArrowDataset(pyarrow.dataset.Dataset):
    def __init__(self, ds, myfilter=None):
        self._ds = ds
        self._filter = myfilter

        self.dataset_fields = []
        dataset_columns = list(self._ds.tensors.keys())
        dataset_columns.sort()
        self.columns = dataset_columns
        for column in dataset_columns:
            tensor = self._ds[column]
            htype = tensor.htype
            if htype != "generic":
                dtype = None
            else:
                dtype = str(tensor.dtype)
            if (htype, dtype) in MULLER2ARROW_TYPE_MAP:
                self.dataset_fields.append(pa.field(column, MULLER2ARROW_TYPE_MAP[(htype, dtype)]))
            else:
                raise UnsupportedArrowConvertError(htype, dtype)

    @property
    def schema(self) -> pa.Schema:
        """
        The pyarrow Schema for this dataset
        """
        return pa.schema(self.dataset_fields)

    @property
    def partition_expression(self):
        """
        Not implemented (just override pyarrow dataset to prevent segfault)
        """
        raise NotImplementedError("partitioning not yet supported")

    def get_muller_dataset(self):
        """Get MULLER Dataset."""
        return self._ds

    def merge_filter(self, my_filter):
        """Merge filter."""
        if self._filter is None and my_filter is None:
            merged_filter = None
        elif self._filter is None or my_filter is None:
            merged_filter = self._filter if self._filter is not None else my_filter
        else:
            merged_filter = my_filter & self._filter
        return merged_filter

    def replace_schema(self, schema):
        """
        Not implemented (just override pyarrow dataset to prevent segfault)
        """
        raise NotImplementedError("not changing schemas yet")

    def count_rows(self, myfilter=None, batch_size=None, batch_readahead=None,
                   fragment_readahead=None, fragment_scan_options=None, use_threads=True,
                   memory_pool=None):
        """Count rows."""
        merged_filter = self.merge_filter(myfilter)
        if merged_filter is None:
            return len(self._ds)
        return self.scanner(myfilter=merged_filter).count_rows()

    def scanner(self, columns=None, myfilter=None, batch_size=None, batch_readahead=None,
                fragment_readahead=None, fragment_scan_options=None, use_threads=True,
                memory_pool=None):
        """The scanner."""
        merged_filter = self.merge_filter(myfilter)
        return MULLERArrowDatasetScanner(self, columns, merged_filter)

    def filter(self, myfilter):
        """Filter dataset."""
        merged_filter = self.merge_filter(myfilter)
        return MULLERArrowDataset(self._ds, merged_filter)

    def head(self, num_rows, columns=None, myfilter=None, batch_size=None, batch_readahead=None,
             fragment_readahead=None, fragment_scan_options=None, use_threads=True,
             memory_pool=None):
        """Return head."""
        return self.scanner(columns, myfilter, batch_size).head(num_rows)

    def join(self, right_dataset, keys, right_keys=None, join_type=None, left_suffix=None, right_suffix=None,
             coalesce_keys=True, use_threads=True):
        """
        Not implemented (just override pyarrow dataset to prevent segfault)
        """
        raise NotImplementedError("do not support join operation")

    def take(self, indices, columns=None, myfilter=None, batch_size=None, batch_readahead=None,
             fragment_readahead=None, fragment_scan_options=None, use_threads=True,
             memory_pool=None):
        """Take."""
        return self.scanner(columns, myfilter).take(indices)

    def to_batches(self, columns=None, myfilter=None, batch_size=None, batch_readahead=None,
                   fragment_readahead=None, fragment_scan_options=None, use_threads=True,
                   memory_pool=None):
        """To Batches."""
        return self.scanner(columns, myfilter).to_batches()

    def to_table(self, columns=None, myfilter=None, batch_size=None, batch_readahead=None,
                 fragment_readahead=None, fragment_scan_options=None, use_threads=True,
                 memory_pool=None):
        """To Arrow Table."""
        merged_filter = self.merge_filter(myfilter)
        return self.scanner(columns, merged_filter).to_table()

    def get_fragments(self, myfilter=None):
        """Get fragments."""
        if myfilter is not None:
            raise ValueError("get_fragments() does not support filter yet")
        return [MULLERArrowFragment(self._ds)]

    def sort_by(self, sorting, **kwargs):
        """Sort"""
        return self.scanner().to_table().sort_by(sorting)

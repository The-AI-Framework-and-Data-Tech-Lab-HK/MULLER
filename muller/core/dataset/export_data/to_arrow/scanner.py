# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import pyarrow as pa
import pyarrow.dataset


class MULLERArrowDatasetScanner(pyarrow.dataset.Scanner):
    def __init__(self, dataset, columns=None, myfilter=None, batch_size=131_072):
        self._dataset = dataset
        self._filter = myfilter
        self.batch_size = batch_size
        self.dataset_fields = self._dataset.dataset_fields
        self.dataset_columns = self._dataset.columns
        if columns is None:
            self._columns = self.dataset_columns
        else:
            self._columns = columns
        self.projected_fields = []
        columns_set = set(self._columns)
        for column, field in zip(self.dataset_columns, self.dataset_fields):
            if column in columns_set:
                self.projected_fields.append(field)

    @property
    def dataset_schema(self):
        """Obtain dataset schema."""
        return pa.schema(self.dataset_fields)

    @property
    def projected_schema(self):
        """Obtain projected schema."""
        return pa.schema(self.projected_fields)

    def to_table(self):
        """Export to Arrow Table."""
        return self.to_reader().read_all()

    def record_batch_generator(self, batch_size=None):
        """Record a batch of records."""
        if batch_size is None:
            batch_size = self.batch_size
        schema = self.projected_schema
        muller_ds = self._dataset.get_muller_dataset()
        num_rows = len(muller_ds)
        for start in range(0, num_rows, batch_size):
            end = min(start + batch_size, num_rows)
            arrays = []
            for column in self._columns:
                value = muller_ds[column][start:end].arrow()
                arrays.append(value)
            record_batch = pa.RecordBatch.from_arrays(arrays, schema)
            if self._filter is not None:
                table = pa.Table.from_batches([record_batch])
                filtered_table = table.filter(self._filter)
                record_batch = filtered_table.to_batches()[0]
            yield record_batch

    def to_reader(self):
        """Transform dataset to Arrow Dataset."""
        record_batch_reader = pa.RecordBatchReader.from_batches(self.projected_schema, self.record_batch_generator())
        return record_batch_reader

    def scan_batches(self):
        """Scan for records in batches."""
        lst = []
        reader = self.to_reader()
        while True:
            try:
                batch = reader.read_next_batch()
            except StopIteration:
                break
            lst.append(batch)
        return lst

    def count_rows(self):
        """Count rows."""
        count = 0
        reader = self.to_reader()
        while True:
            try:
                batch = reader.read_next_batch()
            except StopIteration:
                break
            count += batch.num_rows
        return count

    def take(self, indices):
        """Take."""
        return self.to_table()[indices]

    def head(self, int_num_rows):
        """Return head."""
        record_batch_generator = self.record_batch_generator(int_num_rows)
        batches = []
        num_rows = 0
        while num_rows < int_num_rows:
            try:
                batch = next(record_batch_generator)
            except StopIteration:
                break
            num_rows += batch.num_rows
            batches.append(batch)
        table = pa.Table.from_batches(batches)
        return table[0:int_num_rows]

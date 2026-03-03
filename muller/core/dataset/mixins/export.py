# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

"""Export operations mixin for Dataset class."""

import json
import warnings
from typing import List, Optional, Union

import muller.core.dataset
from muller.constants import TO_DATAFRAME_SAFE_LIMIT
from muller.util.exceptions import (
    InvalidJsonFileName,
    InvalidNumWorkers,
    InvalidTensorList,
    ToDataFrameLimit,
)


class ExportMixin:
    """Mixin providing export operations for Dataset."""

    def numpy(self, aslist=False, fetch_chunks=False, asrow=False) -> Union[dict, list]:
        """Computes the contents of the dataset slices in numpy format."""
        return muller.core.dataset.to_numpy(self, aslist, fetch_chunks, asrow)

    def to_dataframe(self, tensor_list: Optional[List[str]] = None,
                     index_list: Optional[List] = None,
                     force: bool = False):
        """Returns a pandas dataframe of the dataset.

        Example:
            >>> ds.to_dataframe()
            >>> ds.to_dataframe(index_list=[-1, -2])
            >>> ds.to_dataframe(tensor_list=["categories"], index_list=[1, 2, 4])

        Args:
            tensor_list (List of str, Optional): The tensor columns selected to be exported as pandas dataframe.
                        If not provided, we will export all the tensor columns.
            index_list (List of int, Optional): The indices of the rows selected to be exported as pandas dataframe.
                        If not provided, we will export all the row.
            force (bool, Optional): Dataset with more than TO_DATAFRAME_SAFE_LIMIT samples might take a long time to
                        export. If force = True, the dataset will be exported regardless.
                        An error will be raised otherwise.

        Raises:
            InvalidTensorList: If ``tensor_list`` contains tensors that are not in the current columns.
            ToDataFrameLimit: If the length of ``index_list`` exceeds the TO_DATAFRAME_SAFE_LIMIT.
        """
        # Verify that the target column is correctly specified.
        if tensor_list and (len(tensor_list) > len(self.tensors) or
                            not all(isinstance(x, str) and x in self.tensors for x in tensor_list)):
            raise InvalidTensorList(tensor_list)
        max_num = -1
        if index_list and len(index_list) > TO_DATAFRAME_SAFE_LIMIT and not force:
            max_num = len(index_list)
        elif self.max_len > TO_DATAFRAME_SAFE_LIMIT and not force:
            max_num = self.max_len
        if max_num != -1:
            raise ToDataFrameLimit(max_num, TO_DATAFRAME_SAFE_LIMIT)

        return muller.core.dataset.to_dataframe(self, tensor_list, index_list)

    def to_arrow(self):
        """Returns an arrow object of the dataset."""
        return muller.core.dataset.MULLERArrowDataset(self)

    def write_to_parquet(self, path, columns=None):
        """Returns an arrow of the dataset."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        arrow_dataset = self.to_arrow()
        arrow_table = arrow_dataset.to_table(columns)
        writer = pa.BufferOutputStream()
        pq.write_table(arrow_table, writer)
        self.storage[path] = bytes(writer.getvalue())

    def to_json(
            self,
            path,
            tensors: Optional[List[str]] = None,
            num_workers: Optional[int] = 1,
    ):
        """Write MULLER data by row to a jsonl or json file.

        Example:
            >>> ds.to_json("test.jsonl")

        Args:
            path (str): The jsonl or json file name to save MULLER data.
            tensors (List of str, Optional): The tensor columns selected to be exported to the jsonl or json file.
            num_workers (int, Optional): The number of workers that can be used to dump to the path.
        """
        if not (path.endswith("json") or path.endswith("jsonl")):
            raise InvalidJsonFileName(path)
        if num_workers <= 0:
            raise InvalidNumWorkers(num_workers)
        muller.core.dataset.to_json(self, path, tensors, num_workers)

    def to_mindrecord(
            self,
            file_name,
            shard_num: int = 1,
            batch_size: int = 100000,
            overwrite: bool = False,
            scheduler: str = "threaded",
    ):
        """Create MindRecord and save as file_name.

        Example:
            >>> ds.to_mindrecord("ds.mindrecord", shard_num=1, batch_size=1000, overwrite=True)

        Args:
            file_name (str): The filename to save MindRecord.
            shard_num (int, Optional): The number of MindRecord files to generate.
            batch_size (int, Optional): The batch size to get numpy data from MULLER.
            overwrite (bool, Optional): Overwrite if same file name exists.
            scheduler (str): The scheduler to be used for optimization. Supported values include: 'serial', 'threaded',
                'processed' and 'distributed'. Defaults to 'threaded'.
        """
        import muller.core.dataset.export_data.to_mindrecord
        muller.core.dataset.export_data.to_mindrecord.create_mindrecord(self, file_name, shard_num, batch_size,
                                                                       overwrite, scheduler)

    def statistics(self):
        """Get statistics info of dataset. Load from dataset_meta.json first, if empty, calculate and then save it.

        Example:
            >>> ds = muller.load("path_to_dataset")
            >>> ds.statistics()
        """
        from muller.core.dataset.statistics.statistics import get_statistics, load_statistics, save_statistics
        if self.has_head_changes:
            warnings.warn(
                "There are uncommitted changes, showing statistics from last committed version, try again after commit."
            )

        stats = load_statistics(self)
        if not stats:
            stats = get_statistics(self)
            save_statistics(self, stats)
        print(json.dumps(stats))

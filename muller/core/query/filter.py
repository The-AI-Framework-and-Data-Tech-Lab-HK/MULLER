# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/core/query/filter.py
#
# Modifications Copyright (c) 2026 Bingyu Liu

import inspect
import threading
from collections import defaultdict
from queue import Queue
from time import time
from typing import Callable, List, Optional, Dict, Sequence
from uuid import uuid4

import numpy as np

import muller
from muller.constants import (
    QUERY_PROGRESS_UPDATE_FREQUENCY,
    TRANSFORM_PROGRESSBAR_UPDATE_INTERVAL,
)
from muller.core.query.query import DatasetQuery
from muller.util.compute import get_compute_provider
from muller.util.exceptions import FilterError
from muller.util.hash import hash_inputs

_LAST_UPDATED_TIMES: Dict = defaultdict(time)


def _counter(temp_id):
    """A method which returns True only every `QUERY_PROGRESS_UPDATE_FREQUENCY` seconds for each id.
    Used for sending query progress update events and writing to vds.
    """
    last_updated_time = _LAST_UPDATED_TIMES[temp_id]
    curr_time = time()
    if curr_time - last_updated_time > QUERY_PROGRESS_UPDATE_FREQUENCY:
        _LAST_UPDATED_TIMES[temp_id] = curr_time
        return True
    return False


def _del_counter(temp_id):
    _LAST_UPDATED_TIMES.pop(temp_id, None)


def _filter_function_to_query_text(filter_function):
    try:
        query_text = inspect.getsource(filter_function)
    except (OSError, TypeError):
        query_text = (
                "UDF: "
                + getattr(
            filter_function, "__name__", filter_function.__class__.__name__
        )
                + "@"
                + str(uuid4().hex)
        )
    return query_text


def filter_dataset(
        dataset: muller.Dataset,
        filter_function: Callable[[muller.Dataset], bool],
        num_workers: int = 0,
        scheduler: str = "thread",
        progressbar: bool = True,
        save_result: bool = False,
        result_path: Optional[str] = None,
        result_ds_args: Optional[dict] = None,
        offset: Optional[int] = 0,
        limit: Optional[int] = None,
) -> muller.Dataset:
    """Conduct filtering on the dataset."""
    index_map: List[int]
    tm = time()
    query_text = _filter_function_to_query_text(filter_function)

    try:
        if num_workers > 0:  # Sherry:!
            index_map = filter_with_compute(
                dataset=dataset,
                filter_function=filter_function,
                num_workers=num_workers,
                scheduler=scheduler,
                progressbar=progressbar,
                query_text=query_text,
                offset=offset,
                limit=limit,
            )
        else:
            index_map = filter_inplace(
                dataset=dataset,
                filter_function=filter_function,
                progressbar=progressbar,
                query_text=query_text,
                limit=limit,
            )
    except Exception as e:
        raise e

    ds = dataset[index_map]
    ds.is_filtered_view = True
    ds.query_string = query_text
    ds.source_ds_idx = dataset.index.to_json()
    ds.created_at = tm
    ds.filtered_index = index_map

    return ds


def _get_vds_thread(vds: muller.Dataset, queue: Queue, num_samples: int):
    temp_id = str(uuid4().hex)

    def loop():
        processed = 0
        while True:
            index, include = queue.get()
            vds.info["samples_processed"] += 1
            if include:
                vds.VDS_INDEX.append(index)
            processed += 1
            if processed == num_samples:
                vds.flush()
                _del_counter(temp_id)
                break
            if _counter(temp_id):
                vds.flush()

    return threading.Thread(target=loop)


def filter_with_compute(
        dataset: muller.Dataset,
        filter_function: Callable,
        **kwargs,
) -> List[int]:
    """Filter with compute."""

    # 1. Preprocessing
    initial_is_iteration = dataset.is_iteration
    dataset.is_iteration = True
    # Removed SampleStreaming which creates IO Blocks to get dataset index,
    # instead, used dataset.index below to get index.
    compute = get_compute_provider(scheduler=kwargs.get("scheduler", "thread"),
                                   num_workers=kwargs.get("num_workers", 0))
    vds, vds_queue, vds_thread = _vds_setup(kwargs.get("vds", None), compute, len(dataset))
    progress = {"value": 0}
    result: Sequence[List[int]]

    def _event_callback():
        progress["value"] += 1

    # 2. Construct the filtering functions
    slice_fn = _make_slice_filter_fn(filter_function, dataset, vds, vds_queue, kwargs.get("offset", 0),
                                     _event_callback)
    pg_slice_fn = _make_pg_slice_filter_fn(filter_function, dataset, vds, vds_queue, kwargs.get("offset", 0),
                                           _event_callback)

    # 3. Conduct the filtering by dividing into batches.
    idx = _get_index_batches(dataset,
                             kwargs.get("offset", 0),
                             kwargs.get("num_workers", 0))

    try:
        if kwargs.get("progressbar", None):
            result = compute.map_with_progress_bar(pg_slice_fn,
                                                   idx,
                                                   total_length=len(dataset))  # type: ignore
        else:
            result = compute.map(slice_fn, idx)  # type: ignore
    except Exception as e:
        raise FilterError from e

    finally:
        compute.close()
        if vds:
            if hasattr(vds_queue, "close"):
                vds_queue.close()

        _del_counter(hash_inputs(dataset.path, dataset.pending_commit_id, kwargs.get("query_text", None)))

        dataset.is_iteration = initial_is_iteration
    if vds:
        vds.autoflush = True
        vds_thread.join()
    return _get_index_map(result)[:kwargs.get("limit", None)]


def _get_index_map(result):
    index_map = [k
                 for x in result
                 for k in x]  # unfold the result map
    return index_map


def _vds_setup(vds, compute, num_samples):
    vds_queue = None
    vds_thread = None
    if vds:
        vds.autoflush = False
        vds.info["total_samples"] = num_samples
        vds.info["samples_processed"] = 0
        if compute:
            vds_queue = compute.create_queue()
        else:
            vds_queue: Queue = Queue()
        vds_thread = _get_vds_thread(vds, vds_queue, num_samples)
        vds_thread.start()
    return vds, vds_queue, vds_thread


def _make_slice_filter_fn(filter_function, dataset, vds, vds_queue, offset, _event_callback):
    def filter_slice(indices: Sequence[int]):
        result = list()
        for i in indices:
            if filter_function(dataset[i-offset]):
                result.append(i-offset)
                if vds:
                    vds_queue.put((i-offset, True))
                    _event_callback()
            elif vds:
                vds_queue.put((i-offset, False))
                _event_callback()
        return result
    return filter_slice


def _make_pg_slice_filter_fn(filter_function, dataset, vds, vds_queue, offset, _event_callback):
    def pg_filter_slice(pg_callback, indices: Sequence[int]):
        result = list()
        progress = 0
        t1 = time()
        for i in indices:
            if filter_function(dataset[i-offset]):
                result.append(i-offset)
                if vds:
                    vds_queue.put((i-offset, True))
                    _event_callback()
            elif vds:
                vds_queue.put((i-offset, True))
                _event_callback()
            progress += 1

            if time() - t1 > TRANSFORM_PROGRESSBAR_UPDATE_INTERVAL:
                pg_callback(progress)
                progress = 0
                t1 = time()
        if progress > 0:
            pg_callback(progress)
        return result
    return pg_filter_slice


def _get_index_batches(dataset, offset, num_workers):
    version_state = dataset.version_state
    tensor_lengths = [
        len(version_state["full_tensors"][version_state["tensor_names"][tensor]])
        for tensor in dataset.tensors.keys()
    ]
    length = min(tensor_lengths, default=0)
    # get indices from dataset, split by num_workers
    indices = [i for i in dataset.index.values[0].indices(length) if i >= offset]
    temp = np.array_split(indices, min(num_workers, len(indices)))
    idx = [temp[i].tolist() for i in range(len(temp))]
    return idx


def filter_inplace(
        dataset: muller.Dataset,
        filter_function: Callable,
        progressbar: bool,
        query_text: Optional[str] = None,
        vds: Optional[muller.Dataset] = None,
        limit: Optional[int] = None
) -> List[int]:
    """Filter inplace."""
    index_map: List[int] = list()
    it = enumerate(dataset)
    if not limit:
        limit = len(dataset)

    vds, vds_queue, vds_thread = _vds_setup(vds, compute=None, num_samples=len(dataset))

    if progressbar:
        from tqdm import tqdm
        it = tqdm(it, total=len(dataset))
    query_id = hash_inputs(dataset.path, dataset.pending_commit_id, query_text)

    try:
        for i, sample_in in it:
            if len(index_map) >= limit:
                break
            if filter_function(sample_in):
                index_map.append(i)
                if vds:
                    vds_queue.put((i, True))
            elif vds:
                vds_queue.put((i, False))

    except Exception as e:
        raise e
    finally:
        _del_counter(query_id)

    if vds:
        vds.autoflush = True
        vds_thread.join()
    return index_map


def query_dataset(
        dataset: muller.Dataset,
        query: str,
        num_workers: int = 0,
        scheduler: str = "threaded",
        progressbar: bool = True,
        save_result: bool = False,
        result_path: Optional[str] = None,
        result_ds_args: Optional[Dict] = None,
        offset: Optional[int] = 0,
        limit: Optional[int] = None,
) -> muller.Dataset:
    """Query the dataset."""
    index_map: List[int]

    vds = None
    index_map = query_inplace(dataset, query, progressbar, num_workers, scheduler, vds, limit)
    ret = dataset[index_map]
    ret.query_string = query
    ret.filtered_index = index_map
    if vds:
        ret.vds = vds
    return ret


def query_inplace(
        dataset: muller.Dataset,
        query: str,
        progressbar: bool,
        num_workers: int,
        scheduler: str,
        vds: Optional[muller.Dataset] = None,
        limit: Optional[int] = None,
) -> List[int]:
    """Query inplace."""
    if not limit:
        limit = len(dataset)
    compute = (
        get_compute_provider(scheduler=scheduler, num_workers=num_workers)
        if num_workers > 0
        else None
    )
    query_id = hash_inputs(dataset.path, dataset.pending_commit_id, query)
    vds, vds_queue, _ = _vds_setup(vds, compute, num_samples=len(dataset))

    try:
        index_map = _query_inplace_process(scheduler, num_workers, dataset, query, progressbar, limit, vds, vds_queue)

    except Exception as e:
        raise e
    finally:
        _del_counter(query_id)
    if vds:
        pass
    return index_map[:limit]


def _query_inplace_process(scheduler,
                           num_workers,
                           dataset,
                           query,
                           progressbar,
                           limit,
                           vds,
                           vds_queue):
    num_processed = {"value": 0}

    def update_vds(idx, include):
        """Update vds. """
        if vds:
            vds_queue.put((idx, include))
            num_processed["value"] += 1

    def subquery(query_slice):
        """Returns the subquery. """
        dataset = query_slice.slice_dataset()
        query = query_slice.query

        if progressbar:
            from tqdm import tqdm
            bar = tqdm(total=len(dataset))

            def update(idx, include):
                bar.update(1)
                update_vds(idx, include)

            try:
                ds_query = DatasetQuery(dataset, query, update)
                ret = ds_query.execute(limit)
            finally:
                bar.close()
        else:
            ret = DatasetQuery(dataset, query, update_vds).execute(limit)
        return ret

    def pg_subquery(pg_callback, query_slice):
        """Subquery execution. """
        def update(idx, include):
            update_vds(idx, include)
            pg_callback(1)

        dataset = query_slice.slice_dataset()
        ds_query = DatasetQuery(dataset, query, progress_callback=update)
        return ds_query.execute(limit)

    if num_workers == 0:
        return subquery(QuerySlice(0, len(dataset), dataset, query))

    compute = get_compute_provider(scheduler=scheduler, num_workers=num_workers)
    subdatasets = _get_subdatasets(dataset, num_workers, query)

    if progressbar:
        result = compute.map_with_progress_bar(pg_subquery, subdatasets,
                                               total_length=len(dataset))  # type: ignore
    else:
        result = compute.map(subquery, subdatasets)  # type: ignore
    return _get_new_index_map(result, subdatasets)


class QuerySlice:
    """A class of the query slice."""
    def __init__(self, offset, size, dataset, query):
        self.dataset = dataset
        self.offset = offset
        self.size = size
        self.query = query

    def slice_dataset(self):
        """Slice the dataset. """
        return self.dataset[self.offset: (self.offset + self.size)]


def _get_subdatasets(dataset, num_workers, query):
    btch = len(dataset) // num_workers
    subdatasets = [
        QuerySlice(idx * btch, btch, dataset, query)
        for idx in range(0, num_workers)
    ]
    return subdatasets


def _get_new_index_map(result, subdatasets):
    index_map = [
        k + dataset_slice.offset
        for x, dataset_slice in zip(result, subdatasets)
        for k in x
    ]
    return index_map

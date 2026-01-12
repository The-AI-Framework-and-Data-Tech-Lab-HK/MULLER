# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/core/chunk_engine.py
#
# Modifications Copyright (c) 2026 Xueling Lin

from concurrent.futures import ThreadPoolExecutor
from typing import (
    Optional,
    List
)

import numpy as np

from muller.constants import (
    MAX_WORKERS_FOR_CHUNK_ENGINE
)
from muller.core.meta.encode.chunk_id import ChunkIdEncoder
from muller.util.keys import get_chunk_key
from .chunk_engine_to_numpy_interface import get_samples_full


def merge_regions(chunk_engine, rows_groups: List[List[int]], ids_groups: List[List[int]]):
    """Merge regions. """
    rows_to_delete: List[int] = []
    for rows, ids in zip(rows_groups, ids_groups):
        to_row = rows[0]
        to_cid = ids[0]
        to_chunk = chunk_engine.get_chunk_from_chunk_id(to_cid)

        from_cids = ids[1:]
        from_rows = rows[1:]
        chunk_names = [ChunkIdEncoder.name_from_id(chunk_id) for chunk_id in from_cids]
        if chunk_engine.is_fixed_shape:
            samples_to_move = get_samples_full(chunk_engine, chunk_names, len(chunk_names), strategy="threaded")
        else:
            samples_to_move = _read_all_chunks_threaded(chunk_engine, chunk_names, len(chunk_names))
        _clear_chunks(chunk_engine, chunk_names)
        samples = chunk_engine.sanitize_samples(samples_to_move)
        to_chunk.is_dirty = True
        chunk_engine.active_updated_chunk = to_chunk
        _ = chunk_engine.samples_to_chunks(
            samples,
            start_chunk=to_chunk,
            register=True,
            update_commit_diff=False,  # merging chunks should not update diff
            update_tensor_meta=False,
            start_chunk_row=to_row,
        )
        rows_to_delete.extend(from_rows)
        chunk_engine.cache[to_chunk.key] = to_chunk  # type: ignore
    for row in sorted(set(rows_to_delete), reverse=True):
        chunk_engine.chunk_id_encoder.delete_chunk_id(row)


def _read_all_chunks_threaded(chunk_engine, all_chunk_names: Optional[List[str]],
                             max_workers: int = MAX_WORKERS_FOR_CHUNK_ENGINE):
    """Read all chunks in multiple threads. """
    results = []

    workers = min(max_workers, len(all_chunk_names))
    with ThreadPoolExecutor(workers) as executor:
        futs = [executor.submit(_get_chunk_bytes_full, chunk_engine, name)
                for name in all_chunk_names]
        for fut in futs:
            results.extend(fut.result())
    return results


def _get_chunk_bytes_full(chunk_engine, chunk_name):
    chunk_key = get_chunk_key(chunk_engine.key, chunk_name)
    chunk = chunk_engine.get_chunk(chunk_key)
    chunk.check_empty_before_read()
    enc = chunk.byte_positions_encoder.encoded
    if enc.size == 0:
        return []
    buffer = chunk.memoryview_data
    lengths = enc[:, 0].astype(np.int64, copy=False)  # length bytes
    starts = enc[:, 1].astype(np.int64, copy=False)  # start byte
    ends = starts + lengths

    if not chunk.is_text_like:
        raise ValueError("only support text-like yet.")
    return [np.array(bytes(buffer[s:e]).decode("utf-8")).reshape(1)
            for s, e in zip(starts, ends)]


def _clear_chunks(chunk_engine, chunk_names):
    """Clear all chunks. """
    for cuk_name in chunk_names:
        chunk_key = get_chunk_key(chunk_engine.key, cuk_name)
        chunk = chunk_engine.get_chunk(chunk_key)
        chunk.prepare_for_write()

        if not chunk.byte_positions_encoder.is_empty():
            chunk.data_bytes = chunk.data_bytes[:0]
            chunk.byte_positions_encoder.encoded = chunk.byte_positions_encoder.encoded[:0]
            chunk.byte_positions_encoder.is_dirty = True
        if not chunk.shapes_encoder.is_empty():
            chunk.shapes_encoder.encoded = chunk.shapes_encoder.encoded[:0]
            chunk.shapes_encoder.is_dirty = True
        del chunk_engine.cache[chunk_key]

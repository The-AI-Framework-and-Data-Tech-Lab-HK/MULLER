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

from typing import (
    Optional,
    List
)

import numpy as np

from muller.constants import (
    RANDOM_MINIMAL_CHUNK_SIZE
)
from muller.core.chunk.base_chunk import BaseChunk
from muller.core.chunk.chunk_compressed_chunk import ChunkCompressedChunk
from muller.core.meta.encode.base_encoder import LAST_SEEN_INDEX_COLUMN
from muller.core.meta.encode.chunk_id import ChunkIdEncoder
from muller.core.sample import Sample
from muller.util.keys import get_chunk_key


def check_rechunk(chunk_engine, chunk: BaseChunk, chunk_row: int):
    """function to check if there is a need to re-chunk the current one"""

    if _is_tensor_hidden(chunk_engine):
        return
    if chunk.num_data_bytes < RANDOM_MINIMAL_CHUNK_SIZE < chunk_engine.max_chunk_size:
        _try_merge_with_neighbor_and_split(chunk_engine, chunk=chunk, row=chunk_row)

    elif chunk.num_data_bytes > chunk_engine.max_chunk_size:
        _rechunk(chunk_engine, chunk, chunk_row)


def get_sample_object(
        chunk_engine, sample_data, sample_shape, compression, dtype, decompress
):
    """Obtain sample objects. """
    if chunk_engine.is_text_like:
        sample = sample_data
        if chunk_engine.tensor_meta.htype == "json" and isinstance(sample, np.ndarray):
            sample = sample.squeeze()
        return sample

    if decompress:
        sample = Sample(array=sample_data, shape=sample_shape)
    else:
        # sample data should not be an array here
        assert not isinstance(sample_data, np.ndarray)
        sample = Sample(
            buffer=sample_data,
            shape=sample_shape,
            compression=compression,
            dtype=dtype,
        )
    return sample


def _is_tensor_hidden(chunk_engine) -> bool:
    """function to check is the tensors that chunk_engine belongs to is hidden"""
    tensor_name = chunk_engine.tensor_meta.name or chunk_engine.key
    if tensor_name.startswith("_"):
        return (
                tensor_name.endswith("_shape")
                or tensor_name.endswith("_id")
                or tensor_name.endswith("_info")
        )
    return False


def _try_merge_with_neighbor_and_split(chunk_engine, chunk: BaseChunk, row: int):
    if _try_merge_with_previous_chunk(chunk_engine, chunk, row) is False:
        _try_merge_with_next_chunk(chunk_engine, chunk, row)


def _try_merge_with_previous_chunk(chunk_engine, chunk: BaseChunk, row: int) -> bool:
    prev_chunk_id = chunk_engine.chunk_id_encoder.get_prev_chunk_id(row)
    if prev_chunk_id is None:
        return False

    prev_chunk_row = row - 1
    if _is_tiled(chunk_engine, prev_chunk_row):
        return False

    prev_chunk_name = ChunkIdEncoder.name_from_id(int(prev_chunk_id))  # type: ignore
    prev_chunk_key = get_chunk_key(chunk_engine.key, prev_chunk_name)
    prev_chunk_size = chunk_engine.cache.get_object_size(prev_chunk_key)
    prev_chunk = chunk_engine.get_chunk_from_chunk_id(int(prev_chunk_id))
    if prev_chunk_size + chunk.num_data_bytes < prev_chunk.min_chunk_size:
        if not chunk_engine.chunk_in_target_commit(prev_chunk_name, chunk_engine.commit_id):
            prev_chunk = chunk_engine.copy_chunk(prev_chunk, row=prev_chunk_row)
        chunk_id = chunk.id
        chunk_name = ChunkIdEncoder.name_from_id(chunk_id)
        if not chunk_engine.chunk_in_target_commit(chunk_name, chunk_engine.commit_id):
            chunk = chunk_engine.copy_chunk(chunk, row=row)
        # merge with previous chunk
        return _merge_chunks(
            chunk_engine,
            from_chunk=chunk,
            from_chunk_row=row,
            to_chunk=prev_chunk,
            to_chunk_row=prev_chunk_row,
        )
    return False


def _try_merge_with_next_chunk(chunk_engine, chunk: BaseChunk, row: int) -> bool:
    next_chunk_id = chunk_engine.chunk_id_encoder.get_next_chunk_id(row)
    if next_chunk_id is None:
        return False
    next_chunk_row = row + 1
    if _is_tiled(chunk_engine, next_chunk_row):
        return False
    return True


def _is_tiled(chunk_engine, row: int) -> bool:
    """checks whether the chunk is tiled or not

    Args:
        row (int): Represents the row of the chunk.

    Returns:
        bool: return true if the current chunk and previous/next row chunk have the same chunk index
              false otherwise.
    """

    arr = chunk_engine.chunk_id_encoder.array
    if row >= 1 and len(arr) > 1:
        if arr[row][LAST_SEEN_INDEX_COLUMN] == arr[row - 1][LAST_SEEN_INDEX_COLUMN]:
            return True
    if len(arr) > row + 1:
        if arr[row][LAST_SEEN_INDEX_COLUMN] == arr[row + 1][LAST_SEEN_INDEX_COLUMN]:
            return True
    return False


def _merge_chunks(
    chunk_engine,
    from_chunk: BaseChunk,
    from_chunk_row: int,
    to_chunk: BaseChunk,
    to_chunk_row: int,
):
    samples_to_move = _get_chunk_samples(chunk_engine, chunk=from_chunk)
    num_samples = len(samples_to_move)
    if num_samples == 0:
        return True

    from_chunk.pop_multiple(num_samples=num_samples)
    samples = chunk_engine.sanitize_samples(samples_to_move)
    to_chunk.is_dirty = True
    chunk_engine.active_updated_chunk = to_chunk
    chunk_engine.samples_to_chunks(
        samples,
        start_chunk=to_chunk,
        register=True,
        update_commit_diff=False,  # merging chunks should not update diff
        update_tensor_meta=False,
        start_chunk_row=to_chunk_row,
    )
    chunk_engine.chunk_id_encoder.delete_chunk_id(row=from_chunk_row)
    try:
        del chunk_engine.cache[from_chunk.key]  # type: ignore
    except KeyError:
        pass
    chunk_engine.cache[to_chunk.key] = to_chunk  # type: ignore
    return True


def _get_chunk_samples(chunk_engine, chunk) -> List[Optional[Sample]]:
    decompress = isinstance(chunk, ChunkCompressedChunk) or chunk_engine.is_text_like
    all_samples_in_chunk: List[Optional[Sample]] = []

    for idx in range(chunk.num_samples):
        sample_data = chunk.read_sample(idx, decompress=decompress)
        try:
            sample_shape = chunk.shapes_encoder[idx]
        except IndexError:
            all_samples_in_chunk.append(None)
            continue
        new_sample = chunk_engine.get_sample_object(
            sample_data, sample_shape, chunk.compression, chunk.dtype, decompress
        )
        all_samples_in_chunk.append(new_sample)

    return all_samples_in_chunk


def _rechunk(chunk_engine, chunk: BaseChunk, chunk_row: int):
    samples_to_move = _get_samples_to_move(chunk_engine, chunk=chunk)
    num_samples = len(samples_to_move)
    if num_samples == 0:
        return
    new_chunk = chunk_engine.create_new_chunk(register=True, row=chunk_row)
    new_chunk_row = chunk_row + 1

    chunk_engine.chunk_id_encoder.decrease_samples(row=chunk_row, num_samples=num_samples)
    chunk_engine.chunk_id_encoder.decrease_samples(
        row=new_chunk_row, num_samples=num_samples
    )
    chunk.pop_multiple(num_samples=len(samples_to_move))
    samples = chunk_engine.sanitize_samples(samples_to_move)
    chunk_engine.samples_to_chunks(
        samples,
        start_chunk=new_chunk,
        register=True,
        update_commit_diff=True,
        update_tensor_meta=False,
        start_chunk_row=new_chunk_row,
    )


def _get_samples_to_move(chunk_engine, chunk) -> List[Sample]:
    decompress = isinstance(chunk, ChunkCompressedChunk) or chunk_engine.is_text_like
    samples_to_move: List[Sample] = []
    sum_bytes = 0

    for idx in range(chunk.num_samples - 1, 1, -1):
        sample_data = chunk.read_sample(idx, decompress=decompress)
        sum_bytes += len(sample_data)
        if sum_bytes > chunk_engine.max_chunk_size:
            break
        sample_shape = chunk.shapes_encoder[idx]
        new_sample = chunk_engine.get_sample_object(
            sample_data, sample_shape, chunk.compression, chunk.dtype, decompress
        )
        samples_to_move.append(new_sample)
    samples_to_move.reverse()
    return samples_to_move

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
    Callable,
    Optional,
    Tuple
)

import numpy as np

from muller.core.index.index import Index
from muller.core.meta.encode.chunk_id import ChunkIdEncoder
from muller.util.keys import get_chunk_key


def shape(
        chunk_engine,
        index: Index,
        sample_shape_provider: Optional[Callable] = None,
) -> Tuple[Optional[int], ...]:
    """Returns the shape. """
    tensor_ndim = chunk_engine.ndim()

    if len(index) > tensor_ndim:
        raise IndexError(
            f"Too many indices for tensor. Tensor is rank {tensor_ndim} but {len(index)} indices were provided."
        )

    index_0, sample_index = index.values[0], index.values[1:]

    num_samples = index_0.length(chunk_engine.sequence_length or chunk_engine.num_samples)
    if chunk_engine.tensor_meta.min_shape == chunk_engine.tensor_meta.max_shape:
        if index_0.is_trivial() or num_samples == 0:
            tmp_shape = chunk_engine.shape_interval(index).astuple()
            return tmp_shape
        tmp_shape = chunk_engine.shape_interval(index).astuple()[1:]
    else:
        tmp_shape = None

    sample_ndim = tensor_ndim - 1
    sample_shapes = np.zeros((num_samples, sample_ndim), dtype=np.int32)

    if tmp_shape is None or None in tmp_shape:
        sample_shapes, _ = _populate_sample_shapes(
            chunk_engine, sample_shapes, index, sample_shape_provider, flatten=False
        )
        sample_ndim = sample_shapes.shape[1]
    else:
        sample_shapes[:] = tmp_shape

    squeeze_dims = _apply_deeper_indexing(
        sample_shapes, num_samples, sample_index
    )
    tmp_shape = _sample_shapes_to_shape(sample_shapes, squeeze_dims, sample_ndim)

    if index_0.subscriptable():
        tmp_shape = (num_samples, *tmp_shape)  # type: ignore

    return tmp_shape  # type: ignore


def shapes(
        chunk_engine,
        index: Index,
        sample_shape_provider: Optional[Callable] = None,
        convert_bad_to_list: bool = True,
):
    """Returns the shapes of samples. """
    if len(index) > 1:
        raise IndexError("`.shapes` only accepts indexing on the primary axis.")

    index_0 = index.values[0]
    num_samples, sample_ndim = _get_total_samples_and_sample_ndim(chunk_engine, index_0)

    sample_shapes = np.zeros((num_samples, sample_ndim), dtype=np.int32)

    if (
            index.is_trivial()
            or chunk_engine.tensor_meta.min_shape == chunk_engine.tensor_meta.max_shape
            or num_samples == 0
    ):
        tmp_shape = chunk_engine.shape_interval(index).astuple()[1:]
    else:
        tmp_shape = None

    if tmp_shape is None or None in tmp_shape:
        sample_shapes, bad_shapes = _populate_sample_shapes(
            chunk_engine,
            sample_shapes,
            index,
            sample_shape_provider,
            flatten=chunk_engine.is_sequence is True,
        )
        # convert to list if grayscale images were stored as (H, W) instead of (H, W, 1)
        if bad_shapes and convert_bad_to_list:
            sample_shapes = sample_shapes.tolist()
            for i in bad_shapes:
                sample_shapes[i] = sample_shapes[i][:-1]
        if chunk_engine.is_sequence:
            sample_shapes = _group_flat_shapes(
                chunk_engine, sample_shapes, index_0, sample_ndim
            )
    else:
        sample_shapes[:] = tmp_shape

    return sample_shapes


def read_shape_for_sample(
        chunk_engine,
        global_sample_index: int,
) -> Tuple[int, ...]:
    """Read the shape of a sample. """
    enc = chunk_engine.chunk_id_encoder
    if chunk_engine.is_tiled_sample(global_sample_index):
        return chunk_engine.get_tile_encoder.get_sample_shape(global_sample_index)
    local_sample_index = enc.translate_index_relative_to_chunks(global_sample_index)
    if chunk_engine.is_video:
        chunk_id = enc[global_sample_index][0]
        chunk = _get_video_chunk(chunk_engine, chunk_id)[0]
    else:
        chunk_id, _, worst_case_header_size = chunk_engine.get_chunk_info(
            global_sample_index, fetch_chunks=False
        )
        chunk = chunk_engine.get_chunk_from_chunk_id(
            chunk_id, partial_chunk_bytes=worst_case_header_size
        )
    return tuple(map(int, chunk.shapes_encoder[local_sample_index]))


def _populate_sample_shapes(
        chunk_engine,
        sample_shapes: np.ndarray,
        index: Index,
        sample_shape_provider: Optional[Callable] = None,
        flatten: bool = False,
):
    sample_indices = list(
        index.values[0].indices(chunk_engine.sequence_length or chunk_engine.num_samples)
    )
    sample_ndim = chunk_engine.ndim() - 1
    bad_shapes = []
    offset = 0

    for i, idx in enumerate(sample_indices):
        if chunk_engine.tensor_meta.htype in ("text", "json"):
            tmp_shape = (1,)
        elif sample_shape_provider:
            tmp_shape = _get_sample_shape_from_provider(
                chunk_engine, sample_shape_provider, idx, index.values[1:], flatten
            )
        else:
            tmp_shape = read_shape_for_sample(chunk_engine, idx)  # type: ignore
            # if link verification was not done
            if len(tmp_shape) > sample_ndim:
                sample_ndim = len(tmp_shape)
                sample_shapes = np.zeros((len(sample_indices), sample_ndim), dtype=np.int32)

        if flatten:
            assert chunk_engine.sequence_encoder is not None
            # fill sample shapes with sequence item shapes, no nesting
            start, end = chunk_engine.sequence_encoder[idx]
            length = end - start
            sample_shapes[offset: offset + length] = tmp_shape
            offset += length
        else:
            try:
                sample_shapes[i] = tmp_shape
            except ValueError:
                # Backwards compatibility for old datasets with
                # grayscale images stored as (H, W) instead of (H, W, 1)
                if len(tmp_shape) == 2 and sample_shapes.shape[1] == 3:
                    sample_shapes[i] = tmp_shape + (1,)
                    bad_shapes.append(i)
    return sample_shapes, bad_shapes


def _get_sample_shape_from_provider(
    chunk_engine, sample_shape_provider, idx, sample_index, flatten
):
    try:
        tmp_shape = sample_shape_provider(idx)  # type: ignore
    except (
        IndexError
    ):  # Happens during transforms, sample shape tensor is not populated yet
        tmp_shape = read_shape_for_sample(chunk_engine, idx)  # type: ignore

    if isinstance(tmp_shape, tuple) and tmp_shape == ():
        tmp_shape = (0,)
    if chunk_engine.is_sequence and not flatten:
        tmp_shape = _merge_seq_shape(tmp_shape, sample_index)
    return tmp_shape


def _get_total_samples_and_sample_ndim(chunk_engine, index_0):
    """Returns total number of samples (including sequence items) and sample ndim using first index"""
    tensor_ndim = chunk_engine.ndim()
    if chunk_engine.is_sequence:
        sample_indices = list(index_0.indices(chunk_engine.sequence_length))
        num_samples = sum(
            map(
                lambda x: x[1] - x[0],
                [chunk_engine.sequence_encoder[i] for i in sample_indices],
            )
        )
        sample_ndim = tensor_ndim - 2
    else:
        num_samples = index_0.length(chunk_engine.num_samples)
        sample_ndim = tensor_ndim - 1
    return num_samples, sample_ndim


def _group_flat_shapes(chunk_engine, sample_shapes, index_0, sample_ndim):
    """Groups shapes of flattened sequence items"""
    sample_indices = list(index_0.indices(chunk_engine.sequence_length))
    num_samples = len(sample_indices)
    seq_item_length = chunk_engine.sequence_encoder[sample_indices[0]]
    seq_item_length = seq_item_length[1] - seq_item_length[0]
    # Try reshape to (num_samples, seq_item_length, sample_ndim)
    try:
        if isinstance(sample_shapes, list):
            raise ValueError
        sample_shapes = sample_shapes[np.newaxis, :].reshape(
            num_samples, seq_item_length, sample_ndim
        )
        return sample_shapes
    except ValueError:
        sample_shapes_list = []
        offset = 0
        for _, idx in enumerate(sample_indices):
            start, end = chunk_engine.sequence_encoder[idx]
            length = end - start
            sample_shapes_list.append(sample_shapes[offset: offset + length])
            offset += length
        return sample_shapes_list


def _merge_seq_shape(tmp_shape, sample_index):
    """Merges shapes of sequence items into one shape"""
    if sample_index and not sample_index[0].subscriptable():
        tmp_shape = (1, *tuple(tmp_shape[sample_index[0].value].tolist()))  # type: ignore
    else:
        is_same = np.all(tmp_shape == tmp_shape[0, :], axis=0)  # type: ignore
        tmp_shape = (len(tmp_shape),) + (
                tuple(
                    int(tmp_shape[0, i])  # type: ignore
                    if is_same[i]  # type: ignore
                    else -1
                    for i in range(tmp_shape.shape[1])  # type: ignore
                )
                or (1,)
        )
    return tmp_shape


def _apply_deeper_indexing(sample_shapes, num_samples, sample_index):
    """Applies rest of the indexing to the sample shapes. Inplace operation."""
    squeeze_dims = set()
    for _, sample_shape in enumerate(sample_shapes[:num_samples]):
        for j, index in enumerate(sample_index):
            if index.subscriptable():
                if sample_shape[j] != -1:
                    sample_shape[j] = index.length(sample_shape[j])
            else:
                squeeze_dims.add(j)
    return squeeze_dims


def _sample_shapes_to_shape(sample_shapes, squeeze_dims, sample_ndim):
    is_same = np.all(sample_shapes == sample_shapes[0, :], axis=0)
    tmp_shape = [  # type: ignore
        int(sample_shapes[0, i])
        if sample_shapes[0, i] != -1 and is_same[i]
        else None
        for i in range(sample_ndim)
    ]

    return tuple(tmp_shape[i] for i in range(len(tmp_shape)) if i not in squeeze_dims)


def _get_video_chunk(chunk_engine, chunk_id, copy: bool = False):
    """Returns video chunks. Chunk will contain presigned url to the video instead of data if the chunk is large."""
    chunk_name = ChunkIdEncoder.name_from_id(chunk_id)
    chunk_key = get_chunk_key(chunk_engine.key, chunk_name)

    base_storage = chunk_engine.base_storage
    stream = False

    from muller.core.storage import RomaProvider  # Sherry: might have recursive import
    if isinstance(base_storage, RomaProvider):
        chunk_size = base_storage.get_object_size(chunk_key)
        stream = chunk_size > chunk_engine.min_chunk_size

    if stream:
        chunk = chunk_engine.cache.get_muller_object(chunk_key, chunk_engine.chunk_class, meta=chunk_engine.chunk_args,
                                             url=True)
    else:
        chunk = chunk_engine.cache.get_muller_object(chunk_key, chunk_engine.chunk_class, meta=chunk_engine.chunk_args)
    chunk.key = chunk_key
    chunk.id = chunk_id
    if copy and not chunk_engine.chunk_in_target_commit(chunk_name, chunk_engine.commit_id):
        chunk = chunk_engine.copy_chunk(chunk)
    return chunk, stream

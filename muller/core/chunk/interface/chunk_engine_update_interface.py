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

import warnings
from collections.abc import Iterable
from itertools import chain
from typing import List, Sequence, Union
from typing import (
    Optional
)

import numpy as np

import muller
from muller.constants import CHUNK_UPDATE_WARN_PORTION, MAX_WORKERS_FOR_CHUNK_ENGINE
from muller.core.chunk.base_chunk import InputSample
from muller.core.index.index import Index, IndexEntry
from muller.core.tiling.deserialize import coalesce_tiles, translate_slices
from muller.core.tiling.serialize import break_into_tiles
from muller.util.casting import intelligent_cast
from muller.util.exceptions import DynamicTensorNumpyError, SampleUpdateError
from muller.util.exceptions import UnAuthorizationError
from muller.util.keys import get_tensor_commit_chunk_map_key


def update(
        chunk_engine,
        index: Index,
        samples: Union[np.ndarray, Sequence[InputSample], InputSample],
        operator: Optional[str] = None,
):
    """Update the chunks"""
    cmap = chunk_engine.commit_chunk_map
    if cmap is not None:
        # Sherry: import risk here!!
        from muller.core.version_control.commit_chunk_map import CommitChunkMap
        cmap = CommitChunkMap.frombuffer(cmap.tobytes())
    try:
        _update(
            chunk_engine,
            index,
            samples,
            operator,
        )
    except UnAuthorizationError as e:
        raise e from None
    except Exception as e:
        if cmap is not None:
            key = get_tensor_commit_chunk_map_key(chunk_engine.key, chunk_engine.commit_id)
            chunk_engine.meta_cache[key] = cmap
            chunk_engine.commit_chunk_map = cmap
            chunk_engine.meta_cache.register_muller_object(key, cmap)
        raise SampleUpdateError(chunk_engine.name) from e


def _update(
        chunk_engine,
        index: Index,
        samples: Union[np.ndarray, Sequence[InputSample], InputSample],
        operator: Optional[str] = None,
        update_commit_diff: bool = True,
):
    """Update data at `index` with `samples`."""
    chunk_engine.cached_data = None
    initial_autoflush = chunk_engine.cache.autoflush
    chunk_engine.cache.autoflush = False
    try:
        if operator is not None:
            return _update_with_operator(chunk_engine, index, samples, operator)
        index_length = index.length(chunk_engine.num_samples)
        samples = _make_sequence(samples, index_length)
        if chunk_engine.tensor_meta.htype == "class_label":
            samples = chunk_engine.convert_class_labels(samples)
        nbytes_after_updates: List[int] = []
        global_sample_indices = tuple(index.values[0].indices(chunk_engine.num_samples))
        for i, sample in enumerate(samples):  # type: ignore
            sample = None if isinstance(sample, list) and len(sample) == 0 else sample
            global_sample_index = global_sample_indices[i]  # Sherry:!
            if chunk_engine.is_tiled_sample(global_sample_index):
                _update_tiled_sample(
                    chunk_engine, global_sample_index, index, sample, nbytes_after_updates
                )
            else:
                _update_non_tiled_sample(
                    chunk_engine, global_sample_index, index, sample, nbytes_after_updates
                )
            if update_commit_diff:
                chunk_engine.commit_diff.update_data(global_sample_index)
            chunk_min, chunk_max = chunk_engine.min_chunk_size, chunk_engine.max_chunk_size
            _check_suboptimal_chunks(nbytes_after_updates, chunk_min, chunk_max)

    finally:
        chunk_engine.cache.autoflush = initial_autoflush
        chunk_engine.cache.maybe_flush()
    return samples


def _make_sequence(samples, index_length: int):
    """Ensure `samples` is a sequence of length `index_length`."""
    # If samples already looks like a sequence, extract its length
    is_seq = isinstance(samples, Sequence) and not isinstance(samples, (str, bytes))

    if not is_seq:
        samples = [samples]

    # Special case for index_length == 1: force a single-element list
    if index_length == 1:
        samples = samples[:1]  # ensure length == 1
    elif len(samples) != index_length:
        raise ValueError(
            f"Index length ({index_length}) must equal samples length ({len(samples)})."
        )

    return samples


def _check_suboptimal_chunks(chunks_nbytes_after_updates: List[int], min_chunk_size: int, max_chunk_size: int):
    upper_warn_threshold = max_chunk_size * (CHUNK_UPDATE_WARN_PORTION + 1)
    lower_warn_threshold = min_chunk_size * (1 - CHUNK_UPDATE_WARN_PORTION)

    for nbytes in chunks_nbytes_after_updates:
        if nbytes > upper_warn_threshold or nbytes < lower_warn_threshold:
            warnings.warn(
                "After update, some chunks were suboptimal. Be careful when doing lots of updates that modify the "
                "sizes of samples by a large amount, these can heavily impact read performance!"
            )
            break


def _update_with_operator(
        chunk_engine,
        index: Index,
        samples: Union[np.ndarray, Sequence[InputSample], InputSample],
        operator: str,
):
    """Update data at `index` with the output of elem-wise operation with samples"""
    try:
        if isinstance(samples, muller.core.tensor.Tensor):
            samples = samples.numpy()

        def _get_index_pairs(tmp_index):
            if len(tmp_index) > 1:
                tmp_index1 = Index(index.values[:1])
                tmp_index2 = Index(index.values[1:])
            else:
                tmp_index1 = index
                tmp_index2 = None
            return tmp_index1, tmp_index2

        index1, index2 = _get_index_pairs(index)

        arr = chunk_engine.protected_numpy(index1, use_data_cache=False)
        view = arr
        if index2:
            for v in index2.values:
                view = view[v.value]  # type: ignore
    except DynamicTensorNumpyError as e:
        raise NotImplementedError(
            "Inplace update operations are not available for dynamic tensors yet."
        ) from e
    tensor_meta = chunk_engine.tensor_meta

    dt, ht = tensor_meta.dtype, tensor_meta.htype
    samples = intelligent_cast(samples, dt, ht)
    getattr(view, operator)(samples)
    _ = _update(chunk_engine, index1, arr)


def _update_tiled_sample(
        chunk_engine, global_sample_index: int, index: Index, sample, nbytes_after_updates
):
    if len(index.values) == 1:
        _replace_tiled_sample(chunk_engine, global_sample_index, sample)
        return
    chunk_ids = chunk_engine.chunk_id_encoder[global_sample_index]
    sample_shape = chunk_engine.tile_encoder.get_sample_shape(global_sample_index)
    tile_shape = chunk_engine.tile_encoder.get_tile_shape(global_sample_index)
    ordered_tile_ids = np.array(chunk_ids).reshape(
        chunk_engine.tile_encoder.get_tile_layout_shape(global_sample_index)
    )
    tiles_index, sample_index = translate_slices(
        [v.value for v in index.values[1:]], sample_shape, tile_shape  # type: ignore
    )
    required_tile_ids = ordered_tile_ids[tiles_index]
    new_tiles, chunk_ids = _update_tiled_sample_prepare(chunk_engine, required_tile_ids, tile_shape, sample_index,
                                                        sample, chunk_engine.tile_encoder, global_sample_index)
    _update_tiled_sample_process(chunk_engine, chunk_ids, new_tiles)


def _update_tiled_sample_prepare(chunk_engine, required_tile_ids, tile_shape, sample_index, sample,
                                 tile_enc, global_sample_index):

    def _read_tile_from_chunk(chunk_id, tmp_chunk_engine):
        """从 chunk_id 获取 chunk，再读取第 0 个样本（tile 模式）并返回该 sample。"""
        chunk = tmp_chunk_engine.get_chunk_from_chunk_id(chunk_id, copy=True)
        return chunk.read_sample(0, is_tile=True)

    tiles = [_read_tile_from_chunk(cid, chunk_engine) for cid in required_tile_ids]
    tiles = np.array(tiles, dtype=object)

    current_sample = coalesce_tiles(tiles, tile_shape, None, chunk_engine.tensor_meta.dtype)
    new_sample = current_sample
    new_sample[sample_index] = sample
    new_tiles = break_into_tiles(
        new_sample, tile_enc.get_tile_shape(global_sample_index)
    )
    chunk_ids = required_tile_ids
    return new_tiles, chunk_ids


def _update_tiled_sample_process(chunk_engine, chunk_ids, new_tiles):
    for chunk_id, tile in zip(chunk_ids.reshape(-1), new_tiles.reshape(-1)):
        chunk = chunk_engine.get_chunk_from_chunk_id(int(chunk_id), copy=True)
        curr_shape = chunk.shapes_encoder[-1]
        assert curr_shape == tile.shape, (curr_shape, tile.shape)
        chunk.update_sample(0, tile)
        if (
                chunk_engine.active_updated_chunk is not None
                and chunk_engine.active_updated_chunk.key != chunk.key  # type: ignore
        ):
            chunk_engine.write_chunk_to_storage(chunk_engine.active_updated_chunk)
        chunk_engine.active_updated_chunk = chunk


def _replace_tiled_sample(chunk_engine, global_sample_index: int, sample):
    new_chunk_ids, tiles = chunk_engine.samples_to_chunks(
        [sample], start_chunk=None, register=False
    )
    chunk_engine.chunk_id_encoder.replace_chunks_for_tiled_sample(
        global_sample_index, new_chunk_ids
    )
    if tiles:
        chunk_engine.tile_encoder.entries[global_sample_index] = tiles[0]
    else:
        del chunk_engine.tile_encoder.entries[global_sample_index]


def _update_non_tiled_sample(
    chunk_engine, global_sample_index: int, index: Index, sample, nbytes_after_updates
):
    enc = chunk_engine.chunk_id_encoder
    chunk = chunk_engine.get_chunks_for_sample(global_sample_index, copy=True)[0]
    local_sample_index = enc.translate_index_relative_to_chunks(global_sample_index)

    if len(index.values) <= 1:
        chunk.update_sample(local_sample_index, sample)
    else:
        orig_sample = chunk.read_sample(local_sample_index, copy=True)
        sample = np.array(sample)
        lhs = orig_sample[tuple(e.value for e in index.values[1:])]
        if lhs.ndim > sample.ndim:
            sample = np.expand_dims(sample, tuple(range(sample.ndim, lhs.ndim)))
        lhs[:] = sample
        chunk.update_sample(local_sample_index, orig_sample)
    if (
        chunk_engine.active_updated_chunk is not None
        and chunk_engine.active_updated_chunk.key != chunk.key  # type: ignore
    ):
        chunk_engine.write_chunk_to_storage(chunk_engine.active_updated_chunk)
    chunk_engine.active_updated_chunk = chunk

    # only care about deltas if it isn't the last chunk
    if chunk.key != chunk_engine.last_chunk_key:  # type: ignore
        nbytes_after_updates.append(chunk.nbytes)

    chunk_engine.check_rechunk(
        chunk, chunk_row=enc.__getitem__(global_sample_index, True)[0][1]
    )



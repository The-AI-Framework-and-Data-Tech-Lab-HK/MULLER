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
        (_sequence_update if chunk_engine.is_sequence else _update)(
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


def sequence_numpy(
    chunk_engine,
    index: Index,
    aslist: bool = False,  # aslist可能会带来返回类型不一致的问题！
    use_data_cache: bool = True,
    fetch_chunks: bool = False,
    max_workers: int = MAX_WORKERS_FOR_CHUNK_ENGINE,
    continuous: bool = False,
    full: bool = False,
):
    """Returns the numpy array of sequence. """
    arr = chunk_engine.protected_numpy(
        _get_flat_index_from_sequence_index(chunk_engine, index),
        aslist=aslist,
        use_data_cache=use_data_cache,
        fetch_chunks=fetch_chunks,
        max_workers=max_workers,
        continuous=continuous,
        full=full,
    )
    if chunk_engine.num_samples == 0:
        return arr
    if isinstance(arr, np.ndarray) and arr.size == 0:
        return chunk_engine.get_empty_sample()
    if index.subscriptable_at(0) and index.subscriptable_at(1):
        item_lengths = []
        assert chunk_engine.sequence_encoder is not None
        for i in index.values[0].indices(chunk_engine.sequence_length):
            item_length = index.length_at(
                1, -int(np.subtract(*chunk_engine.sequence_encoder[i]))
            )
            item_lengths.append(item_length)

        if aslist:
            ret = []
            for item_length in item_lengths:
                ret.append(arr[:item_length])
                arr = arr[item_length:]
            return ret

        if len(set(item_lengths)) > 1:
            raise DynamicTensorNumpyError(chunk_engine.name, index, "shape")
        try:
            return arr.reshape(  # type: ignore
                index.length_at(0, chunk_engine.sequence_length), -1, *arr.shape[1:]  # type: ignore
            )
        except ValueError as ve:
            raise DynamicTensorNumpyError(chunk_engine.name, index, "shape") from ve

    return arr


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

    if len(index.values) <= 1 + int(chunk_engine.is_sequence):
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


def _sequence_update(
    chunk_engine,
    index: Index,
    samples: Union[np.ndarray, Sequence[InputSample], InputSample],
    operator: Optional[str] = None,
):
    flat_idx = _get_flat_index_from_sequence_index(chunk_engine, index)
    flat_samples = _get_flat_samples_for_sequence_update(chunk_engine, samples, index)
    flat_verified_samples: List = _update(
        chunk_engine,
        flat_idx,
        flat_samples,
        operator,
        update_commit_diff=False,
    )
    i = 0
    if chunk_engine.tensor_meta.htype == "class_label":
        samples = chunk_engine.convert_class_labels(samples)
    if flat_verified_samples:
        verified_samples = []
        for sample in samples:  # type: ignore
            verified_sample = []
            if isinstance(sample, Iterable):
                for _ in sample:  # type: ignore
                    verified_sample.append(flat_verified_samples[i])
                    i += 1
                verified_samples.append(verified_sample)
            else:
                verified_samples.append(flat_verified_samples[i])
                i += 1

    list(
        map(
            chunk_engine.commit_diff.update_data,
            index.values[0].indices(chunk_engine.sequence_length),
        )
    )


def _get_flat_index_from_sequence_index(chunk_engine, index: Index) -> Index:
    if len(index) == 1:
        index = Index([index.values[0], IndexEntry()])
    if index.values[0].is_trivial() and index.values[1].is_trivial():
        return Index([IndexEntry(), *index.values[2:]])
    if index.subscriptable_at(0) or index.subscriptable_at(1):
        idx0 = _translate_2d_index(chunk_engine, index.values[0], index.values[1])
        return Index([idx0, *index.values[2:]])  # type: ignore
    return Index(
        [
            IndexEntry(
                chunk_engine.sequence_encoder[index.values[0].value][0]  # type: ignore
                + index.values[1].value
            ),
            *index.values[2:],
        ]
    )


def _translate_2d_index(
    chunk_engine, x: Optional[IndexEntry] = None, y: Optional[IndexEntry] = None
) -> IndexEntry:
    x = x or IndexEntry()
    y = y or IndexEntry()
    _item_length = chunk_engine.sequence_item_length
    if _item_length is None:

        def idx0_gen():
            for i in x.indices(chunk_engine.sequence_length):
                s, e = chunk_engine.sequence_encoder[i]
                for j in y.indices(e - s):
                    yield s + j

    else:

        def idx0_gen():
            for i in x.indices(chunk_engine.sequence_length):
                for j in y.indices(_item_length):
                    yield i * _item_length + j

    assert chunk_engine.sequence_encoder is not None

    def compute_idx0_len(x, y, tmp_chunk_engine, item_length=None):
        if item_length is None:
            total = 0
            for i in x.indices(tmp_chunk_engine.sequence_length):
                offset = -np.subtract(*tmp_chunk_engine.sequence_encoder[i])
                total += y.length(offset)
            return total

        return x.length(tmp_chunk_engine.sequence_length) * y.length(item_length)

    idx0_gen.__len__ = compute_idx0_len(x, y, chunk_engine, _item_length)


    return IndexEntry(idx0_gen)  # type: ignore


def _get_flat_samples_for_sequence_update(chunk_engine, samples, index: Index):
    ndim = chunk_engine.ndim(index)
    if isinstance(samples, np.ndarray):
        if index.subscriptable_at(0) and index.subscriptable_at(1):
            tmp_diff = ndim - samples.ndim
            if tmp_diff < 0:
                samples, tmp_diff = samples.reshape(samples.shape[-ndim:]), 0
            if tmp_diff > 1:
                return samples.reshape(1, *samples.shape).repeat(
                    _translate_2d_index(chunk_engine, *index.values[:2]).length(None), 0  # type: ignore
                )
            if tmp_diff == 1:
                return (
                    samples.reshape(1, *samples.shape)
                    .repeat(index.length_at(0, chunk_engine.sequence_length), 0)
                    .reshape(-1, *samples.shape[1:])
                )
            return samples.reshape(-1, *samples.shape[2:])
        return samples
    if isinstance(samples, (str, bytes)):  # treated as scalars
        return samples
    if isinstance(samples, Iterable):
        # Note: broadcasting is not supported here
        if index.subscriptable_at(0) and index.subscriptable_at(1):
            return list(chain(*samples))
        return samples

    return samples  # scalars

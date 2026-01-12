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
    Union,
    List
)

import numpy as np

from muller.core.chunk.base_chunk import BaseChunk
from muller.core.meta.encode.base_encoder import LAST_SEEN_INDEX_COLUMN
from muller.core.meta.encode.chunk_id import ChunkIdEncoder
from muller.util.keys import get_chunk_key


def pop(
        chunk_engine,
        indices: Optional[Union[int, List[int]]] = None,
        sample_id_tensor=None,
        rechunk: Optional[bool] = False,
):
    """Pop from the chunk."""
    if indices is None:
        indices = [chunk_engine.tensor_length - 1]

    if not isinstance(indices, list):
        indices = [indices]

    chunk_engine.cached_data = None
    initial_autoflush = chunk_engine.cache.autoflush
    chunk_engine.cache.autoflush = False

    def update_links_and_encoders(private_chunk_engine, private_idxs):
        """Update linked tensors and sample level encoders"""
        if isinstance(private_idxs, int):
            private_idxs = [private_idxs]
        for idx in private_idxs:
            private_chunk_engine.commit_diff.pop(
                idx,
                sample_id_tensor[idx].numpy().item()
                if sample_id_tensor is not None
                else None,
            )
        for idx in private_idxs:
            if private_chunk_engine.is_sequence:
                private_chunk_engine.sequence_encoder.pop(idx)

    if chunk_engine.is_sequence:
        assert chunk_engine.sequence_encoder is not None
        item_lengths = [
            [index, -np.subtract(*chunk_engine.sequence_encoder[index])]
            for index in sorted(indices)
        ]
        flat_indices: List[int] = []
        for index in indices:
            flat_indices.extend(range(*chunk_engine.sequence_encoder[index]))
        indices = flat_indices

    for chunk_id, row, idxs, is_tile in chunk_engine.load_chunks(indices, reverse=True):
        idxs = list(reversed(idxs))
        if chunk_engine.is_sequence:
            num_flat_samples = len(idxs)
            while item_lengths and num_flat_samples >= item_lengths[-1][1]:
                num_flat_samples -= item_lengths[-1][1]
                idx_2d, _ = item_lengths.pop()
                update_links_and_encoders(chunk_engine, idx_2d)

            if num_flat_samples:
                item_lengths[-1][1] -= num_flat_samples
        else:
            update_links_and_encoders(chunk_engine, idxs)
        _pop_samples(chunk_engine, chunk_id, row, idxs, is_tile, rechunk)

    chunk_engine.cache.autoflush = initial_autoflush
    chunk_engine.cache.maybe_flush()


def _pop_samples(
    chunk_engine,
    chunk_id: int,
    row: int,
    idxs: List[int],
    is_tile: bool,
    rechunk: bool = False,
):
    """Pop samples"""
    if not idxs:
        return

    enc = chunk_engine.chunk_id_encoder

    if is_tile:
        assert len(idxs) == 1, "Tile chunks should only have one sample"
        delete = True
        chunk_ids, _, _ = enc.pop(idxs[0])
    else:
        prev = -1 if row == 0 else enc.array[row - 1][LAST_SEEN_INDEX_COLUMN]
        num_samples_in_chunk = (
            enc.array[row][LAST_SEEN_INDEX_COLUMN] - prev
            if prev != -1
            else enc.array[row][LAST_SEEN_INDEX_COLUMN] + np.abs(prev)
        ).item()
        num_samples_indexed = len(idxs)

        assert num_samples_indexed <= num_samples_in_chunk
        delete = num_samples_in_chunk == num_samples_indexed

        chunk_ids = [chunk_id]

    chunk_to_update = (
        chunk_engine.get_chunk_from_chunk_id(chunk_ids[0], copy=True, row=row)
        if not delete
        else None
    )
    for idx in idxs:
        _pop_from_chunk(chunk_engine, chunk_to_update, row, idx)

    _pop_delete_process(chunk_engine, delete, is_tile, enc, row, chunk_ids, chunk_to_update, rechunk)


def _pop_delete_process(chunk_engine, delete, is_tile, enc, row, chunk_ids, chunk_to_update, rechunk):
    if delete:
        # tile rows already deleted
        if not is_tile:
            enc.delete_rows([row])
        for temp_chunk_id in chunk_ids:
            chunk_name = ChunkIdEncoder.name_from_id(temp_chunk_id)
            if chunk_engine.chunk_in_target_commit(chunk_name, chunk_engine.commit_id):
                chunk_key = get_chunk_key(chunk_engine.key, chunk_name)
                _check_remove_active_chunks(chunk_engine, chunk_key)
                try:
                    del chunk_engine.cache[chunk_key]
                except KeyError:
                    pass
    else:
        assert chunk_to_update is not None
        if rechunk:
            chunk_engine.check_rechunk(chunk_to_update, row)
        if (
            chunk_engine.active_updated_chunk is not None
            and chunk_engine.active_updated_chunk.key != chunk_to_update.key  # type: ignore
        ):
            chunk_engine.write_chunk_to_storage(chunk_engine.active_updated_chunk)
        chunk_engine.active_updated_chunk = chunk_to_update


def _pop_from_chunk(chunk_engine, chunk: Optional[BaseChunk], row: int, global_idx: int):
    """Pop sample from chunk. If chunk is ``None``, only updates tensor meta, chunk id encoder and tile encoder."""
    if chunk:
        local_idx = chunk_engine.translate_to_local_index(global_idx, row)
        chunk.pop(local_idx)
        chunk_engine.chunk_id_encoder.encoded[row:, LAST_SEEN_INDEX_COLUMN] -= 1
        chunk_engine.chunk_id_encoder.is_dirty = True
    chunk_engine.tensor_meta.pop(global_idx)
    if chunk_engine.enable_tile_encoder:
        del chunk_engine.tile_encoder[global_idx]


def _check_remove_active_chunks(chunk_engine, chunk_key):
    """Check and remove active chunks."""
    if (
            chunk_engine.active_appended_chunk is not None
            and chunk_engine.active_appended_chunk.key == chunk_key
    ):
        chunk_engine.active_appended_chunk = None
    if (
            chunk_engine.active_updated_chunk is not None
            and chunk_engine.active_updated_chunk.key == chunk_key
    ):
        chunk_engine.active_updated_chunk = None

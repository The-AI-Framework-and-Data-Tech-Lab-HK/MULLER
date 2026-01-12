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

from typing import Optional

from tqdm import tqdm

import muller
from muller.client.log import logger
from muller.core.chunk.base_chunk import BaseChunk
from muller.util.exceptions import SampleAppendError
from muller.util.exceptions import UnAuthorizationError
from muller.util.keys import get_chunk_key


def extend(
        chunk_engine,
        samples,
        progressbar: bool = False,
        pg_callback=None,
        ignore_errors: bool = False,
        is_uuid: bool = False
):
    """Extend the chunks"""
    try:
        assert not (progressbar and pg_callback)
        if not chunk_engine.write_initialization_done:
            chunk_engine.write_initialization_done = True

        initial_autoflush = chunk_engine.cache.autoflush
        chunk_engine.cache.autoflush = False

        if chunk_engine.is_sequence:
            _extend_sequence(
                chunk_engine,
                samples,
                progressbar,
                ignore_errors,
            )
        else:
            # extend 大部分情况下的必经之路
            _extend(
                chunk_engine,
                samples,
                progressbar,
                pg_callback=pg_callback,
                ignore_errors=ignore_errors,
                is_uuid=is_uuid
            )

        chunk_engine.cache.autoflush = initial_autoflush
        chunk_engine.cache.maybe_flush()
    except UnAuthorizationError as e:
        raise e from None
    except Exception as e:
        num_samples = len(samples)
        chunk_engine.pop(list(range(num_samples, chunk_engine.tensor_length)))
        raise SampleAppendError(chunk_engine.name) from e


def pad_and_append(
    chunk_engine,
    num_samples_to_pad: int,
    value,
):
    """Pads the tensor with empty samples and appends value at the end."""
    chunk_engine.start_chunk = _last_appended_chunk(chunk_engine)  # type: ignore
    if num_samples_to_pad > 0:
        logger.warning(
                "Needs to pad the tensor with empty samples. Not implemented. "
            )
    chunk_engine.extend([value])


def _extend_sequence(
    chunk_engine, samples, progressbar, ignore_errors
):
    samples = tqdm(samples) if progressbar else samples
    verified_samples = []
    num_samples_added = 0
    for sample in samples:
        try:
            if sample is None:
                sample = []
            sample = _extend(
                chunk_engine,
                sample,
                progressbar=False,
                update_commit_diff=False
            )
            verified_samples.append(sample)
            chunk_engine.sequence_encoder.register_samples(len(sample), 1)
            chunk_engine.commit_diff.add_data(1)
            num_samples_added += 1
        except Exception:
            if ignore_errors:
                continue
            raise


def _extend(
    chunk_engine,
    samples,
    progressbar,
    pg_callback=None,
    update_commit_diff=True,
    ignore_errors=False,
    is_uuid=False
):
    if isinstance(samples, muller.Tensor):
        samples = tqdm(samples) if progressbar else samples
        for sample in samples:
            _ = _extend(
                chunk_engine,
                [sample],
                progressbar=False,
                pg_callback=pg_callback,
                update_commit_diff=update_commit_diff,
                ignore_errors=ignore_errors
            )  # Sherry: optimize this
        return samples
    if len(samples) == 0:
        return samples
    samples = chunk_engine.sanitize_samples(samples, ignore_errors=ignore_errors)
    samples = chunk_engine.samples_to_chunks(
        samples,
        start_chunk=_last_appended_chunk(chunk_engine, allow_copy=False),
        register=True,
        progressbar=progressbar,
        update_commit_diff=update_commit_diff,
        pg_callback=pg_callback,
        return_samples=True,
        ignore_errors=ignore_errors,
        is_uuid=is_uuid
    )
    return samples


def _last_appended_chunk(chunk_engine, allow_copy=True) -> Optional[BaseChunk]:
    """Last appended chunk. """
    last_index = chunk_engine.num_samples - 1
    last_in_tile = False
    if chunk_engine.enable_tile_encoder:
        if last_index in chunk_engine.tile_encoder:
            last_in_tile = True
    if chunk_engine.num_chunks == 0 or last_in_tile:
        return None
    chunk_name = chunk_engine.last_appended_chunk_name
    chunk_key = get_chunk_key(chunk_engine.key, chunk_name)
    chunk = chunk_engine.get_chunk(chunk_key)
    chunk.key = chunk_key
    chunk.id = chunk_engine.last_appended_chunk_id
    if not chunk_engine.chunk_in_target_commit(chunk_name, chunk_engine.commit_id):
        if not allow_copy:
            return None
        chunk = chunk_engine.copy_chunk(chunk, row=-1)
    if (
            chunk_engine.active_appended_chunk is not None
            and chunk_engine.active_appended_chunk.key != chunk_key
    ):
        chunk_engine.write_chunk_to_storage(chunk_engine.active_appended_chunk)
    chunk_engine.active_appended_chunk = chunk
    return chunk
    
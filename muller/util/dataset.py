# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/util/dataset.py
#
# Modifications Copyright (c) 2026 Bingyu Liu

from typing import Optional, Sequence, List

import numpy as np

from muller.constants import TRANSFORM_RECHUNK_AVG_SIZE_BOUND, RANDOM_MINIMAL_CHUNK_SIZE
from muller.util.exceptions import ReadOnlyModeError, TensorDoesNotExistError


def try_flushing(ds):
    """Try flushing. """
    try:
        ds.storage.flush()
    except ReadOnlyModeError:
        pass


def need_rechunk(tensor) -> bool:
    """Returns whether tensor needs to be rechunked."""
    if tensor.meta.sample_compression or tensor.meta.chunk_compression:
        return False
    eng = tensor.chunk_engine
    if eng.num_chunks <= 1:
        return False
    avg = eng.get_avg_chunk_size() or 0
    return avg < TRANSFORM_RECHUNK_AVG_SIZE_BOUND * eng.min_chunk_size


def rechunk_one_tensor(ds, t_name: str, avg_bytes_per_sample: Optional[int] = None) -> str:
    """Rechunk one tensor chunk by chunk size."""
    try:
        tensor = ds[t_name]
    except TensorDoesNotExistError:
        return f"{t_name}\tSKIP(no tensor)"

    if not need_rechunk(tensor):
        return f"{t_name}\tOK(no need)"

    enc = tensor.chunk_engine.chunk_id_encoder
    engine = tensor.chunk_engine
    n_chunks = len(enc.encoded)

    if avg_bytes_per_sample and avg_bytes_per_sample > 0:
        _rechunk_process(enc, avg_bytes_per_sample, n_chunks, engine)
    else:
        row = 0
        while row < len(enc.encoded) - 1:
            encoded_before = len(enc.encoded)
            chunk_id = enc.encoded[row, 0]
            chunk = engine.get_chunk_from_chunk_id(chunk_id)
            engine.check_rechunk(chunk, row)
            if len(enc.encoded) == encoded_before:
                row += 1
    ds.storage.flush()
    return f"{t_name}\tRECHUNKED"


def _rechunk_process(enc, avg_bytes_per_sample, n_chunks, engine):
    cum_samples = enc.encoded[:, 1].astype(np.int64)
    order = np.argsort(cum_samples)
    sorted_chunk_samples = np.diff(np.insert(cum_samples[order], 0, 0))
    chunk_samples = np.empty_like(sorted_chunk_samples)
    chunk_samples[order] = sorted_chunk_samples
    est_bytes = chunk_samples * avg_bytes_per_sample

    rows_groups, ids_groups = _rechunk_loop(n_chunks, est_bytes, enc)

    engine.merge_regions(rows_groups, ids_groups)


def _rechunk_loop(n_chunks, est_bytes, enc):
    rows_groups: List[List[int]] = []
    ids_groups: List[List[int]] = []
    i = 0
    while i < n_chunks:
        if est_bytes[i] >= RANDOM_MINIMAL_CHUNK_SIZE:
            i += 1
            continue

        rows, ids = [i], [int(enc.encoded[i, 0])]
        total_est = int(est_bytes[i])
        j = i + 1
        while total_est < RANDOM_MINIMAL_CHUNK_SIZE and j < n_chunks:
            rows.append(j)
            ids.append(int(enc.encoded[j, 0]))
            total_est += int(est_bytes[j])
            j += 1

        if len(rows) > 1:
            rows_groups.append(rows)
            ids_groups.append(ids)
        i = j
    return rows_groups, ids_groups


def rechunk_if_necessary(ds):
    """Rechunk if necessary. """
    with ds:
        for tensor in ds.tensors:
            try:
                tensor = ds[tensor]
            # temp tensors
            except TensorDoesNotExistError:
                continue
            if not tensor.meta.sample_compression and not tensor.meta.chunk_compression:
                engine = tensor.chunk_engine
                num_chunks = engine.num_chunks
                if num_chunks > 1:
                    max_shape = tensor.meta.max_shape
                    if len(max_shape) > 0:
                        avg_chunk_size = engine.get_avg_chunk_size()
                        if (
                            avg_chunk_size is not None
                            and avg_chunk_size
                            < TRANSFORM_RECHUNK_AVG_SIZE_BOUND * engine.min_chunk_size
                        ):
                            enc = tensor.chunk_engine.chunk_id_encoder
                            row = 0
                            while row < len(enc.encoded) - 1:
                                encoded = enc.encoded
                                chunk_id = encoded[row, 0]
                                chunk = engine.get_chunk_from_chunk_id(chunk_id)
                                engine.check_rechunk(chunk, row)
                                # np.delete will replace enc._encoded with new array
                                # so this check works
                                rechunked = len(encoded) != len(enc.encoded)
                                if not rechunked:
                                    row += 1


def map_tensor_keys(dataset, tensor_keys: Optional[Sequence[str]] = None) -> List[str]:
    """Sanitizes tensor_keys if not None, else returns all the keys present in the dataset."""

    tensors = dataset.tensors

    if tensor_keys is None:
        tensor_keys = list(tensors)
    else:
        for t in tensor_keys:
            if t not in tensors:
                raise TensorDoesNotExistError(t)

        tensor_keys = list(tensor_keys)

    # Get full path in case of groups
    return [tensors[k].meta.name or tensors[k].key for k in tensor_keys]


_invalid_chars = {"[", "]", "@", ".", ",", "?", "!", "/", "\\", "#", "'", '"'}


def sanitize_tensor_name(temp_input: str) -> str:
    """Sanitize a string to be a valid tensor name

    Args:
        temp_input (str): A string that will be sanitized

    Returns:
        str: A string with the sanitized tensor name
    """
    return "".join("_" if c in _invalid_chars else c for c in temp_input)

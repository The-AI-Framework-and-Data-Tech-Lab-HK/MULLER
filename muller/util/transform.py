# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/util/transform.py
#
# Modifications Copyright (c) 2026 Bingyu Liu

import time
import traceback
import warnings
from collections import defaultdict
from json import JSONDecodeError
from os import urandom
from typing import Any, Dict, List, Tuple, Optional

import math
import numpy as np
import pandas as pd

import muller
from muller import Tensor
from muller.constants import TRANSFORM_RECHUNK_AVG_SIZE_BOUND, TRANSFORM_CHUNK_CACHE_SIZE, MB, \
    TRANSFORM_PROGRESSBAR_UPDATE_INTERVAL, DATASET_UUID_NAME
from muller.core.chunk.chunk_engine import ChunkEngine
from muller.core.index import Index
from muller.core.meta.tensor_meta import TensorMeta
from muller.core.storage import MemoryProvider, StorageProvider, LRUCache
from muller.core.transform.transform_dataset import TransformDataset
from muller.util.dataset import try_flushing
from muller.util.exceptions import InvalidInputDataError, TransformFailure
from muller.util.exceptions import (InvalidOutputDatasetError, AllSamplesSkippedError,
                                   TensorDoesNotExistError, InvalidTransformDataset, TensorMismatchError,
                                   TransformError, SampleAppendError)
from muller.util.keys import get_tensor_meta_key
from muller.util.remove_cache import get_base_storage
from muller.util.remove_cache import get_dataset_with_zero_size_cache
from muller.util.version_control import auto_checkout, load_meta, load_version_info


def check_transform_data_in(data_in, scheduler: str, overwrite: bool, batch_enable: bool = False) -> None:
    """Checks whether the data_in for a transform is valid or not."""
    if overwrite and batch_enable:
        raise InvalidInputDataError("In batch mode, data_out must be `muller.Dataset`, not `None`.")
    if not hasattr(data_in, "__getitem__"):
        raise InvalidInputDataError("__getitem__")
    if not hasattr(data_in, "__len__"):
        raise InvalidInputDataError("__len__")
    if isinstance(data_in, muller.Dataset):
        input_base_storage = get_base_storage(data_in.storage)
        if isinstance(input_base_storage, MemoryProvider) and scheduler not in [
            "serial",
            "threaded",
        ]:
            raise InvalidInputDataError(
                f"Transforms with data_in as a Dataset having base storage as MemoryProvider are only "
                f"supported in threaded and serial mode. Current mode is {scheduler}."
            )
    else:
        if batch_enable:
            first_length = None
            for i, sub_list in enumerate(data_in):
                if not hasattr(sub_list, "__getitem__") or not hasattr(sub_list, "__len__"):
                    raise InvalidInputDataError(
                        f"In batch mode, data_in[{i}] must be a list-like object with __getitem__ and __len__"
                    )

                if first_length is None:
                    first_length = len(sub_list)
                elif len(sub_list) != first_length:
                    raise InvalidInputDataError(
                        f"In batch mode, all sub-lists must have the same length. "
                        f"Expected {first_length}, but data_in[{i}] has length {len(sub_list)}"
                    )



def check_transform_ds_out(
    ds_out: muller.Dataset,
    scheduler: str,
    tmp_check_lengths: bool,
    read_only_ok: bool = False,
) -> None:
    """Checks whether the ds_out for a transform is valid or not."""
    if ds_out.read_only and not read_only_ok:
        raise InvalidOutputDatasetError
    tensors = list(ds_out.tensors)

    if tmp_check_lengths:
        for tensor in tensors:
            if len(ds_out[tensor]) != len(ds_out):
                raise InvalidOutputDatasetError(
                    "One or more tensors of the ds_out have different lengths. "
                    "Transform only supports ds_out having same number of samples for each tensor "
                    "(This includes empty datasets that have 0 samples per tensor)."
                )

    output_base_storage = get_base_storage(ds_out.storage)
    if isinstance(output_base_storage, MemoryProvider) and scheduler not in [
        "serial",
        "threaded",
    ]:
        raise InvalidOutputDatasetError(
            f"Transforms with ds_out having base storage as MemoryProvider are only supported in threaded "
            f"and serial mode. Current mode is {scheduler}."
        )


def len_data_in(data_in, batch_enable: bool = False):
    """Function to calculate the length of data_in"""
    if isinstance(data_in, muller.Dataset):
        return data_in.max_len
    if batch_enable:
        if data_in and len(data_in) > 0:
            return len(data_in[0])
        return 0
    return len(data_in)


def is_empty_transform_dataset(dataset: TransformDataset):
    """Checks if there is any data in the TransformDataset. Returns True if empty, False otherwise."""
    return all(len(dataset[tensor]) == 0 for tensor in dataset.tensors)


def sanitize_workers_scheduler(num_workers, scheduler):
    """Return num_workers and scheduler."""
    if num_workers <= 0:
        scheduler = "serial"
    num_workers = max(num_workers, 1)
    return num_workers, scheduler


def prepare_data_in(data_in, overwrite):
    """Prepare data input."""
    original_data_in = data_in
    if isinstance(data_in, muller.Dataset):
        try_flushing(data_in)
        if overwrite:
            auto_checkout(data_in)
        original_data_in = data_in
        data_in = get_dataset_with_zero_size_cache(data_in)
    return data_in, original_data_in


def check_checkpoint_interval(
    data_in, checkpoint_interval, num_workers, overwrite, verbose, batch_enable: bool = False
):
    """Validate the checkpoint interval."""
    if num_workers > 0 and checkpoint_interval % num_workers != 0:
        raise ValueError(
            "checkpoint_interval should be a multiple of num_workers if num_workers > 0"
        )
    if checkpoint_interval > len_data_in(data_in, batch_enable):
        raise ValueError(
            "checkpoint_interval should be less than or equal to the length of data_in"
        )
    if checkpoint_interval < len_data_in(data_in, batch_enable) / 10 and verbose:
        warnings.warn(
            "checkpoint_interval is less than 10% of the length of data_in, this can lead to too many commits, "
            "consider increasing checkpoint_interval."
        )
    if overwrite:
        raise ValueError(
            "checkpoint_interval > 0 and ds_out is None. Cannot checkpoint during inplace transform."
        )


def check_lengths(all_tensors_generated_length, skip_ok):
    """Guarantee that the length of all tensors generated is the same."""
    if skip_ok:
        return

    first_length = None
    for length in all_tensors_generated_length.values():
        if length == 0:
            continue
        if first_length is None:
            first_length = length
        elif length not in [0, first_length]:
            warnings.warn(
                "Length of all tensors generated is not the same, this may lead to unexpected behavior."
            )
            break


def append_uuid(dataset: TransformDataset):
    """
    Append dataset uuid to dataset length when append data with multi workers
    """
    uuids = np.frombuffer(urandom(8 * len(dataset)), dtype=np.uint64).reshape(-1)
    dataset[DATASET_UUID_NAME].append(uuids)



def transform_sample(
    sample: Any,
    pipeline,
    tensors
) -> TransformDataset:
    """Calls all the functions one after the other on a single sample.
    Can return 0 or more samples.

    Args:
        sample: The sample on which the pipeline of functions is to be applied.
        pipeline (Pipeline): The Sequence of functions to apply on the sample.
        tensors: List of tensors in output.

    Returns:
        TransformDataset: A transform dataset containing all the samples that were generated.
    """
    out = sample
    for index in range(len(pipeline)):
        transform_fn = pipeline.functions[index]
        fn, args, kwargs = transform_fn.func, transform_fn.args, transform_fn.kwargs

        if isinstance(out, TransformDataset):
            result = TransformDataset(tensors)
            for item in out:
                fn(item, result, *args, **kwargs)
                validate_transform_dataset(result)
            out = result
        else:
            result = TransformDataset(tensors)
            fn(out, result, *args, **kwargs)
            append_uuid(result)
            validate_transform_dataset(result)
            out = result
    return out


def validate_transform_dataset(dataset: TransformDataset):
    """Checks if the length of all the tensors is equal. Raises exception if not equal."""
    data = dataset.data
    lengths = [
        len(data[tensor])
        for tensor in data
        if len(data[tensor]) > 0
    ]
    if any(length != lengths[0] for length in lengths):
        raise InvalidTransformDataset(
            "The number of samples added to each tensor in transform should be the same."
        )


def close_states(compute_provider, pbar, pqueue):
    """Close the computer provider"""
    compute_provider.close()
    if pbar and hasattr(pbar, "close"):
        pbar.close()
    if pqueue and hasattr(pqueue, "close"):
        pqueue.close()


def create_slices(data_in, num_workers, batch_enable: bool = False):
    """Function to create slices."""
    size = math.ceil(len_data_in(data_in, batch_enable) / num_workers)
    if isinstance(data_in, Tensor):
        ret = [
            Tensor(data_in.key, data_in.dataset)[i * size: (i + 1) * size]
            for i in range(num_workers)
        ]
    else:
        if batch_enable:
            ret = []
            for i in range(num_workers):
                start_idx = i * size
                end_idx = min((i + 1) * size, len_data_in(data_in, batch_enable))
                slice_chunk = [
                    sublist[start_idx:end_idx]
                    for sublist in data_in
                ]
                ret.append(slice_chunk)
        else:
            ret = [data_in[i * size: (i + 1) * size] for i in range(num_workers)]

    if isinstance(data_in, muller.Dataset) and not batch_enable:
        for ds in ret:
            ds.version_state["full_tensors"] = {}
            _tensors = ds.version_state["full_tensors"]
            for tensor_key in data_in.version_state["tensor_names"].values():
                _tensors[tensor_key] = Tensor(tensor_key, ds)

    offsets = list(range(0, len_data_in(data_in, batch_enable), size))
    return ret, offsets


def delete_overwritten_chunks(old_chunk_paths, storage, overwrite):
    """Delete overwritten chunks."""
    if not overwrite:
        return

    storage.delete_multiple(old_chunk_paths)


def get_lengths_generated(all_tensor_metas, tensors):
    """Return the lengths of all tensors generated."""
    all_num_samples = []
    all_tensors_generated_length = {tensor: 0 for tensor in tensors}
    for tensor_meta_dict in all_tensor_metas:
        num_samples_dict = {}
        for tensor, meta in tensor_meta_dict.items():
            all_tensors_generated_length[tensor] += meta.length
            num_samples_dict[tensor] = meta.length
        all_num_samples.append(num_samples_dict)
    return all_num_samples, all_tensors_generated_length


def get_old_chunk_paths(target_ds, generated_tensors, overwrite):
    """Get the old chunk paths."""
    old_chunk_paths = []
    if overwrite:
        for key in generated_tensors:
            tensor = target_ds[key]
            old_chunk_paths.extend(tensor.chunk_engine.list_all_chunks_path())

    return old_chunk_paths


def process_transform_result(result: List[Dict]):
    """Return the transformed result."""
    if all(res.get("all_samples_skipped") for res in result):
        raise AllSamplesSkippedError
    if not result:
        return result
    final = defaultdict(list)
    keys = list(result[0].keys())
    for item in result:
        for key in keys:
            final[key].append(item[key])
    return final


def get_pbar_description(compute_functions: List):
    """Returns the description string for a :meth:`muller.compute` evaluation progress bar.
    Incoming list should be a list of `ComputeFunction`s.
    """

    num_funcs = len(compute_functions)
    if num_funcs == 0:
        return "Evaluating"

    func_names: List[str] = [f.name for f in compute_functions]
    if num_funcs == 1:
        return f"Evaluating {func_names[0]}"

    names_desc = ", ".join(func_names)
    return f"Evaluating [{names_desc}]"


def rechunk_if_necessary(ds):
    """Rechunking."""
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


def transform_summary(data_in, result):
    """Print the transformation results."""
    samples_skipped = sum(result["samples_skipped"])
    successful = len_data_in(data_in) - samples_skipped
    successful_percent = round((successful / len_data_in(data_in)) * 100, 2)
    skipped_percent = round(100 - successful_percent, 2)

    print(
        "No. of samples successfully processed:", successful, f"({successful_percent}%)"
    )
    print("No. of samples skipped:", samples_skipped, f"({skipped_percent}%)")


def reload_and_rechunk(
    overwrite,
    original_data_in,
    target_ds,
    initial_autoflush,
    kwargs,
    completed=True,
):
    """Reload the dataset and conduct rechunking."""
    if overwrite:
        original_data_in.storage.clear_cache_without_flush()
        load_meta(original_data_in)
        if completed and not kwargs.get("disable_rechunk"):
            rechunk_if_necessary(original_data_in)
    else:
        load_meta(target_ds)
        if completed:
            target_ds.storage.autoflush = initial_autoflush
            if not kwargs.get("disable_rechunk"):
                rechunk_if_necessary(target_ds)


def create_worker_chunk_engines(
    tensors: List[str],
    output_storage: StorageProvider,
    version_state,
    split_tensor_meta
) -> Dict[str, ChunkEngine]:
    """Creates chunk engines corresponding to each storage for all tensors.
    These are created separately for each worker for parallel uploads.
    """
    all_chunk_engines: Dict[str, ChunkEngine] = {}
    num_tries = 1000
    storage_cache = LRUCache(
        MemoryProvider(), output_storage, TRANSFORM_CHUNK_CACHE_SIZE
    )
    storage_cache.autoflush = False

    # Sherry: replace this with simply a MemoryProvider once we get rid of cacheable
    memory_cache = LRUCache(
        MemoryProvider(),
        MemoryProvider(),
        64 * MB,
    )
    memory_cache.autoflush = False

    meta_key = get_tensor_meta_key("", version_state["commit_id"])
    for tensor in tensors:
        for i in range(num_tries):
            try:
                # this chunk engine is used to retrieve actual tensor meta and chunk_size
                storage_chunk_engine = ChunkEngine(tensor, storage_cache, version_state, split_tensor_meta)
                existing_meta = storage_chunk_engine.tensor_meta

                new_tensor_meta = TensorMeta(
                    htype=existing_meta.htype,
                    dtype=(
                        np.dtype(existing_meta.typestr)
                        if existing_meta.typestr
                        else existing_meta.dtype
                    ),
                    sample_compression=existing_meta.sample_compression,
                    chunk_compression=existing_meta.chunk_compression,
                    max_chunk_size=storage_chunk_engine.max_chunk_size,
                    tiling_threshold=storage_chunk_engine.tiling_threshold,
                    is_sequence=existing_meta.is_sequence,
                    hidden=existing_meta.hidden,
                    verify=existing_meta.verify,
                )
                new_tensor_meta.max_shape = existing_meta.max_shape.copy()
                new_tensor_meta.min_shape = existing_meta.min_shape.copy()
                new_tensor_meta.name = existing_meta.name
                if split_tensor_meta:
                    meta_key = get_tensor_meta_key(tensor, version_state["commit_id"])
                    memory_cache[meta_key] = new_tensor_meta  # type: ignore
                else:
                    memory_cache[meta_key] = {tensor: new_tensor_meta}
                storage_cache.clear_cache()
                storage_chunk_engine = ChunkEngine(
                        tensor, storage_cache, version_state, split_tensor_meta, memory_cache
                    )
                storage_chunk_engine.all_chunk_engines = all_chunk_engines
                all_chunk_engines[tensor] = storage_chunk_engine
                break
            except (JSONDecodeError, KeyError):
                if i == num_tries - 1:
                    raise

    return all_chunk_engines


def add_cache_to_dataset_slice(
    dataset_slice: muller.Dataset,
    tensors: List[str],
) -> muller.Dataset:
    """Add cache to dataset slices."""
    base_storage = get_base_storage(dataset_slice.storage)
    # 64 to account for potentially big encoder corresponding to each tensor
    # Sherry: adjust this size once we get rid of cacheable
    cache_size = 64 * len(tensors) * MB
    cached_store = LRUCache(MemoryProvider(), base_storage, cache_size)
    commit_id = dataset_slice.pending_commit_id
    # don't pass version state to constructor as otherwise all workers will share it, checkout to commit_id instead
    index = Index.from_json(
        dataset_slice.index.to_json()
    )  # we don't allow checkouts for views
    dataset_slice = muller.core.dataset.Dataset(
        path=dataset_slice.path,
        storage=cached_store,
        read_only=dataset_slice.read_only,
        verbose=False,
        enabled_tensors=dataset_slice.enabled_tensors,
        split_tensor_meta=dataset_slice.split_tensor_meta
    )
    dataset_slice.checkout(commit_id)
    dataset_slice.index = index
    return dataset_slice


def _normalize_pg(pg_callback, num_tensors):
    """Normalize the transform progress."""
    def inner(num_samples):
        return pg_callback(int(num_samples / num_tensors))

    return inner


def _extend_data_slice(
    data_slice, offset, transform_dataset, transform_fn, pg_callback
):
    """Extend the dataset slice."""
    extend_fn, args, kwargs = (
        transform_fn.func,
        transform_fn.args,
        transform_fn.kwargs,
    )
    if pg_callback is not None:
        pg_callback = _normalize_pg(pg_callback, len(transform_dataset.tensors))
    transform_dataset.set_pg_callback(pg_callback)
    extend_fn(data_slice, transform_dataset, *args, **kwargs)
    transform_dataset.flush()


def _batch_extend_data_slice(
    data_slice,
    offset,
    transform_dataset,
    pipeline,
    pg_callback,
):
    """Extends a data slice as a batch. Returns ``True`` if any samples were appended and ``False`` otherwise."""
    extend_fn, args, kwargs = (
        pipeline.func,
        pipeline.args,
        pipeline.kwargs,
    )
    if pg_callback is not None:
        pg_callback = _normalize_pg(pg_callback, len(transform_dataset.tensors))
    transform_dataset.set_pg_callback(pg_callback)
    try:
        extend_fn(*data_slice, transform_dataset, *args, **kwargs)
        append_uuid(transform_dataset)
    except Exception as e:
        raise TransformError(
            offset, suggest=isinstance(e, SampleAppendError), is_batch=True
        ) from e


def _check_pipeline(out, tensors, skip_ok):
    """Check pipeline."""
    data = out.data
    result_keys = set(k for k in data if len(data[k]) > 0)

    if skip_ok:
        if not result_keys.issubset(tensors):
            raise TensorMismatchError(list(tensors), list(result_keys), skip_ok)
    elif set(result_keys) != set(tensors):
        raise TensorMismatchError(list(tensors), list(result_keys), skip_ok)


def write_sample_to_transform_dataset(out, transform_dataset):
    """Write sample to transform dataset."""
    if not is_empty_transform_dataset(out):
        for tensor in out.tensors:
            out_tensor = out[tensor]
            transform_tensor = transform_dataset[tensor]
            if transform_tensor.numpy_only and out_tensor.numpy_only:
                for item in out_tensor.items:
                    transform_tensor.extend(item)
            else:
                out_tensor.non_numpy_only()
                transform_tensor.extend(out_tensor.items)
            out_tensor.items.clear()


def _handle_transform_error(
    data_slice,
    offset,
    transform_dataset,
    pipeline,
    tensors,
    end_input_idx,
    ignore_errors,
):
    """Handle transform errors."""
    start_input_idx = transform_dataset.start_input_idx
    skipped_samples = 0
    for i in range(start_input_idx, end_input_idx + 1):
        sample = (
            data_slice[i : i + 1]
            if pd and isinstance(data_slice, pd.DataFrame)
            else data_slice[i]
        )
        try:
            out = transform_sample(sample, pipeline, tensors)

            write_sample_to_transform_dataset(out, transform_dataset)

            transform_dataset.flush()
        except Exception as e:
            if ignore_errors:
                skipped_samples += 1
                continue
            raise TransformError(
                offset + i, sample, suggest=isinstance(e, SampleAppendError)
            ) from e
    return skipped_samples


def _process_single_sample(
    sample,
    i,
    offset,
    transform_dataset,
    pipeline,
    tensors,
    skip_ok,
    ignore_errors,
):
    """
    Process a single sample through the transform, write, and pipeline validation steps.
    May raise an exception or return a skip count.

    Returns:
        skipped (int): 0 or 1, indicating whether this sample was skipped.
    """
    skipped = 0

    transform_dataset.set_start_input_idx(i)

    try:
        out = transform_sample(sample, pipeline, tensors)
        # On the first successful transform, validate the pipeline.
        _check_pipeline(out, tensors, skip_ok)
        write_sample_to_transform_dataset(out, transform_dataset)

    except Exception as e:
        if ignore_errors:
            skipped = 1
        else:
            raise TransformError(
                offset + i, sample, suggest=isinstance(e, SampleAppendError)
            ) from e

    finally:
        # flush or check_flush
        # Note: The flush condition is determined by the caller, who provides the length information.
        transform_dataset.check_flush()

    return skipped


def _process_and_write(
    sample,
    i,
    offset,
    transform_dataset,
    pipeline,
    tensors,
    skip_ok,
    ignore_errors,
    pipeline_checked
):
    """
    Internal helper: performs transform, pipeline check, and write.
    Returns skipped (0 or 1).
    """
    try:
        out = transform_sample(sample, pipeline, tensors)
        if not pipeline_checked:
            _check_pipeline(out, tensors, skip_ok)
        write_sample_to_transform_dataset(out, transform_dataset)
        return 0
    except Exception as e:
        if ignore_errors:
            return 1
        raise TransformError(
            offset + i, sample, suggest=isinstance(e, SampleAppendError)
        ) from e


def _make_iterable(data_slice):
    if isinstance(data_slice, pd.DataFrame):
        return (data_slice[i : i + 1] for i in range(len(data_slice)))
    return data_slice


def _maybe_flush(transform_dataset, i, total_len, skipped_in_batch):
    """
    Perform flush or check_flush, and reset skipped_in_batch to 0 when transform_dataset.start_input_idx is None.
    Return the updated skipped_in_batch.
    """
    if i == total_len - 1:
        transform_dataset.flush()
    else:
        transform_dataset.check_flush()

    if transform_dataset.start_input_idx is None:
        return 0
    return skipped_in_batch


def _maybe_progress(pg_callback, progress, last_time, i, total_len):
    """
    Handle progress callback and return (new_progress, new_last_time).
    """
    progress += 1
    now = time.time()
    if (now - last_time > TRANSFORM_PROGRESSBAR_UPDATE_INTERVAL) or (i == total_len - 1):
        pg_callback(progress)
        return 0, now
    return progress, last_time


def _transform_and_append_data_slice(
    data_slice,
    offset,
    transform_dataset,
    pipeline,
    tensors,
    skip_ok,
    pg_callback,
    ignore_errors,
) -> Dict[str, Any]:

    skip_details = [0, 0, 0] # skipped_samples, skipped_in_batch, progress
    pipeline_checked = False
    last_time = time.time()

    for i, sample in enumerate(_make_iterable(data_slice)):
        try:
            # Normal path: attempt transform + write.
            skipped = _process_and_write(
                sample, i, offset,
                transform_dataset, pipeline, tensors,
                skip_ok, ignore_errors,
                pipeline_checked
            )
            if not pipeline_checked and skipped == 0:
                pipeline_checked = True

        except TransformError:
            # TransformError
            raise
        except Exception:
            # On ChunkEngine or other errors, enter retry logic.
            skip_details[0] -= skip_details[1]
            skip_details[1] = 0
            extra_skipped = _handle_transform_error(
                data_slice,
                offset,
                transform_dataset,
                pipeline,
                tensors,
                i,
                ignore_errors,
            )
            skip_details[0] += extra_skipped
            # Skip this sample and skip subsequent flush / progress updates.
            continue
        else:
            # Normal processing path.
            skip_details[0] += skipped
            skip_details[1] += skipped

        # flush / check_flush
        skip_details[1] = _maybe_flush(transform_dataset, i, len(data_slice), skip_details[1])

        # progress call back
        if pg_callback is not None:
            skip_details[2], last_time = _maybe_progress(pg_callback, skip_details[2], last_time, i, len(data_slice))

    return {
        "samples_skipped": skip_details[0],
        "all_samples_skipped": skip_details[0] == len(data_slice),
    }


def _retrieve_memory_objects(all_chunk_engines):
    all_tensor_metas = {}
    all_chunk_id_encoders = {}
    all_tile_encoders = {}
    all_sequence_encoders = {}
    all_chunk_maps = {}
    all_commit_diffs = {}
    all_hash_label_maps = {}
    for tensor, chunk_engine in all_chunk_engines.items():
        chunk_engine.cache.flush()
        chunk_engine.meta_cache.flush()
        all_tensor_metas[tensor] = chunk_engine.tensor_meta
        all_chunk_id_encoders[tensor] = chunk_engine.chunk_id_encoder
        if chunk_engine.enable_tile_encoder:
            all_tile_encoders[tensor] = chunk_engine.tile_encoder
        all_sequence_encoders[tensor] = chunk_engine.sequence_encoder
        all_chunk_maps[tensor] = chunk_engine.commit_chunk_map
        all_commit_diffs[tensor] = chunk_engine.commit_diff
        if chunk_engine.is_temp_label_tensor:
            all_hash_label_maps[tensor] = chunk_engine.hash_label_map

    return {
        "tensor_metas": all_tensor_metas,
        "chunk_id_encoders": all_chunk_id_encoders,
        "sequence_encoders": all_sequence_encoders,
        "tile_encoders": all_tile_encoders,
        "commit_chunk_maps": all_chunk_maps,
        "commit_diffs": all_commit_diffs,
        "hash_label_maps": all_hash_label_maps,
    }


def store_data_slice_with_pbar(pg_callback, transform_input: Tuple) -> Dict:
    """Store data slice with progress bar."""
    data_slice, offset, output_storage, inp = transform_input
    pipeline = inp[2]
    version_state = inp[3]
    version_state = _get_version_state(output_storage, version_state)

    data_slice, transform_dataset, rel_tensors, ret, all_chunk_engines\
        = _construct_data_slices(inp[0],  # tensors
                                 output_storage,
                                 version_state,
                                 inp[9],  # split_tensor_meta,
                                 data_slice,
                                 inp[1], # visible_tensors
                                 inp[6], # cache_size
                                 inp[8]) # batch_enable
    err = None

    try:
        if inp[8]: # batch_enable
            _batch_extend_data_slice(
                data_slice,
                offset,
                transform_dataset,
                pipeline.functions[0],
                pg_callback,
            )
        elif inp[5]:  # extend_only
            _extend_data_slice(
                data_slice,
                offset,
                transform_dataset,
                pipeline.functions[0],
                pg_callback,
            )
        else:
            ret = _transform_and_append_data_slice(
                data_slice,
                offset,
                transform_dataset,
                pipeline,
                rel_tensors,
                inp[4],  # skip_ok
                pg_callback,
                inp[7],  # ignore_errors,
            )
    except Exception as e:
        try:
            transform_dataset.flush()
        except Exception as e1:
            raise TransformFailure from e1
        err = e
    finally:
        # retrieve relevant objects from memory
        return _construct_meta(all_chunk_engines, ret, err)


def _get_version_state(output_storage, version_state):
    version_info = load_version_info(output_storage)
    version_state["commit_node_map"] = version_info["commit_node_map"]
    version_state["commit_node"] = version_info["commit_node_map"].get(version_state["commit_id"], None)
    return version_state


def store_data_slice(transform_input: Tuple) -> Dict:
    """Takes a slice of the original data and iterates through it and stores it in the actual storage.
    The tensor_meta & chunk_id_encoder are not stored to the storage to prevent overwrites/race conditions b/w workers.
    They are instead stored in memory and returned.
    """
    return store_data_slice_with_pbar(None, transform_input)


def _construct_data_slices(tensors, output_storage, version_state, split_tensor_meta, data_slice, visible_tensors,
                           cache_size, batch_enable):

    all_chunk_engines = create_worker_chunk_engines(
        tensors, output_storage, version_state, split_tensor_meta
    )

    if isinstance(data_slice, muller.Dataset):
        data_slice = add_cache_to_dataset_slice(data_slice, tensors)

    rel_tensors = list(visible_tensors)
    if DATASET_UUID_NAME in tensors:
        rel_tensors.append(DATASET_UUID_NAME)

    transform_dataset = TransformDataset(
        rel_tensors,
        all_chunk_engines,
        cache_size=cache_size,
        is_batch=batch_enable
    )

    ret = {
        "all_samples_skipped": False,
        "samples_skipped": 0,
    }

    return data_slice, transform_dataset, rel_tensors, ret, all_chunk_engines


def _construct_meta(all_chunk_engines, ret, err):
    meta = _retrieve_memory_objects(all_chunk_engines)
    meta.update(ret)

    err_dict: Optional[Dict[str, Any]] = None
    if err:
        err_dict = {"raise": err}
        cause = err.__cause__
        if cause:
            cause_traceback = "".join(
                traceback.format_exception(cause.__class__, cause, cause.__traceback__)  # type: ignore
            )
            err_dict["traceback"] = cause_traceback
    meta["error"] = err_dict
    return meta
    
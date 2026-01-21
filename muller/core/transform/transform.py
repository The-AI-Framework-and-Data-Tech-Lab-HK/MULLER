# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/core/transform/transform.py
#
# Modifications Copyright (c) 2026 Bingyu Liu

import ast
import inspect
import json
import logging
import textwrap
from itertools import repeat
from typing import Dict, List
from typing import Optional

import numpy as np

import muller
from muller.constants import DEFAULT_TRANSFORM_SAMPLE_CACHE_SIZE, DATASET_UUID_NAME
from muller.core.compute.provider import get_progress_bar, ComputeProvider
from muller.core.meta.encode.chunk_id import ChunkIdEncoder
from muller.core.meta.encode.sequence import SequenceEncoder
from muller.core.meta.encode.tile import TileEncoder
from muller.core.meta.tensor_meta import TensorMeta
from muller.core.storage import MemoryProvider
from muller.core.storage.provider import StorageProvider
from muller.core.version_control import protected_commit
from muller.core.version_control.commit_chunk_map import CommitChunkMap
from muller.core.version_control.commit_diff import CommitDiff
from muller.util.compute import get_compute_provider
from muller.util.exceptions import AllSamplesSkippedError, TransformError, UnAuthorizationError
from muller.util.json import HubJsonDecoder, HubJsonEncoder
from muller.util.keys import (
    get_sequence_encoder_key,
    get_tensor_commit_chunk_map_key,
    get_tensor_commit_diff_key,
    get_tensor_meta_key,
    get_chunk_id_encoder_key,
    get_tensor_tile_encoder_key,
)
from muller.util.remove_cache import get_base_storage
from muller.util.transform import (
    check_lengths,
    check_transform_data_in,
    check_transform_ds_out,
    close_states,
    create_slices,
    delete_overwritten_chunks,
    get_lengths_generated,
    get_old_chunk_paths,
    get_pbar_description,
    prepare_data_in,
    process_transform_result,
    reload_and_rechunk,
    sanitize_workers_scheduler,
    store_data_slice,
    store_data_slice_with_pbar,
    check_checkpoint_interval,
    len_data_in,
    transform_summary,
)
from muller.util.version_control import auto_checkout


class ComputeFunction:
    def __init__(self, func, args, kwargs, name: Optional[str] = None, batch_enable: bool = False):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.name = self.func.__name__ if name is None else name
        self.batch_enable = batch_enable

        if self.batch_enable:
            self._analyze_function()

    def __call__(self, sample_in):
        if self.batch_enable:
            return self.func(*sample_in, *self.args, **self.kwargs)
        return self.func(sample_in, *self.args, **self.kwargs)

    def eval(self, *args, **kwargs):
        """Function to run eval."""
        if self.batch_enable:
            if len(args) < self.num_data_params:
                raise ValueError(
                    f"Expected at least {self.num_data_params} data arguments, got {len(args)}"
                )

            data_lists = args[:self.num_data_params]
            remaining_args = args[self.num_data_params:]

            lengths = [len(data_list) for data_list in data_lists]
            if len(set(lengths)) > 1:
                raise ValueError(
                    f"All data lists must have the same length, got lengths: {lengths}"
                )

            kwargs['batch_enable'] = True

            self._eval_internal(data_lists, *remaining_args, **kwargs)

        else:
            data_in = args[0]
            remaining_args = args[1:]
            self._eval_internal(data_in, *remaining_args, **kwargs)

    def _analyze_function(self):
        try:
            sig = inspect.signature(self.func)
            all_params = list(sig.parameters.keys())

            if len(all_params) < 2:
                raise ValueError(f"Function {self.func} must have at least 2 parameters (data params + sample_out)")

            self.param_names = all_params[:-1]
            self.sample_out_param = all_params[-1]
            self.num_data_params = len(self.param_names)

            source = inspect.getsource(self.func)
            dedented_source = textwrap.dedent(source)
            tree = ast.parse(dedented_source)
            self.append_mappings = self._extract_append_mappings(tree)

            self._validate_consistency()

        except Exception as e:
            print(f"Warning: Failed to analyze function {self.func} for batch processing: {e}")
            self.batch_enable = False
            self.param_names = []
            self.append_mappings = {}

    def _extract_append_mappings(self, tree) -> Dict[str, str]:
        """从AST中提取 sample_out.field.append(param) 的映射关系"""
        mappings = {}

        class AppendVisitor(ast.NodeVisitor):
            def visit_Call(self, node):
                """Function to check if it is sample_out.field.append(param) mode."""
                if (isinstance(node.func, ast.Attribute) and
                        node.func.attr in ['append', 'extend'] and
                        len(node.args) == 1):

                    # Extract field name: sample_out.field.append
                    if (isinstance(node.func.value, ast.Attribute) and
                            isinstance(node.func.value.value, ast.Name)):  # Support differennt congirations

                        field_name = node.func.value.attr

                        # Extract parameter names
                        if isinstance(node.args[0], ast.Name):
                            param_name = node.args[0].id
                            mappings[param_name] = field_name

                self.generic_visit(node)

        visitor = AppendVisitor()
        visitor.visit(tree)
        return mappings

    def _validate_consistency(self):
        """Validate the consistency between parameters and the append operation."""
        func_name = self.func

        # Check the number of configurations
        if len(self.param_names) != len(self.append_mappings):
            logging.error(f"Warning: Function {func_name}: Parameter count ({len(self.param_names)}) "
                          f"doesn't match append operations count ({len(self.append_mappings)})")
            logging.error(f"Parameters: {self.param_names}")
            logging.error(f"Append operations: {list(self.append_mappings.keys())}")
            self.batch_enable = False
            return

        # Check parameter name matching.
        missing_params = set(self.param_names) - set(self.append_mappings.keys())
        extra_appends = set(self.append_mappings.keys()) - set(self.param_names)

        if missing_params or extra_appends:
            logging.error(f"Warning: Function {func_name}: Parameter mismatch")
            if missing_params:
                logging.error(f"Parameters {missing_params} are not used in append operations")
            if extra_appends:
                logging.error(f"Append operations use undefined parameters {extra_appends}")
            self.batch_enable = False
            return

        logging.info(f"✅ Function {func_name} batch analysis successful:")

    def _eval_internal(
            self,
            data_in,
            ds_out: Optional[muller.Dataset] = None,
            num_workers: int = 0,
            scheduler: str = "threaded",
            progressbar: bool = True,
            skip_ok: bool = False,
            ignore_errors: bool = False,
            **kwargs,
    ):
        """Eval the transform process."""
        pipeline = Pipeline([self])
        pipeline.eval(
            data_in,
            ds_out,
            num_workers=num_workers,
            scheduler=scheduler,
            progressbar=progressbar,
            skip_ok=skip_ok,
            ignore_errors=ignore_errors,
            **kwargs,
        )


class Pipeline:
    def __init__(self, functions: List[ComputeFunction]):
        self.functions = functions

    def __len__(self):
        return len(self.functions)

    @staticmethod
    def raise_eval_exception(e, checkpointing_enabled, target_ds, samples_processed):
        """Raise the exception from eval."""
        if checkpointing_enabled:
            logging.info(
                "Transform failed. Resetting back to last committed checkpoint."
            )
            target_ds.reset(force=True)
        index, sample, suggest = None, None, False
        if isinstance(e, UnAuthorizationError):
            raise e
        if isinstance(e, TransformError):
            index, sample, suggest = e.index, e.sample, e.suggest
            if checkpointing_enabled and isinstance(index, int):
                index = samples_processed + index
            e = e.__cause__  # type: ignore
        if isinstance(e, AllSamplesSkippedError):
            raise e
        raise TransformError(
            index=index,
            sample=sample,
            samples_processed=samples_processed,
            suggest=suggest,
        ) from e

    @staticmethod
    def _get_run_results(result, tensors, skip_ok, target_ds, overwrite, storage, kwargs, data_in):
        result = process_transform_result(result)
        all_num_samples, all_tensors_generated_length = get_lengths_generated(
            result["tensor_metas"], tensors
        )

        check_lengths(all_tensors_generated_length, skip_ok)
        generated_tensors = [
            tensor
            for tensor, length in all_tensors_generated_length.items()
            if length > 0
        ]
        old_chunk_paths = get_old_chunk_paths(target_ds, generated_tensors, overwrite)
        _merge_all_meta_info(
            target_ds, storage, generated_tensors, overwrite, all_num_samples, result
        )
        delete_overwritten_chunks(old_chunk_paths, storage, overwrite)

        if kwargs.get("ignore_errors", False):
            transform_summary(data_in, result)

        for res in result["error"]:
            if res is not None:
                logging.info(res["traceback"])
                logging.info(
                    "The above exception was the direct cause of the following exception:\n"
                )
                raise res["raise"]

    def eval(
            self,
            data_in,
            ds_out: Optional[muller.Dataset] = None,
            num_workers: int = 0,
            scheduler: str = "threaded",
            progressbar: bool = True,
            skip_ok: bool = False,
            ignore_errors: bool = False,
            **kwargs,
    ):
        """Eval the transform process."""

        num_workers, scheduler = sanitize_workers_scheduler(num_workers, scheduler)
        overwrite = ds_out is None

        check_transform_data_in(data_in, scheduler, overwrite, kwargs.get('batch_enable', False))

        data_in, original_data_in = prepare_data_in(
            data_in, overwrite
        )
        target_ds = data_in if overwrite else ds_out

        check_transform_ds_out(
            target_ds, scheduler, kwargs.get("tmp_check_lengths", True), kwargs.get("read_only_ok", False) and overwrite
        )

        # if overwrite then we've already flushed and auto-checked out data_in which is target_ds now
        if not overwrite:
            target_ds.flush()
            auto_checkout(target_ds)

        checkpointing_enabled = kwargs.get("checkpoint_interval", 0) > 0
        total_samples = len_data_in(data_in, kwargs.get('batch_enable', False))
        if checkpointing_enabled:
            check_checkpoint_interval(
                data_in,
                kwargs.get("checkpoint_interval", 0),
                num_workers,
                overwrite,
                kwargs.get("verbose", True),
                kwargs.get('batch_enable', False)
            )
            if kwargs.get('batch_enable', False):
                num_samples = len_data_in(data_in, batch_enable=True)

                datas_in = []
                for i in range(0, num_samples, kwargs.get("checkpoint_interval", 0)):
                    # Perform slicing on each sublist.
                    chunk = [
                        sublist[i: i + kwargs.get("checkpoint_interval", 0)]
                        for sublist in data_in
                    ]
                    datas_in.append(chunk)
            else:
                datas_in = [
                    data_in[i: i + kwargs.get("checkpoint_interval", 0)]
                    for i in range(0, len_data_in(data_in), kwargs.get("checkpoint_interval", 0))
                ]

        else:
            datas_in = [data_in]

        compute_provider = get_compute_provider(scheduler, num_workers)

        target_ds.storage.autoflush = False

        samples_processed = 0
        desc = get_pbar_description(self.functions)
        if progressbar:
            pbar = get_progress_bar(len_data_in(data_in, kwargs.get('batch_enable', False)), desc)
            pqueue = compute_provider.create_queue()
        else:
            pbar, pqueue = None, None
        completed = False
        progress = 0.0
        try:
            for temp_data_in in datas_in:
                if kwargs.get("checkpoint_interval", 0) > 0:
                    protected_commit(
                        target_ds,
                        f"Auto-commit during muller.compute of {desc.split()[1]} after {progress}% progress",
                        None,
                        False,
                        is_checkpoint=True,
                        total_samples_processed=samples_processed,
                    )
                    target_ds.flush()
                progress = round(
                    (samples_processed +
                     len_data_in(data_in, kwargs.get('batch_enable', False))) / total_samples * 100, 2
                )
                end = progress == 100

                try:
                    self.run(
                        data_in=temp_data_in,
                        target_ds=target_ds,
                        tmp_compute=compute_provider,
                        num_workers=num_workers,
                        scheduler=scheduler,
                        progressbar=progressbar,
                        overwrite=overwrite,
                        skip_ok=skip_ok,
                        read_only=kwargs.get("read_only_ok", False) and overwrite,
                        cache_size=kwargs.get("cache_size", DEFAULT_TRANSFORM_SAMPLE_CACHE_SIZE),
                        pbar=pbar,
                        pqueue=pqueue,
                        ignore_errors=ignore_errors,
                        **kwargs,
                    )
                    samples_processed += len_data_in(data_in, kwargs.get('batch_enable', False))
                    completed = end
                except Exception as e:
                    self.raise_eval_exception(e, kwargs.get("checkpoint_interval", 0) > 0, target_ds, samples_processed)
                finally:
                    reload_and_rechunk(
                        overwrite,
                        original_data_in,
                        target_ds,
                        target_ds.storage.autoflush,
                        kwargs,
                        completed,
                    )
        finally:
            close_states(compute_provider, pbar, pqueue)

    def run(
            self,
            data_in,
            target_ds: muller.Dataset,
            tmp_compute: ComputeProvider,
            num_workers: int,
            progressbar: bool = True,
            overwrite: bool = False,
            skip_ok: bool = False,
            **kwargs,
    ):
        """Runs the pipeline on the input data to produce output samples and stores in the dataset.
           This receives arguments processed and sanitized by the Pipeline.eval method.
        """
        batch_enable = kwargs.get('batch_enable', False)
        slices, offsets = create_slices(data_in, num_workers, batch_enable)

        storage = get_base_storage(target_ds.storage)

        tensors = list(target_ds.get_tensors(include_disabled=False))
        tensors = [target_ds[t].key for t in tensors]

        # commit_node_map, commit_node and full_tensors are not used in transform process
        version_state = dict()
        for key in target_ds.version_state.keys():
            if (key not in {"commit_node_map", "commit_node"} and
                    (key != "full_tensors" or DATASET_UUID_NAME not in tensors)):
                version_state[key] = target_ds.version_state[key]
        if isinstance(storage, MemoryProvider):
            storages = [storage] * len(slices)
        else:
            storages = [storage.copy() for _ in slices]
        args = (
            tensors,
            [target_ds[t].key for t in list(target_ds.tensors)],
            self,
            version_state,
            skip_ok,
            kwargs.get("extend_only"),
            kwargs.get("cache_size", 16),
            kwargs.get("ignore_errors", False),
            batch_enable,
            target_ds.split_tensor_meta
        )
        map_inp = zip(slices, offsets, storages, repeat(args))
        try:
            if progressbar:
                desc = get_pbar_description(self.functions)
                result = tmp_compute.map_with_progress_bar(
                    store_data_slice_with_pbar,
                    map_inp,
                    total_length=len_data_in(data_in),
                    desc=desc,
                    pbar=kwargs.get("pbar", None),
                    pqueue=kwargs.get("pqueue", None),
                )
            else:
                result = tmp_compute.map(store_data_slice, map_inp)
        except Exception as e:
            raise e

        if kwargs.get("read_only", False):
            return

        self._get_run_results(result, tensors, skip_ok, target_ds, overwrite, storage, kwargs, data_in)


def compute(fn=None, *, name: Optional[str] = None, batch_enable: bool = False):
    """Function to decorate."""
    def decorator(func):
        def inner(*args, **kwargs):
            return ComputeFunction(func, args, kwargs, name, batch_enable)

        return inner

    if fn is None:
        return decorator
    return decorator(fn)


def _merge_all_meta_info(
    target_ds, storage, generated_tensors, overwrite, all_num_samples, result
):
    """Merge all meta information."""
    _merge_all_commit_diffs(result["commit_diffs"], target_ds, storage, overwrite, generated_tensors)
    _merge_all_tile_encoders(result["tile_encoders"], all_num_samples, target_ds, storage, overwrite, generated_tensors)
    _merge_all_tensor_metas(result["tensor_metas"], target_ds, storage, overwrite, generated_tensors)
    _merge_all_chunk_id_encoders(result["chunk_id_encoders"], target_ds, storage, overwrite, generated_tensors)
    _merge_all_sequence_encoders(result["sequence_encoders"], target_ds, storage, overwrite, generated_tensors)

    if target_ds.commit_id is not None:
        _merge_all_commit_chunk_maps(result["commit_chunk_maps"], target_ds, storage, overwrite, generated_tensors)


def _merge_all_tensor_metas(
    all_workers_tensor_metas: List[Dict[str, TensorMeta]],
    target_ds: muller.Dataset,
    storage: StorageProvider,
    overwrite: bool,
    tensors: List[str],
) -> None:
    """Merges the tensor metas from all workers into one, then stores it in target_ds."""
    commit_id = target_ds.version_state.get("commit_id")
    if not target_ds.split_tensor_meta:
        meta_key = get_tensor_meta_key("", commit_id) # tensor_meta.json
        meta_dict = {}
    for tensor in tensors:
        tensor_meta = None if overwrite else target_ds[tensor].meta
        for current_worker_metas in all_workers_tensor_metas:
            current_meta = current_worker_metas[tensor]
            if tensor_meta is None:
                tensor_meta = current_meta
            else:
                _combine_metas(tensor_meta, current_meta)
        if target_ds.split_tensor_meta:
            meta_key = get_tensor_meta_key(tensor, commit_id)
            storage[meta_key] = tensor_meta.tobytes()  # type: ignore
        else:
            d = {str(k): v for k, v in tensor_meta.__getstate__().items()} # dict(TensorMeta) to dict(str)
            meta_dict[tensor] = d

    if not target_ds.split_tensor_meta:
        loaded_dict = json.loads(storage[meta_key], cls=HubJsonDecoder) # bytes to dict(str)
        loaded_dict.update(meta_dict)
        temp_bytes = bytes(json.dumps(loaded_dict, sort_keys=True, indent=4, cls=HubJsonEncoder), "utf-8")
        storage[meta_key] = temp_bytes  # type: ignore


def _combine_metas(ds_tensor_meta: TensorMeta, worker_tensor_meta: TensorMeta) -> None:
    """Combines the dataset's tensor meta with a single worker's tensor meta."""
    # if tensor meta is empty, copy attributes from current_meta
    ds_tensor_meta.update_length(worker_tensor_meta.length)
    if len(ds_tensor_meta.max_shape) == 0 or ds_tensor_meta.dtype is None:
        ds_tensor_meta.set_dtype_str(worker_tensor_meta.dtype)
        if not ds_tensor_meta.htype and worker_tensor_meta.htype:
            ds_tensor_meta.set_htype(worker_tensor_meta.htype)
        ds_tensor_meta.update_shape_interval(worker_tensor_meta.max_shape)
        ds_tensor_meta.update_shape_interval(worker_tensor_meta.min_shape)
    # len of min_shape will be 0 if 0 outputs from worker
    elif len(worker_tensor_meta.min_shape) != 0:
        assert (
            ds_tensor_meta.dtype == worker_tensor_meta.dtype
            or worker_tensor_meta.dtype is None
        )
        assert (
            ds_tensor_meta.htype == worker_tensor_meta.htype
            or worker_tensor_meta.htype is None
        )
        # Sherry: we can support this once we have ragged tensor support
        assert len(ds_tensor_meta.max_shape) == len(worker_tensor_meta.max_shape)
        assert len(ds_tensor_meta.min_shape) == len(worker_tensor_meta.min_shape)
        ds_tensor_meta.update_shape_interval(worker_tensor_meta.max_shape)
        ds_tensor_meta.update_shape_interval(worker_tensor_meta.min_shape)


def _merge_all_chunk_id_encoders(
    all_workers_chunk_id_encoders: List[Dict[str, ChunkIdEncoder]],
    target_ds: muller.Dataset,
    storage: StorageProvider,
    overwrite: bool,
    tensors: List[str],
) -> None:
    """Merges chunk_id_encoders from all workers into one, then stores it in target_ds."""
    commit_id = target_ds.version_state.get("commit_id")
    for tensor in tensors:
        chunk_id_encoder = (
            None if overwrite else target_ds[tensor].chunk_engine.chunk_id_encoder
        )
        for current_worker_chunk_id_encoders in all_workers_chunk_id_encoders:
            current_chunk_id_encoder = current_worker_chunk_id_encoders[tensor]
            if chunk_id_encoder is None:
                chunk_id_encoder = current_worker_chunk_id_encoders[tensor]
            else:
                _combine_chunk_id_encoders(chunk_id_encoder, current_chunk_id_encoder)

        chunk_id_key = get_chunk_id_encoder_key(tensor, commit_id)
        storage[chunk_id_key] = bytes(chunk_id_encoder.tobytes())  # type: ignore


def _combine_chunk_id_encoders(
    ds_chunk_id_encoder: ChunkIdEncoder,
    worker_chunk_id_encoder: ChunkIdEncoder,
) -> None:
    """Combines the dataset's chunk_id_encoder with a single worker's chunk_id_encoder."""
    encoded_ids = worker_chunk_id_encoder.encoded
    if not encoded_ids.flags.writeable:
        encoded_ids = encoded_ids.copy()
    if encoded_ids.size:
        tmp_offset = ds_chunk_id_encoder.num_samples
        for encoded_id in encoded_ids:
            encoded_id[1] += tmp_offset
            if ds_chunk_id_encoder.encoded.size == 0:
                ds_chunk_id_encoder.encoded = np.reshape(encoded_id, (-1, 2))
            else:
                ds_chunk_id_encoder.encoded = np.vstack(
                    [ds_chunk_id_encoder.encoded, encoded_id]
                )


def _merge_all_tile_encoders(
    all_workers_tile_encoders: List[Dict[str, TileEncoder]],
    all_num_samples: List[Dict[str, int]],
    target_ds: muller.Dataset,
    storage: StorageProvider,
    overwrite: bool,
    tensors: List[str],
) -> None:
    """Merge all tile encoder."""
    commit_id = target_ds.version_state["commit_id"]
    for tensor in tensors:
        chunk_engine = target_ds[tensor].chunk_engine
        offset = 0 if overwrite else chunk_engine.num_samples
        if chunk_engine.enable_tile_encoder:
            if overwrite:
                tile_encoder = None
            else:
                tile_encoder = chunk_engine.tile_encoder
        else:
            if all(not worker_tile for worker_tile in all_workers_tile_encoders):
                continue
            raise Exception("Not suppose to have tile encoders.")
        for i, current_worker_tile_encoder in enumerate(all_workers_tile_encoders):
            current_tile_encoder = current_worker_tile_encoder[tensor]
            if tile_encoder is None:
                tile_encoder = current_tile_encoder
            else:
                _combine_tile_encoders(tile_encoder, current_tile_encoder, offset)
            offset += all_num_samples[i][tensor]
        if muller.constants.WRITE_TILES_INDEX:
            tile_key = get_tensor_tile_encoder_key(tensor, commit_id)
            storage[tile_key] = tile_encoder.tobytes()
    target_ds.flush()


def _combine_tile_encoders(
    ds_tile_encoder: TileEncoder, worker_tile_encoder: TileEncoder, offset: int
) -> None:
    """Combines the dataset's tile_encoder with a single tile_encoder."""

    if len(worker_tile_encoder.entries):
        for sample_index in worker_tile_encoder.entries.keys():
            new_sample_index = int(sample_index) + offset

            if new_sample_index in ds_tile_encoder.entries:
                raise ValueError(
                    f"Sample index {new_sample_index} already exists inside `ds_tile_encoder`. "
                    f"Keys={ds_tile_encoder.entries}"
                )

            ds_tile_encoder.entries[new_sample_index] = worker_tile_encoder.entries[
                sample_index
            ]


def _merge_all_commit_chunk_maps(
    all_workers_commit_chunk_maps: List[Dict[str, CommitChunkMap]],
    target_ds: muller.Dataset,
    storage: StorageProvider,
    overwrite: bool,
    tensors: List[str],
) -> None:
    """Merges commit_chunk_maps from all workers into a single one and stores it in target_ds."""
    commit_id = target_ds.version_state["commit_id"]
    for tensor in tensors:
        commit_chunk_map = (
            None if overwrite else target_ds[tensor].chunk_engine.commit_chunk_map
        )
        for current_worker_commit_chunk_map in all_workers_commit_chunk_maps:
            current_commit_chunk_map = current_worker_commit_chunk_map[tensor]
            if commit_chunk_map is None:
                commit_chunk_map = current_commit_chunk_map
            else:
                _combine_commit_chunk_maps(commit_chunk_map, current_commit_chunk_map)

        commit_chunk_key = get_tensor_commit_chunk_map_key(tensor, commit_id)
        storage[commit_chunk_key] = commit_chunk_map.tobytes()  # type: ignore


def _combine_commit_chunk_maps(
    ds_commit_chunk_map: CommitChunkMap,
    worker_commit_chunk_map: CommitChunkMap,
) -> None:
    """Combines the dataset's commit_chunk_map with a single worker's commit_chunk_map."""
    ds_commit_chunk_map.chunks.update(worker_commit_chunk_map.chunks)


def _merge_all_commit_diffs(
    all_workers_commit_diffs: List[Dict[str, CommitDiff]],
    target_ds: muller.Dataset,
    storage: StorageProvider,
    overwrite: bool,
    tensors: List[str],
) -> None:
    """Merges commit_diffs from all workers into one, then stores it in target_ds."""
    commit_id = target_ds.version_state.get("commit_id")
    for tensor in tensors:
        commit_diff = None if overwrite else target_ds[tensor].chunk_engine.commit_diff  # type: ignore
        for current_commit_diffs in all_workers_commit_diffs:
            tmp_diff = current_commit_diffs[tensor]
            if commit_diff is None:
                commit_diff = tmp_diff
                commit_diff.transform_data()
            else:
                _combine_commit_diffs(commit_diff, tmp_diff)

        commit_chunk_key = get_tensor_commit_diff_key(tensor, commit_id)
        storage[commit_chunk_key] = commit_diff.tobytes()  # type: ignore


def _combine_commit_diffs(
    ds_commit_diff: CommitDiff, worker_commit_diff: CommitDiff
) -> None:
    """Combines the dataset's commit_diff with a single worker's commit_diff."""
    ds_commit_diff.add_data(worker_commit_diff.num_samples_added)


def _merge_all_sequence_encoders(
    all_workers_sequence_encoders: List[Dict[str, SequenceEncoder]],
    target_ds: muller.Dataset,
    storage: StorageProvider,
    overwrite: bool,
    tensors: List[str],
) -> None:
    """Merge all sequence encoders."""
    commit_id = target_ds.version_state["commit_id"]
    for tensor in tensors:
        actual_tensor = target_ds[tensor]
        if not actual_tensor.is_sequence:
            continue

        if overwrite:
            sequence_encoder = None
        else:
            sequence_encoder = actual_tensor.chunk_engine.sequence_encoder

        for current_worker_sequence_encoder in all_workers_sequence_encoders:
            current_sequence_encoder = current_worker_sequence_encoder[tensor]
            if sequence_encoder is None:
                sequence_encoder = current_sequence_encoder
            else:
                _combine_sequence_encoders(sequence_encoder, current_sequence_encoder)

        sequence_key = get_sequence_encoder_key(tensor, commit_id)
        storage[sequence_key] = sequence_encoder.tobytes()  # type: ignore


def _combine_sequence_encoders(
    ds_sequence_encoder: SequenceEncoder, worker_sequence_encoder: SequenceEncoder
) -> None:
    """Combines the dataset's sequence_encoder with a single worker's sequence_encoder."""
    arr = worker_sequence_encoder.array
    last_index = -1
    for _, temp_arr in enumerate(arr):
        next_last_index = temp_arr[2]
        ds_sequence_encoder.register_samples(temp_arr[0], next_last_index - last_index)
        last_index = next_last_index

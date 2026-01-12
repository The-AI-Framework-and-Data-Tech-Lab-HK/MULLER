# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/core/transform/transform_dataset.py
#
# Modifications Copyright (c) 2026 Bingyu Liu

import json

import numpy as np

from muller.constants import MB, DATASET_UUID_NAME
from muller.core.partial_sample import PartialSample
from muller.core.sample import Sample
from muller.core.tensor import Tensor
from muller.core.transform.transform_tensor import TransformTensor
from muller.util.casting import intelligent_cast
from muller.util.exceptions import SampleAppendError, SampleAppendingError, SampleExtendingError, \
    TensorDtypeMismatchError
from muller.util.json import validate_json_object, HubJsonEncoder


class TransformDataset:
    def __init__(
        self,
        tensors,
        all_chunk_engines=None,
        idx=slice(None, None, None),
        cache_size=16,
        is_batch=False
    ):
        self.tensors = tensors
        self.data = {}
        self.all_chunk_engines = all_chunk_engines
        self.cache_size = cache_size * MB
        self.cache_used = 0
        self.idx = idx
        self.is_batch = is_batch
        self.pg_callback = None
        self.start_input_idx = None
        self._init_tensors()

    def __len__(self):
        return max(len(self[tensor]) for tensor in self.data)

    def __getattr__(self, tensor):
        try:
            return self.data[tensor][self.idx]  # TransformTensor object
        except KeyError:
            self.data[tensor] = TransformTensor(self, tensor, is_batch=self.is_batch)
            return self.data[tensor][self.idx]

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.__getattr__(item)
        assert isinstance(item, (slice, int))
        self.idx = item
        return self

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @staticmethod
    def _calculate_sample_size(item, dtype, htype):
        if isinstance(item, str):
            return len(item.encode())
        if htype in ("json", "list"):
            # NOTE: These samples will be serialized twice. Once here, and once in the chunk engine.
            validate_json_object(item, dtype)
            byts = json.dumps(item, cls=HubJsonEncoder).encode()
            return len(byts)

        try:
            item = intelligent_cast(item, dtype, htype)
        except TensorDtypeMismatchError:
            # class_label tensor can have integer dtype, but sample can be a string
            if htype == "class_label":
                item = intelligent_cast(item, "str", htype)
            else:
                raise
        return item.nbytes

    def set_start_input_idx(self, start_input_idx):
        """Set the start input index"""
        if self.start_input_idx is None:
            self.start_input_idx = start_input_idx

    def append(self, sample, skip_ok=False, append_empty=False):
        """Append a sample to the dataset"""
        if not isinstance(sample, dict):
            raise SampleAppendingError()

        if skip_ok:
            raise ValueError(
                "`skip_ok` is not supported for `ds.append` in transforms. "
                "Use `skip_ok` parameter of the `eval` method instead."
            )

        if len(set(map(len, (self[k] for k in sample)))) != 1:
            raise ValueError(
                "All tensors are expected to have the same length before `ds.append`."
            )

        for k in self.tensors:
            if k in sample:
                self[k].append(sample[k])
            elif append_empty:
                self[k].append(None)

    def extend(self, sample, skip_ok=False, append_empty=False):
        """Append samples to the dataset"""
        if not isinstance(sample, dict):
            raise SampleExtendingError()

        if skip_ok:
            raise ValueError(
                "`skip_ok` is not supported for `ds.extend` in transforms. "
                "Use `skip_ok` parameter of the `eval` method instead."
            )

        if len(set(map(len, (self[k] for k in sample)))) != 1:
            raise ValueError(
                "All tensors are expected to have the same length before `ds.extend`."
            )

        n = len(next(iter(sample.values())))
        for v in sample.values():
            if len(v) != n:
                sizes = {k: len(v) for (k, v) in sample.items()}
                raise ValueError(
                    f"Incoming samples are not of equal lengths. Incoming sample sizes: {sizes}"
                )

        for i in range(n):
            self.append(
                {k: v[i] for (k, v) in sample.items()}, append_empty=append_empty
            )

    def update(self, sample):
        """Update a sample in the dataset"""
        raise NotImplementedError("ds.update is not supported in transforms.")

    def item_added(self, item, tensor):
        """Add a sample to the dataset"""
        if isinstance(item, Sample):
            sizeof_item = len(item.buffer)
        elif isinstance(item, np.ndarray):
            sizeof_item = item.nbytes
        elif isinstance(item, (Tensor, type(None), PartialSample)):
            sizeof_item = 0
        else:
            try:
                chunk_engine = self.all_chunk_engines[tensor]
                meta = chunk_engine.tensor_meta
                htype, dtype = meta.htype, meta.dtype
                # First sample in tensor
                # Flush to set meta attributes
                if dtype is None:
                    self.flush(clear_on_fail=False)
                    return
                sizeof_item = self._calculate_sample_size(item, dtype, htype)
            except Exception:
                sizeof_item = 0

        self.cache_used += sizeof_item

    def set_pg_callback(self, callback):
        """Set pg callback"""
        self.pg_callback = callback

    def check_flush(self):
        """Flush if necessary"""
        if self.cache_used >= self.cache_size:
            self.flush()

    def flush(self, clear_on_fail=True):
        """Flush to the next-level storage"""
        all_chunk_engines = self.all_chunk_engines
        updated_tensors = {}
        no_dtype_tensors = []
        try:
            for name, tensor in self.data.items():
                updated_tensors[name] = 0
                chunk_engine = all_chunk_engines[name]
                callback = None
                meta = chunk_engine.tensor_meta
                if meta.length == 0 and meta.dtype is None:
                    # for rolling back dtype change
                    no_dtype_tensors.append(name)

                if tensor.numpy_only:
                    self._flush_numpy_tensor_to_chunk_engine(
                        name, tensor, chunk_engine, callback, updated_tensors
                    )
                else:
                    self._flush_tensor_to_chunk_engine(
                        name, tensor, chunk_engine, callback, updated_tensors
                    )
            self.start_input_idx = None
            self._clear()
        except Exception as e:
            self._rollback(updated_tensors, no_dtype_tensors)
            if clear_on_fail:
                self._clear()
            e = e.__cause__ if isinstance(e, SampleAppendError) else e  # type: ignore
            raise SampleAppendError(name) from e

    def _init_tensors(self):
        for tensor in self.tensors:
            self.data[tensor] = TransformTensor(self, tensor, is_batch=self.is_batch)

    def _flush_numpy_tensor_to_chunk_engine(
        self, full_name, tensor, chunk_engine, callback, updated_tensors
    ):
        items = tensor[:].numpy_compressed()
        for item in items:
            chunk_engine.extend(
                item,
                pg_callback=self.pg_callback,
            )
            updated_tensors[full_name] += len(item)
        tensor.items.clear()

    def _flush_tensor_to_chunk_engine(
        self, full_name, tensor, chunk_engine, callback, updated_tensors
    ):
        items = tensor[:].numpy_compressed()
        chunk_engine.extend(
            items,
            pg_callback=self.pg_callback,
            is_uuid=(full_name == DATASET_UUID_NAME)
        )
        updated_tensors[full_name] = len(items)
        tensor.items.clear()

    def _rollback(self, updated_tensors, no_dtype_tensors):
        for t in updated_tensors:
            chunk_engine = self.all_chunk_engines[t]
            num_samples = updated_tensors[t]
            chunk_engine.pop(
                list(
                    range(
                        chunk_engine.tensor_length - num_samples,
                        chunk_engine.tensor_length,
                    )
                ),
            )

            if t in no_dtype_tensors:
                meta = chunk_engine.tensor_meta
                meta.dtype = None
                meta.typestr = None
                meta.is_dirty = True

    def _clear(self):
        for tensor in self.data.values():
            tensor.items.clear()
        self.cache_used = 0

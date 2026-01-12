# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/core/tensor.py
#
# Modifications Copyright (c) 2026 Xueling Lin

import inspect
import uuid
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import reduce, partial
from multiprocessing import shared_memory
from typing import Dict, List, Sequence, Union, Optional, Tuple, Any, Callable

import numpy as np

import muller
from muller.api.info import Info
from muller.compression import (
    get_compression_type,
    BYTE_COMPRESSION,
)
from muller.constants import (
    FIRST_COMMIT_ID,
    CREATE_TENSOR_HIDDEN_UUID, DATASET_UUID_NAME,
    MAX_WORKERS_FOR_CHUNK_ENGINE, DEFAULT_MAX_NUMPY_BATCH_SIZE
)
from muller.core.chunk.base_chunk import InputSample
from muller.core.chunk.chunk_engine import ChunkEngine
from muller.core.index import Index, IndexEntry
from muller.core.meta.tensor_meta import TensorMeta, _validate_htype_exists
from muller.core.storage import StorageProvider
from muller.core.storage.lru_cache import LRUCache

from muller.core.version_control.commit_chunk_map import CommitChunkMap
from muller.core.version_control.commit_diff import CommitDiff
from muller.htype import (
    HTYPE_CONVERSION_LHS,
    HTYPE_CONSTRAINTS,
    HTYPE_SUPPORTED_COMPRESSIONS,
)
from muller.util.class_label import convert_to_text
from muller.util.exceptions import (
    TensorDoesNotExistError,
    InvalidKeyTypeError,
    TensorAlreadyExistsError,
    UnsupportedCompressionError, MultiProcessUnsupportedError, UnsupportedMethod, MetaNotFound
)
from muller.util.htype import parse_complex_htype
from muller.util.iteration_warning import check_if_iteration
from muller.util.keys import (
    get_chunk_id_encoder_key,
    get_chunk_key,
    get_tensor_commit_chunk_map_key,
    get_tensor_commit_diff_key,
    get_tensor_meta_key,
    get_tensor_tile_encoder_key,
    get_sequence_encoder_key,
    tensor_exists,
    get_sample_id_tensor_key,
    get_sample_info_tensor_key,
    get_sample_shape_tensor_key,
)
from muller.util.permission.invalid_view_op import invalid_view_op
from muller.util.permission.user_permission_check import user_permission_check
from muller.util.shape_interval import ShapeInterval
from muller.util.version_control import auto_checkout


def _numpy_parallel(batch_tensor, fetch_chunks=False, idxs=None, dimension=-1, name=None,
                    dtype_bytes=None):
    """
    This function is designed to process numpy arrays in parallel, extracting data from the batch_tensor and
    storing it in shared memory.

    Parameters:
        batch_tensor: The tensor from which data is to be extracted.
        aslist: Whether to return the result as a list, defaults to False.
        fetch_chunks: Whether to fetch chunks, defaults to False.
        idxs: A tuple containing the start and end indices for the data to be extracted.
        dimension: Specifies the 2-d numpy dimension.
        name: The name of the shared memory to use, if specified, existing shared memory will be used.
        dtype_bytes: The number of bytes for the data type, used to calculate the start and end positions in the buffer.

    Returns:
        None
        """
    ret = batch_tensor.chunk_engine.numpy(
        batch_tensor.index,
        aslist=False,
        fetch_chunks=fetch_chunks or batch_tensor.is_iteration,
    )

    existing_shm = shared_memory.SharedMemory(name=name)
    start = idxs[0]
    end = idxs[1]
    existing_shm.buf[start * dimension * dtype_bytes: end * dimension * dtype_bytes] = ret.tobytes()
    existing_shm.close()


def create_tensor(
        key: str,
        storage: StorageProvider,
        htype: Union[str, None],
        sample_compression: Union[str, None],
        chunk_compression: Union[str, None],
        version_state: Dict[str, Any],
        overwrite: bool = False,
        **kwargs,
):
    """Create tensor. """
    split_tensor_meta = kwargs.pop("split_tensor_meta", False)
    commit_id = version_state["commit_id"]
    if not overwrite and tensor_exists(key, storage, commit_id, split_tensor_meta):
        raise TensorAlreadyExistsError(key)

    meta_key = get_tensor_meta_key(key, commit_id) if split_tensor_meta else get_tensor_meta_key("", commit_id)
    meta = TensorMeta(
        htype=htype,
        sample_compression=sample_compression,
        chunk_compression=chunk_compression,
        **kwargs,
    )
    if split_tensor_meta:
        storage[meta_key] = meta  # type: ignore
    else:
        try:
            temp = storage[meta_key]
        except KeyError:
            storage[meta_key] = {key: meta}  # write first tensor meta
        else:
            temp.update({key: meta})
            storage[meta_key] = temp  # type: ignore

    if commit_id != FIRST_COMMIT_ID:
        storage[get_tensor_commit_chunk_map_key(key, commit_id)] = CommitChunkMap()  # type: ignore

    diff = CommitDiff(created=True, commit_id=commit_id, storage=storage)
    storage[get_tensor_commit_diff_key(key, commit_id)] = diff  # type: ignore


def delete_tensor(key: str, dataset):
    """Delete tensor. """
    tensor = Tensor(key, dataset)
    chunk_engine: ChunkEngine = tensor.chunk_engine
    enc = chunk_engine.chunk_id_encoder

    chunk_names = [enc.get_name_for_chunk(i) for i in range(chunk_engine.num_chunks)]
    chunk_keys = [
        get_chunk_key(key, chunk_name)
        for chunk_name in chunk_names
    ]
    for chunk_key in chunk_keys:
        try:
            del dataset.storage[chunk_key]
        except KeyError:
            pass

    commit_id = dataset.version_state["commit_id"]
    if dataset.split_tensor_meta:
        meta_key = get_tensor_meta_key(key, commit_id)
        try:
            del dataset.storage[meta_key]
        except KeyError:
            pass
    else:
        meta_key = get_tensor_meta_key("", commit_id)
        try:
            tensor_meta = dataset.storage.get_muller_object(meta_key, dict)
            del tensor_meta[key]
            dataset.storage[meta_key] = tensor_meta
        except KeyError:
            pass

    diff_key = get_tensor_commit_diff_key(key, commit_id)
    try:
        del dataset.storage[diff_key]
    except KeyError:
        pass

    chunk_id_encoder_key = get_chunk_id_encoder_key(key, commit_id)
    try:
        del dataset.storage[chunk_id_encoder_key]
    except KeyError:
        pass

    tile_encoder_key = get_tensor_tile_encoder_key(key, commit_id)
    try:
        del dataset.storage[tile_encoder_key]
    except KeyError:
        pass

    seq_encoder_key = get_sequence_encoder_key(key, commit_id)
    try:
        del dataset.storage[seq_encoder_key]
    except KeyError:
        pass


def _inplace_op(f):
    op = f.__name__

    def inner(tensor, other):
        tensor.write_initialization()
        tensor.chunk_engine.update(
            tensor.index,
            other,
            op,
        )
        if not tensor.index.is_trivial():
            tensor.skip_next_setitem = True
        return tensor

    return inner


class Tensor:
    def __init__(
            self,
            key: str,
            dataset,
            index: Optional[Index] = None,
            is_iteration: bool = False,
            chunk_engine: Optional[ChunkEngine] = None,
            check_flag: bool = False
    ):
        """Initializes a new tensor.

        Args:
            key (str): The internal identifier for this tensor.
            dataset (Dataset): The dataset that this tensor is located in.
            index: The Index object restricting the view of this tensor.
                Can be an int, slice, or (used internally) an Index object.
            is_iteration (bool): If this tensor is being used as an iterator.
            chunk_engine (ChunkEngine, optional): The underlying chunk_engine for the tensor.

        Raises:
            TensorDoesNotExistError: If no tensor with ``key`` exists and a ``tensor_meta`` was not provided.
        """
        self.check_flag = check_flag
        self.key = key
        self.dataset = dataset
        self.storage: LRUCache = dataset.storage
        self.index = index or Index()
        self.version_state = dataset.version_state
        commit_id = self.version_state["commit_id"]
        self.is_iteration = is_iteration

        self._chunk_engine = chunk_engine
        self.split_tensor_meta = dataset.split_tensor_meta

        if not self.check_flag:  # Sherry: To improve the remote loading of tensor_meta.json files
            if not self.is_iteration and not tensor_exists(self.key, self.storage, commit_id, self.split_tensor_meta):
                raise TensorDoesNotExistError(self.key)

        if self.split_tensor_meta:
            meta_key = get_tensor_meta_key(self.key, commit_id)
            try:
                _ = self.storage.get_muller_object(meta_key, TensorMeta)
            except ValueError as e:
                raise MetaNotFound(meta_key, TensorMeta) from e
        else:
            meta_key = get_tensor_meta_key("", commit_id)
            _ = self.storage.get_muller_object(meta_key, dict)[self.key]
        if chunk_engine is not None:
            self.chunk_engine = chunk_engine
        else:
            self.chunk_engine = ChunkEngine(self.key, self.storage, self.version_state, self.split_tensor_meta)
        if not self.is_iteration:
            self.index.validate(self.num_samples)

        # An optimization to skip multiple .numpy() calls when performing inplace ops on slices:
        self._skip_next_setitem = False
        self._indexing_history: List[int] = []

    def __len__(self):
        # Sherry: check whether the synchronized is necessary
        self.chunk_engine.validate_num_samples_is_synchronized()
        return self.index.length(self.num_samples)

    def __getitem__(
            self,
            item: Union[int, slice, List[int], Tuple[Union[int, slice, Tuple[int]]], Index],
            is_iteration: bool = False,
    ):
        if not isinstance(item, (int, slice, list, tuple, type(Ellipsis), Index)):
            raise InvalidKeyTypeError(item)
        if not self.meta.hidden and not is_iteration and isinstance(item, int):
            is_iteration = check_if_iteration(self._indexing_history, item)
            if is_iteration and muller.constants.SHOW_ITERATION_WARNING:
                warnings.warn(
                    "Indexing by integer in a for loop, like `for i in range(len(ds)): ... ds.tensor[i]` "
                    "can be quite slow. Use `for i, sample in enumerate(ds)` instead."
                )
        return Tensor(
            self.key,
            self.dataset,
            index=self.index[item],
            is_iteration=is_iteration,
            chunk_engine=self.chunk_engine,
        )

    def __setitem__(self, item: Union[int, slice], value: Any):
        """Update samples with new values.

        Example:

            >>> tensor.append(np.zeros((10, 10)))
            >>> tensor.shape
            (1, 10, 10)
            >>> tensor[0] = np.zeros((3, 3))
            >>> tensor.shape
            (1, 3, 3)
        """
        self._update(item, value)
        self.dataset.append_only = False

    def __iter__(self):
        for i in range(len(self)):
            yield self.__getitem__(
                i, is_iteration=not isinstance(self.index.values[0], list)
            )

    def __str__(self):
        index_str = f", index={self.index}"
        if self.index.is_trivial():
            index_str = ""
        return f"Tensor(key={repr(self.meta.name or self.key)}{index_str})"

    __repr__ = __str__

    @property
    def chunk_engine(self):
        """Returns the chunk engine. """
        if self._chunk_engine:
            return self._chunk_engine
        if self.split_tensor_meta:
            meta_key = get_tensor_meta_key(self.key, self.version_state["commit_id"])
            _ = self.storage.get_muller_object(meta_key, TensorMeta)
        else:
            meta_key = get_tensor_meta_key("", self.version_state["commit_id"])
            _ = self.storage.get_muller_object(meta_key, TensorMeta)[self.key]
        self._chunk_engine = ChunkEngine(self.key, self.storage, self.version_state, self.split_tensor_meta)
        return self._chunk_engine

    @chunk_engine.setter
    def chunk_engine(self, value):
        """Set the chunk engine. """
        assert isinstance(value, ChunkEngine)
        self._chunk_engine = value

    @property
    def info(self) -> Info:
        """Returns the info. """
        commit_id = self.version_state["commit_id"]
        chunk_engine = self.chunk_engine
        if chunk_engine.info is None or chunk_engine.info_commit_id != commit_id:
            chunk_engine.info = Info(self.dataset, self.key)
            chunk_engine.info_commit_id = commit_id
        return chunk_engine.info

    @property
    def meta(self):
        """Metadata of the tensor."""
        return self.chunk_engine.tensor_meta

    @property
    def shape(self) -> Tuple[Optional[int], ...]:
        """Returns the shape. """
        sample_shape_tensor = self.get_sample_shape_tensor()

        sample_shape_provider = (
            self._sample_shape_provider(sample_shape_tensor)
            if sample_shape_tensor
            else None
        )
        shape: Tuple[Optional[int], ...]
        shape = self.chunk_engine.shape(
            self.index,
            sample_shape_provider=sample_shape_provider,
        )

        if len(self.index.values) == 1 and not self.index.values[0].subscriptable():
            if None not in shape and np.sum(shape) == 0 and self.meta.max_shape:  # type: ignore
                shape = (0,) * len(self.meta.max_shape)
        if self.meta.max_shape == [0, 0, 0]:
            shape = ()
        return shape

    @property
    def size(self) -> Optional[int]:
        """Returns the size. """
        s = 1
        for x in self.shape:
            if x is None:
                return None
            s *= x  # not using np.prod to avoid overflow
        return s

    @property
    def ndim(self) -> int:
        """Number of dimensions of the tensor."""
        return self.chunk_engine.ndim(self.index)

    @property
    def dtype(self) -> Optional[np.dtype]:
        """Dtype of the tensor."""
        if self.base_htype in ("json", "list", "tag"):
            return np.dtype(str)
        if self.meta.dtype:
            return np.dtype(self.meta.typestr or self.meta.dtype)
        return None

    @property
    def is_sequence(self):
        """Whether this tensor is a sequence tensor."""
        return self.meta.is_sequence

    @property
    def htype(self):
        """Htype of the tensor."""
        htype = self.meta.htype
        if self.is_sequence:
            htype = f"sequence[{htype}]"
        return htype

    @property
    def hidden(self) -> bool:
        """Whether this tensor is a hidden tensor."""
        return self.meta.hidden

    @property
    def base_htype(self):
        """Base htype of the tensor.

        Example:

            >>> ds.create_tensor("video_seq", htype="sequence[video]", sample_compression="mp4")
            >>> ds.video_seq.htype
            sequence[video]
            >>> ds.video_seq.base_htype
            video
        """
        return self.meta.htype

    @property
    def shape_interval(self) -> ShapeInterval:
        """Returns the shape interval. """
        sample_shape_tensor = self.get_sample_shape_tensor()
        sample_shape_provider = (
            self._sample_shape_provider(sample_shape_tensor)
            if sample_shape_tensor
            else None
        )
        return self.chunk_engine.shape_interval(self.index, sample_shape_provider)

    @property
    def is_dynamic(self) -> bool:
        """Will return ``True`` if samples in this tensor have shapes that are unequal."""
        return self.shape_interval.is_dynamic

    @property
    def num_samples(self) -> int:
        """Returns the length of the primary axis of the tensor.
        Ignores any applied indexing and returns the total length.
        """
        return self.chunk_engine.tensor_length

    @property
    def _sample_id_tensor(self):
        if not CREATE_TENSOR_HIDDEN_UUID:
            return self.dataset[DATASET_UUID_NAME]
        tensor_name = self.meta.name or self.key
        return self.dataset.get_tensors().get(get_sample_id_tensor_key(tensor_name))

    @property
    def sample_info(self) -> Union[Dict, List[Dict]]:
        """Returns info about particular samples in a tensor. Returns dict in case of single sample,
           otherwise list of dicts.
        Data in returned dict would depend on the tensor's htype and the sample itself.

        Example:

            >>> ds.videos[0].sample_info
            {'duration': 400400, 'fps': 29.97002997002997, 'timebase': 3.3333333333333335e-05,
            'shape': [400, 360, 640, 3], 'format': 'mp4', 'filename': '../muller/tests/dummy_data/video/samplemp4.mp4',
            'modified': False}
            >>> ds.images[:2].sample_info
            [{'exif': {'Software': 'Google'}, 'shape': [900, 900, 3], 'format': 'jpeg',
            'filename': '../muller/tests/dummy_data/images/cat.jpeg', 'modified': False},
            {'exif': {}, 'shape': [495, 750, 3], 'format': 'jpeg',
            'filename': '../muller/tests/dummy_data/images/car.jpg', 'modified': False}]
        """
        return self._sample_info(self.index)

    @property
    def config(self):
        """Returns a summary of the configuration of the tensor."""
        tensor_meta = self.meta
        return {
            "htype": tensor_meta.htype,
            "dtype": tensor_meta.dtype,
            "sample_compression": tensor_meta.sample_compression,
            "chunk_compression": tensor_meta.chunk_compression,
            "hidden": tensor_meta.hidden,
            "is_sequence": tensor_meta.is_sequence,
        }

    @property
    def sample_indices(self):
        """Returns all the indices pointed to by this tensor in the dataset view."""
        return self.dataset.get_sample_indices(self.num_samples)

    @info.setter
    def info(self, value):
        """Set the info."""
        if isinstance(value, dict):
            info = self.info
            info.replace_with(value)
            self.meta.replace_info(value)
        else:
            raise TypeError("Info must be set with type Dict")

    @property
    def is_empty_tensor(self):
        """Returns whether it is an empty tensor. """
        if (
                self.meta.chunk_compression
                and get_compression_type(self.meta.chunk_compression) != BYTE_COMPRESSION
        ):
            return self.meta.max_shape == [0, 0, 0] or len(self.meta.max_shape) == 0
        return len(self.meta.max_shape) == 0

    @htype.setter
    def htype(self, value):
        """Returns the htype. """
        self._check_compatibility_with_htype(value)
        self.meta.htype = value
        if value == "class_label":
            self.meta.disable_temp_transform = False
        self.meta.is_dirty = True
        self.dataset.maybe_flush()

    @property
    def skip_next_setitem(self):
        """Returns whether skip the next setitem. """
        return self._skip_next_setitem

    @skip_next_setitem.setter
    def skip_next_setitem(self, value):
        """Set the value of self._skip_next_setitem. """
        self._skip_next_setitem = value

    @staticmethod
    def _get_bigger_dtype(d1, d2):
        if np.can_cast(d1, d2):
            if np.can_cast(d2, d1):
                return d1
            return d2

        if np.can_cast(d2, d1):
            return d2
        return np.object

    @invalid_view_op
    @user_permission_check
    def append(self,
               sample: InputSample,
               ignore_errors: bool = False):
        """Append samples."""
        self.protected_extend([sample], progressbar=False, ignore_errors=ignore_errors)

    @invalid_view_op
    @user_permission_check
    def extend(
            self,
            samples: Union[np.ndarray, Sequence[InputSample], "Tensor"],
            progressbar: bool = False,
            ignore_errors: bool = False,
    ):
        """Extend samples."""
        self.protected_extend(samples, progressbar=progressbar, ignore_errors=ignore_errors)

    @invalid_view_op
    @user_permission_check
    def clear(self):
        """Deletes all samples from the tensor"""
        self.chunk_engine.clear()

    @user_permission_check
    def _update(self, item: Union[int, slice], value: Any):
        self.write_initialization()
        if isinstance(value, Tensor):
            if value.skip_next_setitem:
                value.skip_next_setitem = False
                return
            value = value.numpy(aslist=True)
        item_index = Index(item)
        if (
                muller.constants.ENABLE_RANDOM_ASSIGNMENT
                and isinstance(item, int)
                and item >= self.num_samples
        ):
            if self.is_sequence:
                raise NotImplementedError(
                    "Random assignment is not supported for sequences yet."
                )
            num_samples_to_pad = item - self.num_samples

            self.chunk_engine.pad_and_append(
                num_samples_to_pad,
                value,
            )
        else:
            if not item_index.values[0].subscriptable() and not self.is_sequence:
                # we're modifying a single sample, convert it to a list as chunk engine expects multiple samples
                value = [value]

            self.chunk_engine.update(
                self.index[item_index],
                value,
            )
        self.dataset.append_only = False

    def get_array(self, dtype=None) -> np.ndarray:
        """Returns the tensor as array. """
        ret = self.numpy()  # type: ignore
        if self.base_htype == "polygon":
            return np.array(ret, dtype=dtype)
        if dtype and ret.dtype != dtype:  # type: ignore
            ret = ret.astype(dtype)  # type: ignore
        return ret  # type: ignore

    def get_sample_info_tensor(self):
        """Get sample info tensor. """
        ds = self.dataset
        tensor_name = self.meta.name or self.key
        return ds.version_state["full_tensors"].get(
            ds.version_state["tensor_names"].get(
                get_sample_info_tensor_key(tensor_name)
            )
        )

    def get_sample_shape_tensor(self):
        """Get sample shape tensor. """
        ds = self.dataset
        tensor_name = self.meta.name or self.key
        return ds.version_state["full_tensors"].get(
            ds.version_state["tensor_names"].get(
                get_sample_shape_tensor_key(tensor_name)
            )
        )

    def pop(self, index: Optional[Union[int, List[int]]] = None):
        """Removes element(s) at the given index / indices."""

        # 先确认不是直接外部调用
        caller_frame = inspect.stack()[1] # 获取调用栈信息
        caller_module = inspect.getmodule(caller_frame[0]) # 获取调用者模块
        if caller_module.__name__ not in ["muller.core.dataset.dataset",
                                          "muller.api.info",
                                          "muller.core.tensor",
                                          "muller.util.merge"]:
            raise UnsupportedMethod("Currently, we do not support directly pop from a single tensor column. "
                                    "Please consider pop from dataset.")

        if index is None:
            index = [self.num_samples - 1]

        if not isinstance(index, list):
            index = [index]

        if not index:
            return

        if len(set(index)) != len(index):
            raise ValueError("Duplicate indices are not allowed.")

        length = self.num_samples
        if length == 0:
            raise IndexError("Can't pop from empty tensor")

        for idx in index:
            if idx < 0:
                raise IndexError("Pop doesn't support negative indices.")
            if idx >= length:
                raise IndexError(
                    f"Index {idx} is out of range. The tensor has {length} samples."
                )

        index = sorted(index, reverse=True)
        self.protected_pop(index)


    def shapes(self):
        """Get the shapes of all the samples in the tensor.

        Returns:
            np.ndarray: List of shapes of all the samples in the tensor.
        """
        sample_shape_tensor = self.get_sample_shape_tensor()
        sample_shape_provider = (
            self._sample_shape_provider(sample_shape_tensor)
            if sample_shape_tensor
            else None
        )
        return self.chunk_engine.shapes(
            self.index,
            sample_shape_provider=sample_shape_provider,
        )

    def arrow(self):
        """Returns as arrow. """
        ret = self.chunk_engine.arrow(self.index)
        return ret

    def numpy(
            self, aslist=False, fetch_chunks=False, max_workers=MAX_WORKERS_FOR_CHUNK_ENGINE
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Computes the contents of the tensor in numpy format.

        Args:
            aslist (bool): If ``True``, a list of np.ndarrays will be returned. Helpful for dynamic tensors.
                If ``False``, a single np.ndarray will be returned unless the samples are dynamically shaped,
                in which case an error is raised.
            fetch_chunks (bool): If ``True``, full chunks will be retrieved from the storage, otherwise only
                required bytes will be retrieved.
                This will always be ``True`` even if specified as ``False`` in the following cases:

                - The tensor is ChunkCompressed.
                - The chunk which is being accessed has more than 128 samples.
            max_workers (int): max workers used in thread pool.

        Raises:
            DynamicTensorNumpyError: If reading a dynamically-shaped array slice without ``aslist=True``.
            ValueError: If the tensor is a link and the credentials are not populated.

        Returns:
            A numpy array containing the data represented by this tensor.

        Note:
            For tensors of htype ``polygon``, aslist is always ``True``.
        """
        ret = self.chunk_engine.numpy(
            self.index,
            aslist=aslist,
            fetch_chunks=fetch_chunks or self.is_iteration,
            max_workers=max_workers,
        )
        return ret

    def numpy_continuous(
            self, aslist=False, fetch_chunks=False, max_workers=MAX_WORKERS_FOR_CHUNK_ENGINE
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Computes the continuous contents of the tensor in numpy format.

        Args:
            aslist (bool): If ``True``, a list of np.ndarrays will be returned. Helpful for dynamic tensors.
                If ``False``, a single np.ndarray will be returned unless the samples are dynamically shaped,
                in which case an error is raised.
            fetch_chunks (bool): If ``True``, full chunks will be retrieved from the storage,
                otherwise only required bytes will be retrieved.
                This will always be ``True`` even if specified as ``False`` in the following cases:

                - The tensor is ChunkCompressed.
                - The chunk which is being accessed has more than 128 samples.
            max_workers (int): max workers used in thread pool.

        Raises:
            DynamicTensorNumpyError: If reading a dynamically-shaped array slice without ``aslist=True``.
            ValueError: If the tensor is a link and the credentials are not populated.

        Returns:
            A numpy array containing the data represented by this tensor.

        Note:
            For tensors of htype ``polygon``, aslist is always ``True``.
        """
        ret = self.chunk_engine.numpy(
            self.index,
            aslist=aslist,
            fetch_chunks=fetch_chunks or self.is_iteration,
            max_workers=max_workers,
            continuous=True,
        )
        return ret

    def numpy_full(
            self, aslist=False, fetch_chunks=False, max_workers=MAX_WORKERS_FOR_CHUNK_ENGINE
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Computes the continuous contents of the tensor in numpy format."""
        ret = self.chunk_engine.numpy(
            self.index,
            aslist=aslist,
            fetch_chunks=True,
            max_workers=max_workers,
            full=True,
        )
        return ret

    def numpy_batch_random_access(
            self, aslist=True, index_list=None, max_workers=MAX_WORKERS_FOR_CHUNK_ENGINE, parallel: Optional[str] = None
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Computes the continuous contents of the tensor in numpy format."""
        ret = self.chunk_engine.numpy(
            self.index,
            aslist=aslist,
            fetch_chunks=True,
            max_workers=max_workers,
            batch_random_access=True,
            index_list=index_list,
            parallel=parallel
        )
        return ret

    def numpy_multi_process(self, aslist=False, fetch_chunks=False, max_workers=50,
                            batch_size=DEFAULT_MAX_NUMPY_BATCH_SIZE):
        """Computes the contents of the tensor in numpy format with multi processes.

        Args:
            aslist (bool): If ``True``, a list of np.ndarrays will be returned. Helpful for dynamic tensors.
                If ``False``, a single np.ndarray will be returned unless the samples are dynamically shaped,
                in which case an error is raised.
            fetch_chunks (bool): If ``True``, full chunks will be retrieved from the storage, otherwise only required
                bytes will be retrieved.
                This will always be ``True`` even if specified as ``False`` in the following cases:

                - The tensor is ChunkCompressed.
                - The chunk which is being accessed has more than 128 samples.
            max_workers (int): max workers used in thread pool.
            batch_size (int): max data size a process can deal with

        Raises:
            DynamicTensorNumpyError: If reading a dynamically-shaped array slice without ``aslist=True``.
            ValueError: If the tensor is a link and the credentials are not populated.

        Returns:
            Shared Memory Object which stores the returned numpy data. The returned 'shm' must be shut down
            by shm.close() and shm.unlink() to avoid memory leak.
            A numpy array containing the data represented by this tensor.
        """
        if self.htype not in ('vector', 'generic'):
            raise MultiProcessUnsupportedError(False, self.htype, self.dtype)
        if self.meta.min_shape != self.meta.max_shape or len(self.meta.min_shape) > 2:
            raise MultiProcessUnsupportedError(True, self.htype, self.dtype)
        if self.htype == "vector":
            vector_dimension = self.info.dimension
        else:
            vector_dimension = self.meta.min_shape[-1]
        batch_tensors = []
        start = 0
        idxs = []
        dataset_length = len(self)
        while start < dataset_length:
            batch_tensors.append(self[start:start + batch_size])
            idxs.append([start, min(start + batch_size, dataset_length)])
            start = start + batch_size
        shared_memory_name = str(uuid.uuid4())
        shm = shared_memory.SharedMemory(create=True, size=dataset_length * vector_dimension * self.dtype.itemsize,
                                         name=shared_memory_name)
        batch_tensors_length = len(batch_tensors)
        with ProcessPoolExecutor(max_workers=min(max_workers, batch_tensors_length)) as executor:
            executor.map(_numpy_parallel, batch_tensors,
                         [fetch_chunks] * batch_tensors_length, idxs, [vector_dimension] * batch_tensors_length,
                         [shared_memory_name] * batch_tensors_length, [self.dtype.itemsize] * batch_tensors_length)
        data = np.ndarray((dataset_length, vector_dimension), dtype=self.dtype, buffer=shm.buf)
        if aslist:
            data = list(data)
            return shm, data
        return shm, data

    def text(self, fetch_chunks: bool = False):
        """Return text data. Only applicable for tensors with 'text' base htype."""
        return self._extract_value("text", fetch_chunks=fetch_chunks)

    def dict(self, fetch_chunks: bool = False):
        """Return json data. Only applicable for tensors with 'json' base htype."""
        return self._extract_value("json", fetch_chunks=fetch_chunks)

    def list(self, fetch_chunks: bool = False):
        """Return list data. Only applicable for tensors with 'list' or 'tag' base htype."""
        if self.base_htype not in ("list", "tag"):
            raise Exception("Only supported for list and tag tensors.")

        if self.ndim == 1:
            return list(self.numpy(fetch_chunks=fetch_chunks))

        return list(map(list, self.numpy(aslist=True, fetch_chunks=fetch_chunks)))

    def data(self, aslist: bool = False, fetch_chunks: bool = False) -> Any:
        """Returns data in the tensor in a format based on the tensor's base htype.

        - If tensor has ``text`` base htype
            - Returns dict with dict["value"] = :meth:`Tensor.text() <text>`

        - If tensor has ``json`` base htype
            - Returns dict with dict["value"] = :meth:`Tensor.dict() <dict>`

        - If tensor has ``list`` base htype
            - Returns dict with dict["value"] = :meth:`Tensor.list() <list>`

        - For ``video`` tensors, returns a dict with keys "frames", "timestamps" and "sample_info":

            - Value of dict["frames"] will be same as :meth:`numpy`.
            - Value of dict["timestamps"] will be same as :attr:`timestamps` corresponding to the frames.
            - Value of dict["sample_info"] will be same as :attr:`sample_info`.

        - For ``class_label`` tensors, returns a dict with keys "value" and "text".

            - Value of dict["value"] will be same as :meth:`numpy`.
            - Value of dict["text"] will be list of class labels as strings.

        - For ``image`` or ``dicom`` tensors, returns dict with keys "value" and "sample_info".

            - Value of dict["value"] will be same as :meth:`numpy`.
            - Value of dict["sample_info"] will be same as :attr:`sample_info`.

        - For all else, returns dict with key "value" with value same as :meth:`numpy`.
        """
        htype = self.base_htype
        if htype == "text":
            return {"value": self.text(fetch_chunks=fetch_chunks)}
        if htype == "json":
            return {"value": self.dict(fetch_chunks=fetch_chunks)}
        if htype in ("list", "tag"):
            return {"value": self.list(fetch_chunks=fetch_chunks)}
        if self.htype == "video":
            data = {}
            data["frames"] = self.numpy(aslist=aslist, fetch_chunks=fetch_chunks)
            index = self.index
            if index.values[0].subscriptable():
                root = Tensor(self.key, self.dataset)
                if len(index.values) > 1:
                    data["timestamps"] = np.array(
                        [
                            root[i, index.values[1].value].timestamps  # type: ignore
                            for i in index.values[0].indices(self.num_samples)
                        ]
                    )
                else:
                    data["timestamps"] = [
                        root[i].timestamps
                        for i in index.values[0].indices(self.num_samples)
                    ]
            else:
                data["timestamps"] = self.timestamps
            if not aslist:
                try:
                    data["timestamps"] = np.array(data["timestamps"])  # type: ignore
                except ValueError:
                    data["timestamps"] = np.array(data["timestamps"], dtype=object)  # type: ignore

            data["sample_info"] = self.sample_info  # type: ignore
            return data
        if htype == "class_label":
            labels = self.numpy(aslist=aslist, fetch_chunks=fetch_chunks)
            data = {"value": labels}
            class_names = self.info.class_names
            if class_names:
                data["text"] = convert_to_text(labels, class_names)
            return data
        if htype in ("image", "image.rgb", "image.gray", "dicom", "nifti"):
            return {
                "value": self.numpy(aslist=aslist, fetch_chunks=fetch_chunks),
                "sample_info": self.sample_info or {},
            }

        try:
            return {
                "value": self.chunk_engine.numpy(
                    index=self.index, aslist=aslist, fetch_chunks=fetch_chunks
                ),
            }
        except NotImplementedError:
            return {
                "value": self.numpy(aslist=aslist, fetch_chunks=fetch_chunks),
            }

    def tobytes(self) -> bytes:
        """Returns the bytes of the tensor.

        - Only works for a single sample of tensor.
        - If the tensor is uncompressed, this returns the bytes of the numpy array.
        - If the tensor is sample compressed, this returns the compressed bytes of the sample.
        - If the tensor is chunk compressed, this raises an error.

        Returns:
            bytes: The bytes of the tensor.

        Raises:
            ValueError: If the tensor has multiple samples.
        """
        if self.index.values[0].subscriptable() or len(self.index.values) > 1:
            raise ValueError("tobytes() can be used only on exactly 1 sample.")
        idx = self.index.values[0].value
        ret = self.chunk_engine.read_bytes_for_sample(idx)  # type: ignore
        return ret

    def protected_extend(
            self,
            samples: Union[np.ndarray, Sequence[InputSample], "Tensor"],
            progressbar: bool = False,
            ignore_errors: bool = False,
    ):
        """Extend samples to tensor."""
        self.write_initialization()
        self.chunk_engine.extend(
            samples,
            progressbar=progressbar,
            ignore_errors=ignore_errors,
            is_uuid=(self.key == DATASET_UUID_NAME)
        )
        if self.key != DATASET_UUID_NAME:
            self.dataset.resize_uuid()

    def protected_append(self, sample: InputSample):
        """Append sample to tensor."""
        self.protected_extend([sample], progressbar=False)

    def protected_pop(self, index: List[int], rechunk: bool = False):
        """Removes elements at the given indices. ``index`` must be sorted in descending order."""
        self.chunk_engine.pop(
            index,
            sample_id_tensor=self._sample_id_tensor,
            rechunk=rechunk,
        )

    def write_initialization(self):
        """Initialize the writing. """
        self.storage.check_readonly()
        # if not the head node, checkout to an auto branch that is newly created
        if auto_checkout(self.dataset):
            self.chunk_engine = self.version_state["full_tensors"][
                self.key
            ].chunk_engine

    def _infer_np_dtype(self, val: Any) -> np.dtype:
        # Sherry: may need refactor
        if hasattr(val, "dtype"):
            return val.dtype
        if isinstance(val, int):
            return np.array(0).dtype
        if isinstance(val, float):
            return np.array(0.0).dtype
        if isinstance(val, str):
            return np.array("").dtype
        if isinstance(val, bool):
            return np.dtype(bool)
        if isinstance(val, Sequence):
            return reduce(self._get_bigger_dtype, map(self._infer_np_dtype, val))

        raise TypeError(f"Cannot infer numpy dtype for {val}")

    def _sample_shape_provider(self, sample_shape_tensor) -> Callable:
        if self.is_sequence:

            def get_sample_shape(global_sample_index: int):
                assert self.chunk_engine.sequence_encoder is not None
                seq_pos = slice(
                    *self.chunk_engine.sequence_encoder[global_sample_index]
                )
                idx = Index([IndexEntry(seq_pos)])
                shapes = sample_shape_tensor[idx].numpy(fetch_chunks=True)
                return shapes

        else:
            def get_sample_shape(global_sample_index: int):
                return tuple(
                    sample_shape_tensor[global_sample_index]
                    .numpy(fetch_chunks=True)
                    .tolist()
                )

        return get_sample_shape

    def _get_sample_info_at_index(self, global_sample_index: int, sample_info_tensor):
        if self.is_sequence:
            assert self.chunk_engine.sequence_encoder is not None
            return [
                sample_info_tensor[i].data()
                for i in range(*self.chunk_engine.sequence_encoder[global_sample_index])
            ]
        return sample_info_tensor[global_sample_index].data()["value"]

    def _sample_info(self, index: Index):
        sample_info_tensor = self.get_sample_info_tensor()
        if sample_info_tensor is None:
            return None
        if index.subscriptable_at(0):
            return list(
                map(
                    partial(
                        self._get_sample_info_at_index,
                        sample_info_tensor=sample_info_tensor,
                    ),
                    index.values[0].indices(self.num_samples),
                )
            )
        return self._get_sample_info_at_index(index.values[0].value, sample_info_tensor)  # type: ignore

    def _extract_value(self, htype: str, fetch_chunks: bool = False):
        if self.base_htype != htype:
            raise Exception(f"Only supported for {htype} tensors.")

        if self.ndim == 1:
            data = self.numpy(fetch_chunks=fetch_chunks)
            if len(data) == 0:
                return []
            return data[0]

        data = self.numpy(aslist=True, fetch_chunks=fetch_chunks)
        if len(data) == 0:
            return []
        return [sample[0] for sample in data]

    def _check_compatibility_with_htype(self, htype):
        """Checks if the tensor is compatible with the given htype.
        Raises an error if not compatible.
        """
        is_sequence, _, htype = parse_complex_htype(htype)
        if is_sequence:
            raise ValueError(f"Cannot change htype to a sequence.")
        _validate_htype_exists(htype)
        if self.htype not in HTYPE_CONVERSION_LHS:
            raise NotImplementedError(
                f"Changing the htype of a tensor of htype {self.htype} is not supported."
            )
        if htype not in HTYPE_CONSTRAINTS:
            raise NotImplementedError(
                f"Changing the htype to {htype} is not supported."
            )
        compression = self.meta.sample_compression or self.meta.chunk_compression
        if compression:
            supported_compressions = HTYPE_SUPPORTED_COMPRESSIONS.get(htype)
            if supported_compressions and compression not in supported_compressions:
                raise UnsupportedCompressionError(compression, htype)
        constraints = HTYPE_CONSTRAINTS[htype]
        constraints(self.shape, self.dtype)

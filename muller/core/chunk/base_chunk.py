# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/core/chunk/base_chunk.py
#
# Modifications Copyright (c) 2026 Xueling Lin

import warnings
from abc import abstractmethod
from functools import reduce, wraps
from operator import mul
from typing import List, Optional, Tuple, Union
from typing import Sequence

import numpy as np

import muller
from muller.constants import CONVERT_GRAYSCALE
from muller.core.compression import get_compression_type
from muller.core.fast_forwarding import ffw_chunk
from muller.core.meta.encode.byte_positions import BytePositionsEncoder
from muller.core.meta.encode.shape import ShapeEncoder
from muller.core.meta.tensor_meta import TensorMeta
from muller.core.partial_reader import PartialReader
from muller.core.partial_sample import PartialSample
from muller.core.sample import Sample  # type: ignore
from muller.core.serialize import deserialize_chunk, serialize_text_sample_object
from muller.core.serialize import (serialize_numpy_and_base_types,
                                  serialize_text,
                                  serialize_tensor,
                                  serialize_partial_sample_object,
                                  serialize_sample_object,
                                  serialize_chunk)
from muller.core.storage.muller_memory_object import MULLERMemoryObject
from muller.core.tiling.sample_tiles import SampleTiles
from muller.util.exceptions import EmptyTensorError, ReadSampleFromChunkError
from muller.util.exceptions import TensorInvalidSampleShapeError

InputSample = Union[
    Sample,
    np.ndarray,
    int,
    float,
    bool,
    dict,
    list,
    str,
    np.integer,
    np.floating,
    np.bool_,
]

SerializedOutput = Tuple[bytes, Tuple]


class BaseChunk(MULLERMemoryObject):
    def __init__(
            self,
            min_chunk_size: int,
            max_chunk_size: int,
            tiling_threshold: int,
            tensor_meta: TensorMeta,
            compression: Optional[str] = None,
            encoded_shapes: Optional[np.ndarray] = None,
            encoded_byte_positions: Optional[np.ndarray] = None,
            data: Optional[Union[memoryview, PartialReader]] = None,
    ):
        super().__init__()
        self._data_bytes: Union[bytearray, bytes, memoryview, PartialReader] = (
                data or bytearray()
        )
        self.version = muller.__version__
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.tiling_threshold = tiling_threshold

        self.tensor_meta = tensor_meta
        self.num_dims = (
             (len(tensor_meta.max_shape) if tensor_meta.max_shape else None)
        )
        self.is_text_like = (
                self.htype in {"json", "list", "text"}
        )
        self.compression = compression
        compression_type = get_compression_type(compression)
        self.is_byte_compression = compression_type == "byte"
        self.is_image_compression = compression_type == "image"
        # Sherry: should support other compression type later
        self.is_convert_candidate = self.htype == "image" or self.is_image_compression
        self.shapes_encoder = ShapeEncoder(encoded_shapes)
        self.byte_positions_encoder = BytePositionsEncoder(encoded_byte_positions)

        if self.is_text_like and self.is_image_compression:
            raise ValueError("Can't use image compression with text data.")
        # These caches are only used for ChunkCompressed chunk.
        self.decompressed_samples: Optional[List[np.ndarray]] = None
        self.decompressed_bytes: Optional[bytes] = None

        # Whether tensor meta length is updated by chunk. Used by chunk engine while replacing chunks.
        self._update_tensor_meta_length: bool = (
            True  # Note: tensor meta shape interval is updated regardless.
        )
        self._item_size = None
        self._sample_size = None
        self.write_initialization_done = False
        self.id: Optional[str] = None
        self.key: Optional[str] = None

    @property
    def header_bytes(self):
        """Returns header bytes."""
        return infer_header_num_bytes(
            self.version, self.shapes_encoder.array, self.byte_positions_encoder.array
        )

    @property
    def data_bytes(self) -> Union[bytearray, bytes, memoryview, PartialReader]:
        """Returns data bytes."""
        return self._data_bytes

    @data_bytes.setter
    def data_bytes(self, value: Union[bytearray, bytes, memoryview, PartialReader]):
        """Set data bytes."""
        self._data_bytes = value

    @property
    def htype(self):
        """Returns type of tensor meta. """
        return self.tensor_meta.htype

    @property
    def dtype(self):
        """Returns type of tensor meta. """
        return self.tensor_meta.dtype

    @property
    def is_empty(self):
        """Returns True if empty."""
        return (
                self.num_data_bytes == 0
                and len(self.shapes_encoder.array) == 0
                and len(self.byte_positions_encoder.array) == 0
        )

    @property
    def is_empty_tensor(self):
        """Returns True if empty."""
        return len(self.tensor_meta.max_shape) == 0 and (
                not isinstance(self.data_bytes, PartialReader) and len(self.data_bytes) == 0
        )

    @property
    def is_fixed_shape(self):
        """Returns True if shape is fixed."""
        return (
                self.tensor_meta.min_shape == self.tensor_meta.max_shape
                and not self.is_text_like
        )

    @property
    def item_size(self):
        """Returns item size."""
        # should only be called if self.is_fixed_shape
        if self._item_size is None:
            if self.dtype is None:
                raise ValueError("Can't get item size as dtype is not set.")
            self._item_size = np.dtype(self.dtype).itemsize
        return self._item_size

    @property
    def sample_size(self):
        """Returns sample size."""
        # should only be called if self.is_fixed_shape
        shape = self.tensor_meta.max_shape
        if self._sample_size is None:
            self._sample_size = self.item_size * reduce(mul, shape, 1)
        return self._sample_size

    @property
    def memoryview_data(self):
        """Returns memory view."""
        if isinstance(self.data_bytes, (memoryview, PartialReader)):
            return self.data_bytes
        return memoryview(self.data_bytes)

    @property
    def num_data_bytes(self) -> int:
        """Returns num data bytes."""
        return len(self._data_bytes)

    @property
    def nbytes(self):
        """
        Calculates the number of bytes `tobytes` will be without having to call `tobytes`.
        Used by `LRUCache` to determine if this chunk can be cached.
        """
        return _infer_chunk_num_bytes(
            self.version,
            self.shapes_encoder.array,
            self.byte_positions_encoder.array,
            data=None,
            len_data=self.num_data_bytes,
        )

    @property
    def num_samples(self) -> int:
        """Returns number of samples."""
        if not self.shapes_encoder.is_empty():
            return self.shapes_encoder.num_samples

        return self.byte_positions_encoder.num_samples

    @property
    def update_tensor_meta_length(self):
        """Returns the update tensor meta length."""
        return self._update_tensor_meta_length

    @update_tensor_meta_length.setter
    def update_tensor_meta_length(self, value):
        """Set the update tensor data length."""
        self._update_tensor_meta_length = value

    @staticmethod
    def normalize_shape(shape):
        """Normalize the shape"""
        if shape is not None and len(shape) == 0:
            shape = (1,)
        return shape

    @staticmethod
    def infer_chunk_num_bytes(
            version: str,
            shape_info: np.ndarray,
            byte_positions: np.ndarray,
            data: Optional[Union[Sequence[bytes], Sequence[memoryview]]] = None,
            len_data: Optional[int] = None,
    ) -> int:
        """Calculates the number of bytes in a chunk without serializing it.
        Used by `LRUCache` to determine if a chunk can be cached.

        Args:
            version: (str) Version of muller library
            shape_info: (numpy.ndarray) Encoded shapes info from the chunk's `ShapeEncoder` instance.
            byte_positions: (numpy.ndarray) Encoded byte positions from the chunk's `BytePositionsEncoder` instance.
            data: (list) `_data` field of the chunk
            len_data: (int, optional) Number of bytes in the chunk

        Returns:
            Length of the chunk when serialized as int
        """
        # NOTE: Assumption: version string contains ascii characters only (ord(c) < 128)
        # NOTE: Assumption: len(version) < 256
        if len_data is None:
            len_data = sum(map(len, data))  # type: ignore

        header_size = infer_header_num_bytes(version, shape_info, byte_positions)
        return header_size + len_data

    @staticmethod
    def _text_sample_to_byte_string(sample):
        try:
            return str(sample.numpy().reshape(())).encode("utf-8")
        except AttributeError:
            pass
        try:
            return sample.encode("utf-8")
        except AttributeError:
            try:
                return str(sample.reshape(())).encode("utf-8")
            except AttributeError:  # None
                return b""

    @classmethod
    def frombuffer(cls, buffer: bytes, chunk_args: list, copy=True, url=False, partial=False):  # type: ignore
        """Read from buffer. """
        if not buffer:
            return cls(*chunk_args)

        version, shapes, byte_positions, data = deserialize_chunk(buffer, copy=copy)
        if partial:
            data = None

        chunk = cls(*chunk_args, shapes, byte_positions, data=data)  # type: ignore
        chunk.version = version  # Sherry: delete it?
        chunk.is_dirty = False
        return chunk

    @abstractmethod
    def extend_if_has_space(
        self,
        incoming_samples,
        update_tensor_meta: bool = True,
        lengths: Optional[List[int]] = None,
        ignore_errors: bool = False,
        **kwargs,
    ) -> float:
        """Extends the chunk with the incoming samples."""

    @abstractmethod
    def read_sample(
            self,
            local_index: int,  # The local index of the sample in the chunk.
            cast: bool = True,
            copy: bool = False,  # Whether conduct hard copy (defaults to be False)
            decompress: bool = True,  # Whether decompress (defaults to be True)
            is_tile: bool = False,  # Whether it is tiled (defaults to be False)
            **kwargs,
    ):
        """Reads a sample from the chunk."""

    def register_sample_to_headers(
        self,
        incoming_num_bytes: Optional[int],
        sample_shape: Union[None, Tuple[int]],
        num_samples: int = 1,
    ):
        """Registers sample to headers."""
        # incoming_num_bytes is not applicable for image compressions
        if incoming_num_bytes is not None:
            self.byte_positions_encoder.register_samples(
                incoming_num_bytes, num_samples
            )
        if sample_shape is not None:
            if self.shapes_encoder.is_empty():
                padding = self.byte_positions_encoder.num_samples - num_samples
                self._fill_empty_shapes(sample_shape, padding)
            self.shapes_encoder.register_samples(sample_shape, num_samples)

    def update_tensor_meta(self, shape, num_samples):
        """Update tensor meta. """
        if self._update_tensor_meta_length:
            self.tensor_meta.update_length(num_samples)
        if shape is not None:
            self.tensor_meta.update_shape_interval(shape)

    def register_in_meta_and_headers(
        self,
        sample_nbytes: Optional[int],
        shape,
        update_tensor_meta: bool = True,
        num_samples: int = 1,
    ):
        """Registers a new sample in meta and headers

        Args:
           sample_nbytes (Optional[int]): Parameter shat shows the numbero of bytes
           shape (Any): Parameter that shows the shape of the added elements
           update_commit_diff (bool): Parameter that shows if we need to update the commit diffs
           update_tensor_meta (bool): Parameter that shows if it is need to update tensor metas,
                in case of rechunk we do not need to update meta as we do not add new elements
           num_samples (int): Number of incoming samples.
        """
        self.register_sample_to_headers(sample_nbytes, shape, num_samples)
        if update_tensor_meta:
            self.update_tensor_meta(shape, num_samples)

    def tobytes(self) -> memoryview:
        """Returns the bytes of this chunk. """
        if isinstance(self.data_bytes, PartialReader):
            self.make_data_bytearray()

        assert isinstance(self.data_bytes, (memoryview, bytearray, bytes))
        return serialize_chunk(
            self.version,
            self.shapes_encoder.array,
            self.byte_positions_encoder.array,
            [self.data_bytes],
        )

    def prepare_for_write(self):
        """Prepare for write. """
        if not self.write_initialization_done:
            ffw_chunk(self)
            self.write_initialization_done = True
        self.make_data_bytearray()
        self.is_dirty = True

    def copy(self, chunk_args=None):
        """Copy this chunk. """
        return self.frombuffer(bytes(self.tobytes()), chunk_args)

    def convert_to_rgb(self, shape):
        """Convert to RGB (is this is image.)"""
        if shape is not None and self.is_convert_candidate and CONVERT_GRAYSCALE:
            ndim = len(shape)
            if self.num_dims is None:
                self.num_dims = max(3, ndim)
            if ndim < self.num_dims:
                message = "Grayscale images will be reshaped from (H, W) to (H, W, 1) to match tensor dimensions. " \
                          "This warning will be shown only once."
                warnings.warn(message)
                shape += (1,) * (self.num_dims - ndim)  # type: ignore[assignment]
        return shape

    def serialize_sample(
            self,
            incoming_sample: InputSample,
            sample_compression: Optional[str] = None,
            chunk_compression: Optional[str] = None,
            break_into_tiles: bool = True,
            store_uncompressed_tiles: bool = False,
    ) -> SerializedOutput:
        """Converts the sample into bytes"""
        dt, ht, min_chunk_size, tiling_threshold = (
            self.dtype,
            self.htype,
            self.min_chunk_size,
            self.tiling_threshold,
        )
        if tiling_threshold < 0:
            break_into_tiles = False

        if self.is_text_like:
            if incoming_sample is None:
                htype = self.htype
                empty_mapping = {"text": "", "list": [], "json": {}}
                incoming_sample = empty_mapping.get(htype, None)

            if isinstance(incoming_sample, Sample):
                if incoming_sample.is_text_like:
                    incoming_sample, shape = serialize_text_sample_object(  # type: ignore
                        incoming_sample, sample_compression
                    )
                else:
                    htype = self.htype
                    raise TypeError(
                        f"Cannot append to {htype} tensor with Sample object"
                    )
            else:
                incoming_sample, shape = serialize_text(
                    incoming_sample, sample_compression, dt, ht  # type: ignore
                )
        elif incoming_sample is None:
            shape = (0,) * self.num_dims if self.num_dims else None
            incoming_sample = b""
        elif isinstance(incoming_sample, Sample):
            incoming_sample, shape = serialize_sample_object(  # type: ignore
                incoming_sample,
                sample_compression,
                chunk_compression,
                dt,
                ht,
                tiling_threshold,
                break_into_tiles,
                store_uncompressed_tiles,
            )
        elif isinstance(incoming_sample, PartialSample):
            incoming_sample, shape = serialize_partial_sample_object(
                incoming_sample,
                sample_compression,
                chunk_compression,
                dt,
                ht,
                min_chunk_size,
            )
        elif isinstance(incoming_sample, muller.core.tensor.Tensor):
            incoming_sample, shape = serialize_tensor(
                incoming_sample,
                sample_compression,
                chunk_compression,
                dt,
                ht,
                tiling_threshold,
                break_into_tiles,
                store_uncompressed_tiles,
            )
        elif isinstance(
                incoming_sample,
                (np.ndarray, list, int, float, bool, np.integer, np.floating, np.bool_),
        ):
            incoming_sample, shape = serialize_numpy_and_base_types(
                incoming_sample,
                sample_compression,
                chunk_compression,
                dt,
                ht,
                tiling_threshold,
                break_into_tiles,
                store_uncompressed_tiles,
            )
        elif isinstance(incoming_sample, SampleTiles):
            shape = incoming_sample.sample_shape
        else:
            msg = f"Cannot serialize sample of type {type(incoming_sample)}."
            if isinstance(msg, str):
                method = "read"
                msg += f"If you are appending data from a file, please pass muller.{method}(filename) " \
                       f"to the append operation, instead of the filename string."
            raise TypeError(msg)
        shape = self.convert_to_rgb(shape)
        shape = self.normalize_shape(shape)
        return incoming_sample, shape  # type: ignore

    def can_fit_sample(self, sample_nbytes, buffer_nbytes=0):
        """Can fit samples."""
        if self.num_data_bytes == 0:
            if self.tiling_threshold < 0:  # tiling disabled
                return True

            return buffer_nbytes + sample_nbytes <= self.tiling_threshold

        return (
            self.num_data_bytes + buffer_nbytes + sample_nbytes
            <= self.min_chunk_size
        )

    def get_byte_positions(self, local_index):
        """Returns the byte positions."""
        # should only be called if self.is_fixed_shape
        return local_index * self.sample_size, (local_index + 1) * self.sample_size

    def update_sample(self, local_index: int, new_sample: InputSample):
        """Updates a sample in the chunk."""

    def check_empty_before_read(self):
        """Check whether the chunk is empty. """
        if self.is_empty_tensor:
            raise EmptyTensorError(
                "This tensor has only been populated with empty samples. "
                "Need to add at least one non-empty sample before retrieving data."
            )

    def check_shape_for_update(self, shape):
        """Checks if the shape being assigned at the new index is valid."""
        if shape is None:
            return
        max_shape = self.tensor_meta.max_shape
        if max_shape:
            expected_dimensionality = len(max_shape)
            if expected_dimensionality != len(shape):
                raise TensorInvalidSampleShapeError(shape, expected_dimensionality)

    def create_updated_data(self, local_index: int, old_data, new_sample_bytes: bytes):
        """Create updated data. """
        if not old_data or self.byte_positions_encoder.is_empty():  # tiled sample
            return new_sample_bytes
        old_start_byte, old_end_byte = self.byte_positions_encoder[local_index]
        left_data = old_data[:old_start_byte]  # type: ignore
        right_data = old_data[old_end_byte:]  # type: ignore

        # preallocate
        total_new_bytes = len(left_data) + len(new_sample_bytes) + len(right_data)
        new_data = bytearray(total_new_bytes)

        # copy old data and add new data
        new_start_byte = old_start_byte
        new_end_byte = old_start_byte + len(new_sample_bytes)
        new_data[:new_start_byte] = left_data
        new_data[new_start_byte:new_end_byte] = new_sample_bytes
        new_data[new_end_byte:] = right_data
        return new_data

    def update_in_meta_and_headers(
        self, local_index: int, sample_nbytes: Optional[int], shape
    ):
        """Updates an existing sample in meta and headers"""
        if sample_nbytes is not None:
            self.byte_positions_encoder[local_index] = sample_nbytes
        if shape is not None:
            if self.shapes_encoder.is_empty():
                num_samples = self.byte_positions_encoder.num_samples
                self._fill_empty_shapes(shape, num_samples)
            self.shapes_encoder[local_index] = shape
            self.tensor_meta.update_shape_interval(shape)

    def write_tile(self, sample: SampleTiles):
        """Write the tiling data. """
        data, tile_shape = sample.yield_tile_sample()
        self.data_bytes = data
        self.register_sample_to_headers(None, tile_shape)
        if sample.is_first_write:
            self.tensor_meta.update_shape_interval(sample.sample_shape)  # type: ignore
            if self._update_tensor_meta_length:
                self.tensor_meta.update_length(1)

    def make_data_bytearray(self):
        """Copies `self.data_bytes` into a bytearray if it is a memoryview."""
        # data_bytes will be a memoryview if frombuffer is called.
        if isinstance(self.data_bytes, PartialReader):
            chunk_bytes = self.data_bytes.get_all_bytes()
            self.data_bytes = bytearray(chunk_bytes[self.header_bytes:])
        elif isinstance(self.data_bytes, memoryview):
            self.data_bytes = bytearray(self.data_bytes)

    def pop(self, index):
        """Pop samples. """
        self.prepare_for_write()
        sb, eb = self.byte_positions_encoder[index]
        self.data_bytes = self.data_bytes[:sb] + self.data_bytes[eb:]
        if not self.shapes_encoder.is_empty():
            self.shapes_encoder.pop(index)
        if not self.byte_positions_encoder.is_empty():
            self.byte_positions_encoder.pop(index)

    def pop_multiple(self, num_samples):
        """Pop multiple samples. """
        self.prepare_for_write()

        if not self.byte_positions_encoder.is_empty():
            total_samples = self.shapes_encoder.num_samples
            starting_byte_first_popped_sample = self.byte_positions_encoder[
                total_samples - num_samples
                ][0]
            self.data_bytes = self.data_bytes[:starting_byte_first_popped_sample]

        for _ in range(num_samples):
            if not self.shapes_encoder.is_empty():
                self.shapes_encoder.pop()
            if not self.byte_positions_encoder.is_empty():
                self.byte_positions_encoder.pop()

    def _get_partial_sample_tile(self, is_bytes=False):
        if (
                not isinstance(self.data_bytes, PartialReader)
                and not self.data_bytes
                and len(self.shapes_encoder.encoded) > 0
        ):
            shape = self.shapes_encoder.encoded[0][:-1]
            if len(shape) and np.all(shape):

                if is_bytes:
                    total_elements = int(np.prod(np.array(shape, dtype=np.uint64)))
                    bytes_per_element = np.dtype(self.dtype).itemsize
                    total_bytes = total_elements * bytes_per_element
                    return b"0" * total_bytes

                return np.zeros(shape, dtype=self.dtype)

        return None

    def _fill_empty_shapes(self, shape, num_samples):
        dims = len(shape)
        self.num_dims = self.num_dims or dims
        if num_samples > 0:
            empty_shape = (0,) * dims
            self.shapes_encoder.register_samples(empty_shape, num_samples)
            self.tensor_meta.update_shape_interval(empty_shape)




def catch_chunk_read_error(fn):
    """Catch the chunk read error. """
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except EmptyTensorError:
            raise
        except Exception as e:
            raise ReadSampleFromChunkError(self.key) from e

    return wrapper


def _infer_chunk_num_bytes(
        version: str,
        shape_info: np.ndarray,
        byte_positions: np.ndarray,
        data: Optional[Union[Sequence[bytes], Sequence[memoryview]]] = None,
        len_data: Optional[int] = None,
) -> int:
    """
    Calculates the number of bytes in a chunk without serializing it.
    Used by `LRUCache` to determine if a chunk can be cached.

    Args:
        version: (str) Version of muller library
        shape_info: (numpy.ndarray) Encoded shapes info from the chunk's `ShapeEncoder` instance.
        byte_positions: (numpy.ndarray) Encoded byte positions from the chunk's `BytePositionsEncoder` instance.
        data: (list) `_data` field of the chunk
        len_data: (int, optional) Number of bytes in the chunk

    Returns:
        Length of the chunk when serialized as int
    """
    # NOTE: Assumption: version string contains ascii characters only (ord(c) < 128)
    # NOTE: Assumption: len(version) < 256
    if len_data is None:
        len_data = sum(map(len, data))  # type: ignore

    header_size = infer_header_num_bytes(version, shape_info, byte_positions)
    return header_size + len_data


def infer_header_num_bytes(
        version: str, shape_info: np.ndarray, byte_positions: np.ndarray
):
    """
    Calculates the number of header bytes in a chunk without serializing it.

    Args:
        version: (str) Version of muller library
        shape_info: (numpy.ndarray) Encoded shapes info from the chunk's `ShapeEncoder` instance.
        byte_positions: (numpy.ndarray) Encoded byte positions from the chunk's `BytePositionsEncoder` instance.

    Returns:
        Length of the headers of chunk when serialized as int.
    """
    return len(version) + shape_info.nbytes + byte_positions.nbytes + 13

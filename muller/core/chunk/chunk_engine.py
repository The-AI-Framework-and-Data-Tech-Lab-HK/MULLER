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

import logging
import threading
from collections import OrderedDict
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Union,
    List,
    Tuple,
    Sequence
)

import numpy as np
from tqdm import tqdm

import muller
from muller.api.info import Info
from muller.compression import (
    VIDEO_COMPRESSIONS, BYTE_COMPRESSION,
)
from muller.constants import (
    FAST_EXTEND_BAIL,
    DEFAULT_MAX_CHUNK_SIZE,
    FIRST_COMMIT_ID,
    PARTIAL_NUM_SAMPLES,
    DEFAULT_TILING_THRESHOLD,
    WRITE_TILES_INDEX, MAX_WORKERS_FOR_CHUNK_ENGINE
)
from muller.core.chunk.base_chunk import BaseChunk, InputSample
from muller.core.chunk.chunk_compressed_chunk import ChunkCompressedChunk
from muller.core.chunk.sample_compressed_chunk import SampleCompressedChunk
from muller.core.chunk.uncompressed_chunk import UncompressedChunk
from muller.core.compression import get_compression_type
from muller.core.fast_forwarding import ffw_chunk_id_encoder
from muller.core.index.index import Index
from muller.core.meta.encode.base_encoder import LAST_SEEN_INDEX_COLUMN
from muller.core.meta.encode.chunk_id import CHUNK_ID_COLUMN, ChunkIdEncoder
from muller.core.meta.encode.sequence import SequenceEncoder
from muller.core.meta.encode.tile import TileEncoder
from muller.core.meta.tensor_meta import TensorMeta, _validate_required_htype_overwrites
from muller.core.partial_reader import PartialReader
from muller.core.sample import Sample
from muller.core.serialize import HEADER_SIZE_BYTES, deserialize_chunkids
from muller.core.storage import MemoryProvider, RomaProvider
from muller.core.storage.lru_cache import LRUCache
from muller.core.storage.lru_cache import _get_nbytes  # Sherry: not elegant
from muller.core.storage.provider import StorageProvider
from muller.core.tiling.deserialize import coalesce_tiles, translate_slices, combine_chunks
from muller.core.version_control.commit_chunk_map import CommitChunkMap
from muller.core.version_control.commit_diff import CommitDiff
from muller.util.casting import get_empty_text_like_sample
from muller.util.casting import get_htype, get_dtype
from muller.util.class_label import convert_to_hash, convert_to_idx
from muller.util.exceptions import CorruptedMetaError
from muller.util.exceptions import GetChunkError, DynamicTensorNumpyError
from muller.util.exceptions import ReadOnlyModeError, ReadSampleFromChunkError
from muller.util.exceptions import SampleHtypeMismatchError
from muller.util.image import convert_sample, convert_img_arr
from muller.util.keys import get_tensor_commit_chunk_map_key, get_tensor_tile_encoder_key, get_sequence_encoder_key
from muller.util.keys import get_tensor_commit_diff_key
from muller.util.keys import get_tensor_meta_key, get_chunk_id_encoder_key, get_chunk_key
from muller.util.remove_cache import get_base_storage
from muller.util.shape_interval import ShapeInterval
from muller.util.version_control import get_dataset_diff_at_commit


class ChunkEngine:
    def __init__(
            self,
            key: str,
            cache: LRUCache,
            version_state: Dict[str, Any],  # record the current version state
            split_tensor_meta: bool,
            meta_cache: Optional[LRUCache] = None
    ):
        self.key = key
        self.cache = cache
        self.base_storage = get_base_storage(cache)
        self._meta_cache = meta_cache
        self.version_state = version_state
        self.split_tensor_meta = split_tensor_meta
        self.name = version_state["tensor_names"].get(self.key)
        self.compression = None
        self.chunk_class = BaseChunk

        self._tensor_meta: Optional[TensorMeta] = None
        self._tensor_meta_commit_id: Optional[str] = None

        self._chunk_id_encoder: Optional[ChunkIdEncoder] = None
        self._chunk_id_encoder_commit_id: Optional[str] = None

        self._sequence_encoder: Optional[SequenceEncoder] = None
        self._sequence_encoder_commit_id: Optional[str] = None

        self._tile_encoder: Optional[TileEncoder] = None
        self._tile_encoder_commit_id: Optional[str] = None

        self._commit_chunk_map: Optional[CommitChunkMap] = None
        self._commit_chunk_map_commit_id: Optional[str] = None

        self._commit_diff: Optional[CommitDiff] = None
        self._commit_diff_commit_id: Optional[str] = None

        self._active_appended_chunk: Optional[BaseChunk] = None
        self._active_updated_chunk: Optional[BaseChunk] = None

        self._info: Optional[Info] = None
        self._info_commit_id: Optional[str] = None

        self._all_chunk_engines: Optional[Dict[str, ChunkEngine]] = None
        self._is_temp_label_tensor: bool = False
        self._hash_label_map: Dict[int, str] = OrderedDict()
        self._sample_compression = None
        self._chunk_compression = None

        tensor_meta = self.tensor_meta
        self.name = tensor_meta.name or self.key
        numpy_extend_optimization_enabled = False

        if tensor_meta.sample_compression:
            self._sample_compression = self.compression = tensor_meta.sample_compression
            self.chunk_class = SampleCompressedChunk

        elif tensor_meta.chunk_compression:
            self._chunk_compression = self.compression = tensor_meta.chunk_compression
            self.chunk_class = ChunkCompressedChunk
            if get_compression_type(tensor_meta.chunk_compression) == BYTE_COMPRESSION:
                numpy_extend_optimization_enabled = True
        else:
            self.chunk_class = UncompressedChunk
            numpy_extend_optimization_enabled = True

        self._numpy_extend_optimization_enabled = numpy_extend_optimization_enabled

        self.cache_enabled = True
        self.cached_data: Optional[np.ndarray] = None
        self.cache_range: range = range(0)

        self._chunk_args = None
        self._num_samples_per_chunk: Optional[int] = None
        self.write_initialization_done = False
        self.start_chunk = None

    @property
    def commit_id(self):
        """Returns the commit id. """
        return self.version_state["commit_id"]

    @property
    def meta_cache(self) -> LRUCache:
        """Returns the meta cache. """
        return self._meta_cache or self.cache

    @property
    def tensor_meta(self): # return meta for this tensor only
        """Returns the tensor meta. """
        commit_id = self.commit_id
        if self._tensor_meta is None or self._tensor_meta_commit_id != commit_id:
            if self.split_tensor_meta:
                key = get_tensor_meta_key(self.key, commit_id)
                self._tensor_meta = self.meta_cache.get_muller_object(key, TensorMeta)
            else:
                key = get_tensor_meta_key("", commit_id) # tensor_meta.json
                self._tensor_meta = self.meta_cache.get_muller_object(key, dict)  # one meta for all tensors
            self._tensor_meta_commit_id = commit_id
            self.meta_cache.register_muller_object(key, self._tensor_meta)
        return self._tensor_meta if self.split_tensor_meta else self._tensor_meta[self.key]

    @property
    def is_sequence(self):
        """Returns whether the tensor is a sequence. """
        return self.tensor_meta.is_sequence

    @property
    def tensor_length(self) -> int:
        """Length of primary axis of tensor (does not include samples in sequences). """
        # Sherry: need to figure out the function of self._sequence_length
        return self.sequence_length or self.tensor_meta.length  # return self.tensor_meta.length

    @property
    def sequence_length(self):
        """Return the sequence length. """
        if self.is_sequence:
            return self.sequence_encoder.num_samples
        return 0

    @property
    def chunk_id_encoder_exists(self) -> bool:
        """Returns whether the chunk id encoder exists. """
        commit_id = self.commit_id
        if (
                self._chunk_id_encoder is not None
                and self._chunk_id_encoder_commit_id == commit_id
        ):
            return True
        try:
            key = get_chunk_id_encoder_key(self.key, commit_id)
            _ = self.meta_cache[key]
            return True
        except KeyError:
            return False

    @property
    def chunk_id_encoder(self) -> ChunkIdEncoder:
        """Gets the chunk id encoder from cache, if one is not found it creates a blank encoder.

        Raises:
            CorruptedMetaError: If chunk id encoding was corrupted.

        Returns:
            ChunkIdEncoder: The chunk ID encoder handles the mapping between sample indices
                and their corresponding chunks.
        """
        commit_id = self.commit_id
        if (
                self._chunk_id_encoder is None
                or self._chunk_id_encoder_commit_id != commit_id
        ):
            commit_id = self.commit_id
            key = get_chunk_id_encoder_key(self.key, commit_id)  # The name of the unsharded files
            if not self.chunk_id_encoder_exists:
                enc = ChunkIdEncoder(dtype=np.uint64)
                try:
                    self.meta_cache[key] = enc
                except ReadOnlyModeError:
                    pass
            else:
                enc = self.meta_cache.get_muller_object(key=key, expected_class=ChunkIdEncoder)

            self._chunk_id_encoder = enc
            self._chunk_id_encoder_commit_id = commit_id
            self.meta_cache.register_muller_object(key, enc)
        return self._chunk_id_encoder

    @property
    def num_samples(self) -> int:
        """Total length of tensor (includes samples in sequences)
        Ignores any applied indexing and returns the total length.
        """
        return self.tensor_meta.length

    @property
    def chunk_compression(self):
        """Returns the chunk compression. """
        return self._chunk_compression

    @property
    def is_data_cachable(self):
        """Returns whether the tensor meta is cached. """
        if self.cache_enabled:
            tensor_meta = self.tensor_meta
            return (
                    self.chunk_class == UncompressedChunk
                    and tensor_meta.htype not in ["text", "json", "list", "polygon"]
                    and tensor_meta.max_shape
                    and (tensor_meta.max_shape == tensor_meta.min_shape)
                    and (np.prod(tensor_meta.max_shape) < 20)
            )
        return False

    @property
    def max_chunk_size(self):
        """Returns the maximum chunk size of the tensor. """
        # no chunks may exceed this
        return (
                getattr(self.tensor_meta, "max_chunk_size", None) or DEFAULT_MAX_CHUNK_SIZE
        )

    @property
    def min_chunk_size(self):
        """Returns the minimum chunk size of the tensor. """
        # only the last chunk may be less than this
        return self.max_chunk_size // 2

    @property
    def tiling_threshold(self):
        """Returns the tiling threshold of the tensor. """
        return (
                getattr(self.tensor_meta, "tiling_threshold", None)
                or DEFAULT_TILING_THRESHOLD
                or self.min_chunk_size
        )

    @property
    def chunk_args(self):
        """Returns the chunk arguments. """
        if self._chunk_args is None:
            self._chunk_args = [
                self.min_chunk_size,
                self.max_chunk_size,
                self.tiling_threshold,
                self.tensor_meta,
                self.compression,
            ]
        return self._chunk_args

    @property
    def commit_diff(self) -> CommitDiff:
        """Returns the commit diff of the tensor. """
        commit_id = self.commit_id
        if self._commit_diff is None or self._commit_diff_commit_id != commit_id:
            key = get_tensor_commit_diff_key(self.key, commit_id)
            if not self.commit_diff_exists:
                diff = CommitDiff(self.num_samples, commit_id=commit_id, storage=self.cache)
                try:
                    self.meta_cache[key] = diff
                except ReadOnlyModeError:
                    pass
            else:
                diff = self.meta_cache.get_muller_object(key, CommitDiff)
            self._commit_diff = diff
            self._commit_diff_commit_id = commit_id
            self.meta_cache.register_muller_object(key, diff)
        return self._commit_diff

    @property
    def commit_diff_exists(self) -> bool:
        """Returns whether the commit diff exists. """
        commit_id = self.commit_id
        if self._commit_diff is not None and self._commit_diff_commit_id == commit_id:
            return True
        try:
            key = get_tensor_commit_diff_key(self.key, commit_id)
            _ = self.meta_cache[key]
            return True
        except KeyError:
            return False

    @property
    def last_chunk_key(self) -> str:
        """Returns the last chunk key. """
        last_chunk_name = self.last_appended_chunk_name
        return get_chunk_key(self.key, last_chunk_name)

    @property
    def active_appended_chunk(self):
        """Returns the active appended chunk. """
        return self._active_appended_chunk

    @property
    def commit_chunk_map(self) -> Optional[CommitChunkMap]:
        """Gets the commit chunk map from cache, if one is not found it creates a blank one.

        Returns:
            Optional[CommitChunkMap]: The commit chunk map keeps track of all the chunks present in the current commit,
            returns None for the first commit.
        """
        commit_id = self.commit_id
        if commit_id == FIRST_COMMIT_ID:
            # the first commit doesn't need a commit chunk map
            return None
        if (
                self._commit_chunk_map is None
                or self._commit_chunk_map_commit_id != commit_id
        ):
            key = get_tensor_commit_chunk_map_key(self.key, commit_id)
            if not self.commit_chunk_map_exists:
                cmap = CommitChunkMap()
                try:
                    self.meta_cache[key] = cmap
                except ReadOnlyModeError:
                    pass
            else:
                cmap = self.meta_cache.get_muller_object(key, CommitChunkMap)
            self._commit_chunk_map = cmap
            self._commit_chunk_map_commit_id = commit_id
            self.meta_cache.register_muller_object(key, cmap)
        return self._commit_chunk_map

    @commit_chunk_map.setter
    def commit_chunk_map(self, value):
        """Set the value of commit chunk map. """
        self._commit_chunk_map = value

    @property
    def commit_chunk_map_exists(self) -> bool:
        """Checks if the commit chunk map exists for the given tensor in the current commit."""
        commit_id = self.commit_id
        if (
                self._commit_chunk_map is not None
                and self._commit_chunk_map_commit_id == commit_id
        ):
            return True

        try:
            key = get_tensor_commit_chunk_map_key(self.key, commit_id)
            _ = self.meta_cache[key]
            return True
        except KeyError:
            return False

    @property
    def last_appended_chunk_name(self) -> str:
        """Returns the last appended chunk name. """
        return self.chunk_id_encoder.get_name_for_chunk(-1)

    @property
    def last_appended_chunk_id(self) -> str:
        """Returns the last appended chunk id. """
        return self.chunk_id_encoder.get_id_for_chunk(-1)

    @property
    def num_chunks(self) -> int:
        """Returns the number of chunks. """
        if not self.chunk_id_encoder_exists:
            return 0
        return self.chunk_id_encoder.num_chunks

    @property
    def tile_encoder_exists(self) -> bool:
        """Returns whether the tile encoder exists. """
        commit_id = self.commit_id
        if self._tile_encoder is not None and self._tile_encoder_commit_id == commit_id:
            return True

        try:
            key = get_tensor_tile_encoder_key(self.key, commit_id)
            _ = self.meta_cache[key]
            return True
        except KeyError:
            return False

    @property
    def enable_tile_encoder(self) -> bool:  # Sherry: multi-model?
        """Returns whether the tile encoder is enabled. """
        return muller.constants.TILE_ENCODER_ENABLED

    @property
    def tile_encoder(self) -> Optional[TileEncoder]:
        """Gets the tile encoder from cache, if one is not found it creates a blank encoder."""
        if self._tile_encoder is None or self._tile_encoder_commit_id != self.commit_id:
            key = get_tensor_tile_encoder_key(self.key, self.commit_id)
            if not self.tile_encoder_exists:
                if muller.constants.TILE_ENCODER_ENABLED:
                    enc = TileEncoder()
                    if muller.constants.WRITE_TILES_INDEX:
                        try:
                            self.meta_cache[key] = enc
                        except ReadOnlyModeError:
                            pass
                else:
                    return None
            else:
                enc = self.meta_cache.get_muller_object(key, TileEncoder)
            self._tile_encoder = enc
            self._tile_encoder_commit_id = self.commit_id
            self.meta_cache.register_muller_object(key, enc)
        return self._tile_encoder

    @property
    def get_tile_encoder(self):
        """Returns the tile encoder (without processing)"""
        return self._tile_encoder

    @property
    def sequence_encoder_exists(self) -> bool:
        """Returns whether the sequence encoder exists. """
        commit_id = self.commit_id
        if (
                self._sequence_encoder is not None
                and self._sequence_encoder_commit_id == commit_id
        ):
            return True
        try:
            key = get_sequence_encoder_key(self.key, commit_id)
            _ = self.meta_cache[key]
            return True
        except KeyError:
            return False

    @property
    def sequence_encoder(self) -> Optional[SequenceEncoder]:
        """Returns whether the sequence encoder is enabled. """
        if not self.is_sequence:
            return None  # type: ignore
        commit_id = self.commit_id
        if (
                self._sequence_encoder is None
                or self._sequence_encoder_commit_id != commit_id
        ):
            commit_id = self.commit_id
            key = get_sequence_encoder_key(self.key, commit_id)
            if not self.sequence_encoder_exists:
                enc = SequenceEncoder()
                try:
                    self.meta_cache[key] = enc
                except ReadOnlyModeError:
                    pass
            else:
                enc = self.meta_cache.get_muller_object(key, SequenceEncoder)
            self._sequence_encoder = enc
            self._sequence_encoder_commit_id = commit_id
            self.meta_cache.register_muller_object(key, enc)
        return self._sequence_encoder

    @property
    def sequence_item_length(self):
        """Returns the length of sequence item. """
        enc = self.sequence_encoder
        nrows = len(enc.encoded)
        if nrows == 0:
            return 0
        if nrows == 1:
            s, e = enc[0]
            return e - s
        return None

    @property
    def _sequence_item_length_range(self):
        """Returns minimum and maximum length of items in a sequence. """
        enc = self.sequence_encoder
        nrows = len(enc.encoded)
        if nrows == 0:
            return 0, 0
        min_ = max_ = enc[0][1] - enc[0][0]
        # sequence length is number of samples in tensor
        for i in range(1, self.sequence_length):
            length = enc[i][1] - enc[i][0]
            if length < min_:
                min_ = length
            elif length > max_:
                max_ = length
        return min_, max_

    @property
    def is_video(self):
        """Returns whether this is a video. """
        return (
                self.compression in VIDEO_COMPRESSIONS or self.tensor_meta.htype == "video"
        )

    @property
    def is_fixed_shape(self):
        """returns whether the shape is fixed. """
        return (
                self.tensor_meta.min_shape == self.tensor_meta.max_shape
                and not self.is_text_like
        )

    @property
    def active_updated_chunk(self):
        """Returns active updated chunk. """
        return self._active_updated_chunk

    @property
    def is_text_like(self):
        """Returns whether this is a text like chunk. """
        return (
                self.tensor_meta.htype in {"text", "json", "list", "tag"}
        )

    @property
    def sample_compression(self):
        """Returns whether this is a sample compression chunk. """
        return self._sample_compression

    @property
    def all_chunk_engine(self):
        """Returns whether this is a complete chunk engine. """
        return self._all_chunk_engines

    @property
    def is_temp_label_tensor(self):
        """Returns whether this is a temp label tensor. """
        return self._is_temp_label_tensor

    @property
    def hash_label_map(self):
        """Returns whether this is a hash label map. """
        return self._hash_label_map

    @active_updated_chunk.setter
    def active_updated_chunk(self, value):
        """Set the active updated chunk. """
        if self.active_updated_chunk is not None:
            self.cache.remove_muller_object(self.active_updated_chunk.key)
        self._active_updated_chunk = value
        if value is not None:
            self.cache.register_muller_object(value.key, value)

    @active_appended_chunk.setter
    def active_appended_chunk(self, value):
        """Set the active appended chunk. """
        if self.active_appended_chunk is not None:
            self.cache.remove_muller_object(self.active_appended_chunk.key)
        self._active_appended_chunk = value
        if value is not None:
            self.cache.register_muller_object(value.key, value)

    @property
    def info(self):
        """Returns information about this chunk engine. """
        return self._info

    @info.setter
    def info(self, value):
        """Set the info."""
        self._info = value

    @property
    def info_commit_id(self):
        """Returns information (commit id) about this chunk engine. """
        return self._info_commit_id

    @info_commit_id.setter
    def info_commit_id(self, value):
        """Set the info of commit id."""
        self._info_commit_id = value

    @staticmethod
    def _handle_one_or_more_samples(
            enc: ChunkIdEncoder,
            register,
            samples,
            num_samples_added,
            updated_chunks,
            start_chunk_row,
            current_chunk,
            enc_count,
            lengths,
    ):
        if not register and not updated_chunks:
            updated_chunks.append(current_chunk)
        num_samples_added = int(num_samples_added)
        if register:
            if start_chunk_row is not None:
                enc.register_samples(num_samples_added, start_chunk_row)
            else:
                enc_count[-1] += num_samples_added
        if lengths is not None:
            lengths = lengths[num_samples_added:]
        samples = samples[num_samples_added:]
        return num_samples_added, samples, lengths

    @staticmethod
    def _handle_tiled_sample(
            enc: ChunkIdEncoder,
            register,
            samples,
            orig_meta_length,
            incoming_num_samples,
            start_chunk_row,
            enc_count,
            tiles,
            lengths,
    ):
        sample = samples[0]
        if sample.is_first_write:
            if register:
                if start_chunk_row is not None:
                    enc.register_samples(1)
                else:
                    enc_count[-1] += 1
        if sample.is_last_write:
            tiles[
                incoming_num_samples - len(samples) + bool(register) * orig_meta_length
                ] = (
                sample.sample_shape,
                sample.tile_shape,
            )
            samples = samples[1:]
            if lengths is not None:
                lengths = lengths[1:]
            num_samples_added = 1
        else:
            num_samples_added = 0
        return num_samples_added, samples, lengths

    @staticmethod
    def _process_image_samples(tensor_meta, samples, ignore_errors):
        mode = "L" if tensor_meta.htype == "image.gray" else "RGB"
        converted = []
        for sample in samples:
            try:
                if isinstance(sample, Sample):
                    converted.append(convert_sample(sample, mode))
                elif isinstance(sample, np.ndarray):
                    converted.append(convert_img_arr(sample, mode))
                else:
                    raise SampleHtypeMismatchError(tensor_meta.htype, type(sample))
            except Exception:
                if ignore_errors:
                    continue
                raise
        samples = converted
        return samples

    @staticmethod
    def _get_num_sample_added(current_chunk, current_chunk_full, samples, update_tensor_meta,
                              incoming_num_samples, ignore_errors, extra_args):
        if current_chunk_full:
            num_samples_added = 0
            current_chunk_full = False
        else:
            initial_num_samples = len(samples)
            num_samples_added = current_chunk.extend_if_has_space(
                list(samples), update_tensor_meta=update_tensor_meta,
                ignore_errors=ignore_errors, **extra_args  # type: ignore
            )  # type: ignore
            skipped_num_samples = initial_num_samples - len(samples)
            incoming_num_samples -= skipped_num_samples
        return num_samples_added, incoming_num_samples, current_chunk_full

    @staticmethod
    def _check_samples_type(samples):
        if not isinstance(samples, (List, np.ndarray)):
            raise TypeError(f"Cannot extend with samples of type {type(samples)}")

    def update(
            self,
            index: Index,
            samples: Union[np.ndarray, Sequence[InputSample], InputSample],
            operator: Optional[str] = None,
    ):
        """Update the chunks"""
        muller.core.chunk.update(self, index, samples, operator)

    def extend(
            self,
            samples,
            progressbar: bool = False,
            pg_callback=None,
            ignore_errors: bool = False,
            is_uuid: bool = False
    ):
        """Extend the chunks"""
        muller.core.chunk.extend(self, samples, progressbar, pg_callback, ignore_errors)

    def pop(
            self,
            indices: Optional[Union[int, List[int]]] = None,
            sample_id_tensor=None,
            rechunk: Optional[bool] = False,
    ):
        """Pop from the chunk."""
        muller.core.chunk.pop(self, indices, sample_id_tensor, rechunk)

    def write_chunk_to_storage(self, chunk):
        """Dump the chunk to storage."""
        if chunk is None or not chunk.is_dirty:
            return
        storage = self.cache
        key = chunk.key
        storage[key] = chunk
        chunk.is_dirty = False

    def samples_to_chunks(
            self,
            samples,
            start_chunk: Optional[BaseChunk] = None,
            start_chunk_row: Optional[int] = None,
            return_samples: bool = False,
            ignore_errors: bool = False,
            is_uuid: bool = False,
            **kwargs
    ):
        """Samples to chunks"""

        extending = start_chunk_row is None and kwargs.get("register", True)
        incoming_num_samples = len(samples)
        enc_count = [0]
        lengths = self._obtain_lengths(extending, samples, is_uuid)
        current_chunk = start_chunk
        current_chunk, updated_chunks, enc_ids = self._obtain_current_chunk(current_chunk,
                                                                            kwargs.get("register", True),
                                                                            start_chunk_row,
                                                                            extending)
        tiles: Dict[int, Tuple[Tuple[int, ...], Tuple[int, ...]]] = {}
        verified_samples = []
        current_chunk_full = False

        if kwargs.get("register", True) and kwargs.get("update_commit_diff", False):
            commit_diff = self.commit_diff
        if kwargs.get("progressbar", False):
            pbar = tqdm(total=len(samples))
        if not isinstance(samples, list) and not (
                isinstance(samples, np.ndarray) and self._numpy_extend_optimization_enabled
        ):
            # Note: in the future we can get rid of this conversion of sample compressed chunks too
            # by predicting the compression ratio.
            samples = list(samples)

        while len(samples) > 0:
            (num_samples_added, incoming_num_samples,
             current_chunk_full) = self._get_num_sample_added(current_chunk,
                                                              current_chunk_full,
                                                              samples,
                                                              kwargs.get("update_tensor_meta", True),
                                                              incoming_num_samples,
                                                              ignore_errors,
                                                              {"lengths": lengths, "is_uuid": is_uuid})
            if num_samples_added == 0:
                current_chunk, updated_chunks, enc_ids, enc_count, start_chunk_row = (
                    self._update_current_chunk(kwargs.get("register", True),
                                               start_chunk_row,
                                               enc_ids,
                                               enc_count,
                                               updated_chunks))

            elif num_samples_added == PARTIAL_NUM_SAMPLES:
                if samples[0].is_first_write:
                    verified_samples.append(samples[0])
                num_samples_added, samples, lengths = self._handle_tiled_sample(
                    self.chunk_id_encoder,
                    kwargs.get("register", True),
                    samples,
                    self.tensor_meta.length,
                    incoming_num_samples,
                    start_chunk_row,
                    enc_count,
                    tiles,
                    lengths,
                )
                if len(samples) > 0:
                    current_chunk, updated_chunks, enc_ids, enc_count, start_chunk_row = (
                        self._update_current_chunk(kwargs.get("register", True),
                                                   start_chunk_row,
                                                   enc_ids,
                                                   enc_count,
                                                   updated_chunks))
            elif num_samples_added == FAST_EXTEND_BAIL:
                num_samples_added = 0
                samples = list(samples)
            else:
                current_chunk_full = True
                verified_samples.extend(samples[:int(num_samples_added)])
                num_samples_added, samples, lengths = self._handle_one_or_more_samples(
                    self.chunk_id_encoder,
                    kwargs.get("register", True),
                    samples,
                    num_samples_added,
                    updated_chunks,
                    start_chunk_row,
                    current_chunk,
                    enc_count,
                    lengths,
                )
            if kwargs.get("progressbar", False):
                pbar.update(num_samples_added)
            elif kwargs.get("pg_callback", None):
                kwargs.get("pg_callback")(num_samples_added)

        if extending:
            self._update_enc_ids_and_count(self.chunk_id_encoder,
                                           enc_ids,
                                           enc_count,
                                           incoming_num_samples)
        if kwargs.get("register", True):
            if kwargs.get("update_commit_diff", False):
                commit_diff.add_data(incoming_num_samples)
        if self.enable_tile_encoder:
            self.tile_encoder.entries.update(tiles)
            if WRITE_TILES_INDEX:
                self.tile_encoder.is_dirty = True
        if kwargs.get("progressbar", False):
            pbar.close()

        if return_samples:
            return verified_samples

        if not kwargs.get("register", True):
            return updated_chunks, tiles


    def copy_chunk(self, chunk, row=None):
        # 重新copy一份chunk，但是现在把所有的chunk文件都存在最外层的tensor下面了，因此不能采用原来的方案copy同名chunk，
        # 需要生成新的chunk_id和chunk_name，同时需要修改该commit下面的unsharded文件，使之与新的chunk_id对应。
        """Copies the chunk to the current commit.

        Returns the copied chunk.
        """
        enc = self.chunk_id_encoder
        new_chunk_id = enc.generate_chunk_id(register=False)
        get_right_row = True
        if row is not None and enc.encoded[row][0] != int(chunk.id):
            get_right_row = False
        if row is None or get_right_row is False:
            row = self.chunk_id_encoder.get_row_from_id(int(chunk.id))
        self.chunk_id_encoder.update_chunk_id(row, new_chunk_id)
        new_chunk_name = ChunkIdEncoder.name_from_id(new_chunk_id)
        new_chunk_key = get_chunk_key(self.key, new_chunk_name)
        chunk = chunk.copy(self.chunk_args)
        chunk.key = new_chunk_key
        chunk.id = new_chunk_id
        if self.commit_chunk_map is not None:
            self.commit_chunk_map.add(new_chunk_name)
        return chunk

    def chunk_in_target_commit(self, chunk_name, commit_id):
        """return whether the chunk is in target commit"""
        key = self.key
        chunk_map_key = get_tensor_commit_chunk_map_key(key, commit_id)
        if commit_id == FIRST_COMMIT_ID:
            chunk_index_path = get_chunk_id_encoder_key(self.key, commit_id)
            all_chunk_names = self._get_chunk_names_from_path(chunk_index_path)
            return chunk_name in all_chunk_names
        if commit_id == self.commit_id:
            v = self.commit_chunk_map.chunks.get(chunk_name)
        else:
            try:
                chunk_map = self.meta_cache.get_muller_object(chunk_map_key, CommitChunkMap).chunks
            except Exception:
                commit_chunk_map = CommitChunkMap()
                try:
                    self.meta_cache[chunk_map_key] = commit_chunk_map
                except ReadOnlyModeError:
                    self.meta_cache.muller_objects[chunk_map_key] = commit_chunk_map
                chunk_map = dict()
            v = chunk_map.get(chunk_name)
        return v is not None

    def shape(
            self,
            index: Index,
            sample_shape_provider: Optional[Callable] = None,
    ) -> Tuple[Optional[int], ...]:
        """Returns the shape. """
        return muller.core.chunk.shape(self, index, sample_shape_provider)

    def ndim(self, index: Optional[Index] = None) -> int:
        """Returns the number of dimensions."""
        ndim = len(self.tensor_meta.min_shape) + 1
        if self.is_sequence:
            ndim += 1
        if index:
            for idx in index.values:
                if not idx.subscriptable():
                    ndim -= 1
        return ndim

    def arrow(
            self,
            index: Index,
            aslist: bool = False,
            use_data_cache: bool = True,
            fetch_chunks: bool = False,
        ):
        """Returns as arrow."""
        return muller.core.chunk.arrow(self, index, aslist, use_data_cache,
                                                                      fetch_chunks)

    def numpy(
            self,
            index: Index,
            aslist: bool = False,
            index_list: List = None,
            **kwargs,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Returns as numpy. """
        return muller.core.chunk.numpy(self, index, aslist, index_list, **kwargs)

    def sanitize_samples(self, samples, ignore_errors=False):
        """Sanitize samples."""
        self._check_samples_type(samples)
        samples = [
            None
            if isinstance(s, list) and len(s) == 0
               or (isinstance(s, muller.core.tensor.Tensor) and s.is_empty_tensor)
            else s
            for s in samples
        ]
        tensor_meta = self.tensor_meta
        all_empty = all(sample is None for sample in samples)
        if tensor_meta.htype is None and not all_empty:
            htype = get_htype(samples)
            if tensor_meta.dtype is not None:
                _validate_required_htype_overwrites(
                    htype,
                    {
                        "sample_compression": tensor_meta.sample_compression,
                        "chunk_compression": tensor_meta.chunk_compression,
                        "dtype": tensor_meta.dtype,
                    },
                )
            tensor_meta.set_htype(htype)
        if tensor_meta.dtype is None and not all_empty:
            non_empty_samples = list(filter(lambda x: x is not None, samples))
            for sample in non_empty_samples:
                try:
                    dtype = get_dtype(sample)
                    break
                except Exception:
                    continue
            else:
                if not ignore_errors:
                    raise ValueError("Could not determine dtype of samples")
            tensor_meta.set_dtype(dtype)
        if self._is_temp_label_tensor:
            samples = convert_to_hash(samples, self._hash_label_map)
        elif tensor_meta.htype in ("image.gray", "image.rgb"):
            samples = self._process_image_samples(tensor_meta, samples, ignore_errors)
        elif tensor_meta.htype == "class_label":
            samples = self.convert_class_labels(samples)
        elif tensor_meta.htype == "tag":
            samples = [
                sample if isinstance(sample, list) else [sample]
                for sample in samples
            ]
        return samples

    def pad_and_append(
        self,
        num_samples_to_pad: int,
        value,
    ):
        """Pads the tensor with empty samples and appends value at the end."""
        muller.core.chunk.pad_and_append(self, num_samples_to_pad, value)

    def get_fixed_shape_chunk_id(self, file_group_offset: int, enc_array: np.ndarray, sample_size: int):
        """get the chunk id when this tensor is fixed_shape"""
        row = int(((file_group_offset + 1) * sample_size) / self.min_chunk_size)
        row = row if row < len(enc_array) else len(enc_array) - 1
        while enc_array[row][1] < file_group_offset:
            row += 1
        while not ((row > 0 and enc_array[row - 1][1] < file_group_offset) or row == 0):
            row -= 1
        return row, enc_array[row][0]

    def compute_offset_in_chunk(self, file_group_offset: int, chunk_index: int, sample_size: int):
        """compute the start byte and end byte in chunk"""
        local_sample_index = self.translate_to_local_index(file_group_offset, chunk_index)
        sb = local_sample_index * sample_size
        return sb, sb + sample_size

    def get_single_sample(
            self,
            global_sample_index,
            index,
            fetch_chunks=False,
            decompress=True,
    ):
        """Get a single sample."""
        if not self.is_tiled_sample(global_sample_index):
            if self.is_fixed_shape and self.chunk_class == UncompressedChunk:
                sample = self._get_fixed_shape_sample(global_sample_index, decompress)
            else:
                sample = self.get_non_tiled_sample(
                    global_sample_index,
                    index,
                    fetch_chunks=fetch_chunks,
                    decompress=decompress,
                )
        elif len(index.values) == 1:
            sample = self.get_full_tiled_sample(global_sample_index)
        else:
            sample = self.get_partial_tiled_sample(global_sample_index, index)

        return sample

    def translate_to_local_index(self, global_sample_index: int, row: int):
        """Translate global sample index to local index relative to chunks without another encoder lookup."""
        if row == 0:
            return global_sample_index
        return global_sample_index - (
                self.chunk_id_encoder.array[row - 1][-1].item() + 1
        )

    def read_basic_sample_from_chunk(
            self,
            chunk_id: int,
            local_sample_index: int,
            index: Index,
            worst_case_header_size: int = 0,
            is_tile: bool = False,
            decompress: bool = True,
    ):
        """Read basic sample from chunk. """
        chunk = self.get_chunk_from_chunk_id(
            chunk_id, partial_chunk_bytes=worst_case_header_size
        )
        decompress = decompress or (
                isinstance(chunk, ChunkCompressedChunk) or len(index) > 1
        )
        ret = chunk.read_sample(
            local_sample_index,
            cast=self.tensor_meta.htype != "dicom",
            is_tile=is_tile,
            decompress=decompress,
        )
        if len(index) > 1:
            ret = ret[tuple(entry.value for entry in index.values[1:])]
        return ret

    def read_multiple_basic_samples_from_chunk(self,
                                               chunk_ids: List[int],
                                               chunk_id_local_indexes: List[List[int]],
                                               index: Index,  # All with one index
                                               max_workers: int = MAX_WORKERS_FOR_CHUNK_ENGINE,
                                               ) -> List[Any]:
        """This function is only temporarily used by self.get_samples, for fixing multithreading bugs in LRU_Cache.
            Call get_multiple_chunks_from_chunk_ids, find out whether to decompress and call chunk.read_sample
            Added by zhouzhenyu 20240929.
        """
        chunks = self.get_multiple_chunks_from_chunk_ids(chunk_ids)

        def process_chunks(chunk, local_sample_indexes, tmp_index):
            """read samples from certain chunks"""
            _results = []
            for local_sample_index in local_sample_indexes:
                _result = chunk.read_sample(
                    local_sample_index,
                    cast=True,
                    is_tile=False,
                    decompress=True,
                )
                if len(tmp_index) > 1:
                    _result = _result[tuple(entry.value for entry in tmp_index.values[1:])]
                _results.append(_result)
            return _results

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_chunks, chunks, chunk_id_local_indexes,
                                        [index]*len(chunks)))
        merge_results = []
        for result in results:
            merge_results.extend(result)
        return merge_results



    def load_chunks(
            self,
            indices: List[int],
            in_order: bool = False,
            reverse: bool = False,
    ):
        """Fetches relevant chunks from base storage, adds them to cache and yields chunk info.
        If ``in_order`` is ``True``, chunks are yielded in order of the chunk_id_encoder.
        If ``reverse`` is ``True``, chunks are yielded in reverse order of the chunk_id_encoder.
        """
        chunk_infos = self.get_chunk_infos(indices)

        # some storage providers are not thread safe
        storages: Dict[int, StorageProvider] = {}

        if not (in_order or reverse):
            with ThreadPoolExecutor() as executor:
                futures_list = [
                    executor.submit(self._load_chunk, chunk_info, storages)
                    for chunk_info in chunk_infos
                ]
                for future in futures.as_completed(futures_list):
                    exception = future.exception()
                    if exception:
                        raise exception
                    chunk, chunk_info = future.result()
                    if chunk:
                        if _get_nbytes(chunk) <= self.cache.cache_size:  # Sherry: not elegant
                            self.cache.insert_in_cache(self.get_chunk_key_for_id(chunk_info[0]), chunk)
                    yield chunk_info
        else:
            with ThreadPoolExecutor() as executor:
                for result in executor.map(
                        self._load_chunk,
                        reversed(chunk_infos) if reverse else chunk_infos,
                        repeat(storages),
                ):
                    chunk, chunk_info = result
                    if chunk:
                        if _get_nbytes(chunk) <= self.cache.cache_size:
                            self.cache.insert_in_cache(self.get_chunk_key_for_id(chunk_info[0]), chunk)
                    yield chunk_info

    def get_chunk_key_for_id(self, chunk_id) -> str:
        """Get chunk key of an id. """
        chunk_name = ChunkIdEncoder.name_from_id(chunk_id)
        return get_chunk_key(self.key, chunk_name)

    def get_full_tiled_sample(self, global_sample_index: int):
        """Get fill tiled sample. """
        # Sherry: Tiled samples here.
        chunks = self.get_chunks_for_sample(global_sample_index)
        return combine_chunks(chunks, global_sample_index, self.tile_encoder)

    def get_partial_tiled_sample(self, global_sample_index: int, index: Index):
        """Get partial tiled sample. """
        # Sherry: Tiled samples here.
        tile_enc = self.tile_encoder
        chunk_ids = self.chunk_id_encoder[global_sample_index]
        sample_shape = tile_enc.get_sample_shape(global_sample_index)
        tile_shape = tile_enc.get_tile_shape(global_sample_index)
        ordered_tile_ids = np.array(chunk_ids).reshape(
            tile_enc.get_tile_layout_shape(global_sample_index)
        )
        tiles_index, sample_index = translate_slices(
            [v.value for v in index.values[1:]], sample_shape, tile_shape  # type: ignore
        )
        required_tile_ids = ordered_tile_ids[tiles_index]

        def _vectorized(_required_tile_ids):
            tile_list = []
            for chunk_id in _required_tile_ids:
                tile_list.append(self.get_chunk_from_chunk_id(chunk_id).read_sample(0, is_tile=True))
            return tile_list
        tiles = np.vectorize(_vectorized, otypes=[object])(required_tile_ids)

        sample = coalesce_tiles(tiles, tile_shape, None, self.tensor_meta.dtype)
        sample = sample[sample_index]
        return sample

    def get_empty_sample(self):
        """Get empty sample. """
        if self.num_samples == 0:
            raise ValueError("This tensor has no samples, cannot get empty sample.")
        htype = self.tensor_meta.htype
        dtype = self.tensor_meta.dtype
        if htype in ("text", "json", "list"):
            return get_empty_text_like_sample(htype)
        ndim = len(self.tensor_meta.max_shape)
        if self.is_sequence:
            ndim += 1
        shape = (0,) * ndim
        return np.ones(shape, dtype=dtype)

    def get_non_tiled_sample(
            self, global_sample_index, index, fetch_chunks=False, decompress=True
    ):
        """Get non-tiled sample. """
        # Sherry: if it is a video_sample, you need to use specific functions.
        return self.get_basic_sample(
            global_sample_index, index, fetch_chunks=fetch_chunks, decompress=decompress
        )

    def get_basic_sample(
            self,
            global_sample_index,
            index,
            fetch_chunks=False,
            is_tile=False,
            decompress=True,
    ):
        """Get basic samples."""
        chunk_id, row, worst_case_header_size = self.get_chunk_info(
            global_sample_index, fetch_chunks
        )
        local_sample_index = self.translate_to_local_index(global_sample_index, row)
        chunk = self.get_chunk_from_chunk_id(
            chunk_id, partial_chunk_bytes=worst_case_header_size
        )

        decompress = decompress
        ret = chunk.read_sample(
            local_sample_index,
            cast=self.tensor_meta.htype != "dicom",
            is_tile=is_tile,
            decompress=decompress,
        )
        if len(index) > 1:
            ret = ret[tuple(entry.value for entry in index.values[1:])]
        return ret

    def get_chunk_info(self, global_sample_index, fetch_chunks):
        """Returns the chunk_id, row and worst case header size of chunk containing the given sample."""
        enc = self.chunk_id_encoder
        out = enc.__getitem__(global_sample_index, return_row_index=True)  # Sherry: why so ugly here?
        chunk_id, row = out[0][0], out[0][1]

        worst_case_header_size = 0  # Sherry：why s3 protocol needs this different worst_case_header_size?
        if (
            not fetch_chunks
            and self.chunk_class != ChunkCompressedChunk
            and isinstance(self.base_storage, RomaProvider)
        ):
            prev = int(enc.array[row - 1][LAST_SEEN_INDEX_COLUMN]) if row > 0 else -1
            num_samples_in_chunk = int(enc.array[row][LAST_SEEN_INDEX_COLUMN]) - prev
            worst_case_header_size += HEADER_SIZE_BYTES + 10  # 10 for version
            ENTRY_SIZE = 4
            if self.tensor_meta.max_shape == self.tensor_meta.min_shape:
                num_shape_entries = 1 * (len(self.tensor_meta.min_shape) + 1)
                if self.is_text_like:
                    num_bytes_entries = num_samples_in_chunk * 3
                elif self.sample_compression is None:
                    num_bytes_entries = 1 * 3
                else:
                    num_bytes_entries = num_samples_in_chunk * 3
            else:
                num_shape_entries = num_samples_in_chunk * (
                    1 + len(self.tensor_meta.max_shape)
                )
                num_bytes_entries = num_samples_in_chunk * 3
            bytes_enc_size = num_bytes_entries * ENTRY_SIZE
            shape_enc_size = num_shape_entries * ENTRY_SIZE
            worst_case_header_size += shape_enc_size
            worst_case_header_size += bytes_enc_size

        return chunk_id, row, worst_case_header_size



    def get_chunks_for_sample(
            self,
            global_sample_index: int,
            copy: bool = False,
    ) -> List[BaseChunk]:
        """Retrieves the `Chunk` object corresponding to `global_sample_index`.
        Args:
            global_sample_index (int): Index relative to the entire tensor representing the sample.
            copy (bool): If True and the chunk exists in a different commit to the current commit, it will be copied.
                         Defaults to False.
        Returns:
            List[BaseChunk]: BaseChunk objects that contains `global_sample_index`.
        """
        return [
            self.get_chunk_from_chunk_id(chunk_id, copy, row=row)
            for chunk_id, row in self.chunk_id_encoder.__getitem__(global_sample_index, return_row_index=True)
        ]

    def get_chunk_from_chunk_id(
            self, chunk_id, copy: bool = False, partial_chunk_bytes=0, row=None
    ) -> BaseChunk:
        """Get chunks based on given chunk ids."""
        chunk_key = None
        try:
            chunk_name = ChunkIdEncoder.name_from_id(chunk_id)
            chunk_key = get_chunk_key(self.key, chunk_name)
            chunk = self.get_chunk(chunk_key, partial_chunk_bytes=partial_chunk_bytes)
            chunk.key = chunk_key
            chunk.id = chunk_id
            if copy and not self.chunk_in_target_commit(chunk_name, self.commit_id):
                chunk = self.copy_chunk(chunk, row=row)  # Sherry: need to handle this fault
            return chunk
        except Exception as e:
            raise GetChunkError(chunk_key) from e

    def get_multiple_chunks_from_chunk_ids(self, chunk_ids: List[int]) -> List[BaseChunk]:
        """This function is only temporarily used by self.get_samples, for fixing multithreading bugs in LRU_Cache.
            Get chunk_key from multiple chunk_ids and call get_multiple_chunks.
            Added by zhouzhenyu 20240929.

            Raises:
                GetChunkError
        """
        chunk_keys = []
        try:
            for chunk_id in chunk_ids:
                # Get chunk keys
                chunk_name = ChunkIdEncoder.name_from_id(chunk_id)
                chunk_key = get_chunk_key(self.key, chunk_name)
                chunk_keys.append(chunk_key)
            # Read chunks
            chunks = self.get_multiple_chunks(chunk_keys)
            assert len(chunks) == len(chunk_ids) == len(chunk_keys)
            # Do something about chunks (copied from get_chunk_from_chunk_id)
            for chunk, chunk_key, chunk_id in zip(chunks, chunk_keys, chunk_ids):  # Why modify those attributes ?
                chunk.key = chunk_key
                chunk.id = chunk_id
            return chunks
        except Exception as e:
            raise GetChunkError(str(chunk_keys)) from e

    def get_chunk(self, chunk_key: str, partial_chunk_bytes=0) -> BaseChunk:
        """Get chunks."""
        chunk = self.cache.get_muller_object(chunk_key, self.chunk_class, self.chunk_args,
                                           partial_bytes=partial_chunk_bytes)
        if not partial_chunk_bytes and isinstance(chunk.data_bytes, PartialReader):
            chunk.make_data_bytearray()
        return chunk

    def get_multiple_chunks(self, chunk_keys: List[str]) -> List[BaseChunk]:
        """This function is only temporarily used by self.get_samples, for fixing multithreading bugs in LRU_Cache.
            Read multiple chunks by calling self.cache's reading multiple API.
            Added by zhouzhenyu 20240929.
        """
        results = self.cache.get_multiple_muller_objects_with_muller_objects(chunk_keys,
                                                                         expected_class=self.chunk_class,
                                                                         meta=self.chunk_args)
        return results

    def read_bytes_for_sample(self, global_sample_index: int) -> bytes:
        """Read the bytes for a sample."""
        if self.chunk_compression:
            raise ValueError(
                "Cannot retrieve original bytes for samples in chunk-wise compressed tensors."
            )
        enc = self.chunk_id_encoder
        chunks = self.get_chunks_for_sample(global_sample_index)
        if len(chunks) > 1:
            raise NotImplementedError(
                "read_bytes_for_sample() is not implemented for tiled samples."
            )
        chunk = chunks[0]
        buffer = chunk.memoryview_data
        if not buffer:
            return b""
        if self.is_sequence:
            assert self.sequence_encoder is not None
            start_idx, end_idx = self.sequence_encoder[global_sample_index]
            end_idx -= 1
            start_idx, end_idx = map(
                enc.translate_index_relative_to_chunks, (start_idx, end_idx)
            )
            sb = chunk.byte_positions_encoder[start_idx][0]
            eb = chunk.byte_positions_encoder[end_idx][1]
        else:
            local_sample_index = enc.translate_index_relative_to_chunks(
                global_sample_index
            )
            sb, eb = chunk.byte_positions_encoder[local_sample_index]
        return buffer[sb:eb].tobytes()



    def read_shape_for_sample(
            self,
            global_sample_index: int,
    ) -> Tuple[int, ...]:
        """Read the shape of a sample. """
        return muller.core.chunk.read_shape_for_sample(self, global_sample_index)

    def shapes(
            self,
            index: Index,
            sample_shape_provider: Optional[Callable] = None,
            convert_bad_to_list: bool = True,
    ):
        """Returns the shapes of samples. """
        return muller.core.chunk.shapes(self, index, sample_shape_provider,
                                                                    convert_bad_to_list)

    def shape_interval(self, index: Index, sample_shape_provider: Optional[Callable] = None) -> ShapeInterval:
        """Shape interval. """
        meta = self.tensor_meta
        if self.is_sequence:
            tensor_length = index.length(self.sequence_length)
        else:
            tensor_length = index.length(meta.length)

        if index.is_trivial() or meta.min_shape == meta.max_shape or tensor_length == 0:
            if self.is_sequence:
                min_item_length, max_item_length = self._sequence_item_length_range
                min_length = [tensor_length, min_item_length]
                max_length = [tensor_length, max_item_length]
            else:
                min_length = max_length = [tensor_length]
            min_shape = min_length + list(meta.min_shape)
            max_shape = max_length + list(meta.max_shape)
        else:
            # need to fetch all shapes for the index
            shapes = self.shapes(
                index, sample_shape_provider, convert_bad_to_list=False
            )
            if self.is_sequence:
                if isinstance(shapes, np.ndarray):
                    # uniform sequence of shape (num_samples, num_items, ...)
                    min_shape = [*shapes.shape[:-1], *np.amin(shapes, axis=(0, 1))]
                    max_shape = [*shapes.shape[:-1], *np.amax(shapes, axis=(0, 1))]
                else:
                    # non-uniform sequence
                    item_lengths = list(map(len, shapes))
                    min_item_length, max_item_length = min(item_lengths), max(
                        item_lengths
                    )
                    min_item_shape = np.amin(
                        list(map(lambda x: np.amin(x, axis=0), shapes)), axis=0
                    )
                    max_item_shape = np.amax(
                        list(map(lambda x: np.amax(x, axis=0), shapes)), axis=0
                    )
                    min_shape = [len(shapes), min_item_length, *min_item_shape]
                    max_shape = [len(shapes), max_item_length, *max_item_shape]
            else:
                min_shape = [len(shapes), *np.amin(shapes, axis=0)]
                max_shape = [len(shapes), *np.amax(shapes, axis=0)]

        return ShapeInterval(min_shape, max_shape)

    def get_avg_chunk_size(self):
        """Returns the average chunk size. """
        num_chunks, num_samples = self.num_chunks, self.num_samples
        max_shape = self.tensor_meta.max_shape
        dtype = self.tensor_meta.dtype
        if dtype in ("Any", "List", None):
            return None
        shape = [num_samples] + max_shape
        nbytes = 1
        for dim in shape:  # not using np.prod to avoid overflow
            nbytes *= dim
        nbytes = nbytes * np.dtype(dtype).itemsize
        avg_chunk_size = nbytes / num_chunks
        return avg_chunk_size

    def list_all_chunks(self) -> List[str]:
        """Return list of all chunks for current `version_state['commit_id']` and tensor"""
        commit_id = self.commit_id
        if commit_id == FIRST_COMMIT_ID:
            arr = self.chunk_id_encoder.encoded
            if not arr.size:
                return []
            return [
                ChunkIdEncoder.name_from_id(chunk_id)
                for chunk_id in self.chunk_id_encoder.encoded[:, CHUNK_ID_COLUMN]
            ]  # type: ignore
        return [k for (k, v) in self.commit_chunk_map.chunks.items() if not v]  # type: ignore

    def list_all_chunks_path(self) -> List[str]:
        """Return list of paths to all chunks"""
        return [
            get_chunk_key(self.key, chunk)
            for chunk in self.list_all_chunks()
        ]

    def validate_num_samples_is_synchronized(self):
        """Check if tensor meta length and chunk ID encoder are representing the same number of samples.
        It determines if a user has tampered with the tensor meta or the chunk ID encoder, or if
        the tensor was corrupt.

        Raises:
            CorruptedMetaError: tensor_meta and chunk_id_encoder must have the same num samples.
        """

        tensor_meta_length = self.tensor_meta.length

        # compare chunk ID encoder and tensor meta

        # update this if we change self.num_samples implementation later to use tensor meta length instead of
        # chunk_id_encoder
        chunk_id_num_samples = self.num_samples

        if tensor_meta_length != chunk_id_num_samples:
            commit_id = self.commit_id
            tkey = get_tensor_meta_key(self.key, commit_id) \
                if self.split_tensor_meta else get_tensor_meta_key("", commit_id)
            ikey = get_chunk_id_encoder_key(self.key, commit_id)
            raise CorruptedMetaError(
                f"'{tkey}' and '{ikey}' have a record of different numbers of samples. "
                f"Got {tensor_meta_length} and {chunk_id_num_samples} respectively."
            )

    def clear(self):
        """Clears all samples and cachables."""
        self.cache.check_readonly()

        commit_id = self.commit_id

        chunk_folder_path = get_chunk_key(self.key, "")
        self.cache.clear(prefix=chunk_folder_path)

        enc_key = get_chunk_id_encoder_key(self.key, commit_id)
        self._chunk_id_encoder = None
        try:
            del self.meta_cache[enc_key]
        except KeyError:
            pass

        self.commit_diff.clear_data()

        tile_encoder_key = get_tensor_tile_encoder_key(self.key, commit_id)
        try:
            del self.cache[tile_encoder_key]
        except KeyError:
            pass
        self._tile_encoder = None

        seq_encoder_key = get_sequence_encoder_key(self.key, commit_id)
        try:
            del self.cache[seq_encoder_key]
        except KeyError:
            pass
        self._sequence_encoder = None

        self.tensor_meta.length = 0
        self.tensor_meta.min_shape = []
        self.tensor_meta.max_shape = []
        self.tensor_meta.is_dirty = True

        self.cache.maybe_flush()
        self.meta_cache.maybe_flush()

    def check_rechunk(self, chunk: BaseChunk, chunk_row: int):
        """function to check if there is a need to re-chunk the current one"""
        muller.core.chunk.check_rechunk(self, chunk, chunk_row)

    def is_tiled_sample(self, global_sample_index=None):
        """Whether it is tiled samples."""
        # Sherry: Tiled samples here.
        if self.enable_tile_encoder:
            if global_sample_index:
                return global_sample_index in self.tile_encoder
            raise ValueError("Can't find the tiled sample without global_sample_index.")
        return False

    def get_sample_object(
            self, sample_data, sample_shape, compression, dtype, decompress
    ):
        """Obtain sample objects. """
        return muller.core.chunk.get_sample_object(self, sample_data, sample_shape,
                                                                                 compression, dtype, decompress)

    def merge_regions(self, rows_groups: List[List[int]], ids_groups: List[List[int]]):
        """Merge regions. """
        muller.core.chunk.merge_regions(self, rows_groups, ids_groups)

    def write_initialization(self):
        """Write initialization. """
        # Sherry: To be removed.
        ffw_chunk_id_encoder(self.chunk_id_encoder)

    def convert_class_labels(self, samples):
        """Convert the class-labels type chunk."""
        tensor_info = self.tensor_meta.info
        tensor_name = self.tensor_meta.name or self.key
        class_names = tensor_info["class_names"]
        labels, additions = convert_to_idx(samples, class_names)
        if additions:
            for new in additions:
                class_names.append(new[0])
                logging.info(
                    f"'{new[0]}' added to {tensor_name}.info.class_names at index {new[1]}"
                )
            tensor_info["class_names"] = class_names
            self.tensor_meta.is_dirty = True
        self.commit_diff.modify_info()
        dataset_diff = get_dataset_diff_at_commit(self.commit_id, self.cache)
        dataset_diff.modify_tensor_info()
        self.cache.maybe_flush()
        return labels

    def sequence_numpy(
            self,
            index: Index,
            aslist: bool = False,  # aslist可能会带来返回类型不一致的问题！
            use_data_cache: bool = True,
            fetch_chunks: bool = False,
            max_workers: int = MAX_WORKERS_FOR_CHUNK_ENGINE,
            continuous: bool = False,
            full: bool = False,
    ):
        """Returns sequence as numpy. """
        return muller.core.chunk.sequence_numpy(self, index, aslist, use_data_cache,
                                                                             fetch_chunks, max_workers,
                                                                             continuous, full)

    def protected_numpy(
            self,
            index: Index,
            aslist: bool = False,
            index_list=None,
            **kwargs
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Reads samples from chunks and returns as a numpy array. If `aslist=True`, returns a sequence of numpy arrays.

        Args:
            index (Index): Represents the samples to read from chunks. See `Index` for more information.
            aslist (bool): If True, the samples will be returned as a list of numpy arrays. If False, returns a single
                           numpy array. Defaults to False. For polygons, aslist is always True.
            use_data_cache (bool): If True, the data cache is used to speed up the read if possible.
                                   If False, the data cache is ignored. Defaults to True.
            fetch_chunks (bool): If True, full chunks will be retrieved from the storage,
                                 otherwise only required bytes will be retrieved.
                This will always be True even if specified as False in the following cases:
                - The tensor is ChunkCompressed
                - The chunk which is being accessed has more than 128 samples.
            max_workers (int): max workers used in thread pool.
            continuous (bool): The data is continuous.

        Raises:
            DynamicTensorNumpyError: If shapes of the samples being read are not all the same.
            GetChunkError: If a chunk cannot be retrieved from the storage.
            ReadSampleFromChunkError: If a sample cannot be read from a chunk.
            GetDataFromLinkError: If data cannot be retrieved from a link.

        Returns:
            Union[np.ndarray, List[np.ndarray]]: Either a list of numpy arrays or a single numpy array
            (depending on the `aslist` argument).
        """
        return muller.core.chunk.protected_numpy(self, index, aslist, index_list,
                                                                                **kwargs)

    def create_new_chunk(self, register=True, row: Optional[int] = None) -> BaseChunk:
        """Creates and returns a new `Chunk`. Automatically creates an ID for it and puts a reference in the cache."""
        chunk_id = self.chunk_id_encoder.generate_chunk_id(register=register, row=row)
        chunk = self.chunk_class(*self.chunk_args)  # type: ignore
        chunk_name = ChunkIdEncoder.name_from_id(chunk_id)  # type: ignore
        chunk_key = get_chunk_key(self.key, chunk_name)
        if self.commit_chunk_map is not None:
            self.commit_chunk_map.add(chunk_name)
        chunk.key = chunk_key
        chunk.id = chunk_id
        chunk.update_tensor_meta_length = register
        if self.active_appended_chunk is not None:
            self.write_chunk_to_storage(self.active_appended_chunk)
        self.active_appended_chunk = chunk
        return chunk

    def get_chunk_infos(self, indices: List[int]):
        """Returns chunk infos for the chunks covered by the given indices."""
        indices = sorted(indices)
        indices_np = np.asarray(indices, dtype=np.uint32)  # type: ignore
        encoded = self.chunk_id_encoder.encoded
        last_idxs = encoded[:, -1]

        pos = np.searchsorted(indices_np, last_idxs, side="right")

        chunk_infos: List[List[Union[int, int, List[int], bool]]] = []

        last_pos = 0

        for i, _ in enumerate(last_idxs):
            is_tile = False
            if pos[i] == 0:
                continue
            if pos[i] == last_pos:
                # not tile
                if last_idxs[i] != last_idxs[i - 1]:
                    if pos[i] == len(indices_np):
                        break
                    continue
                # mark the previous chunk as tile
                chunk_infos[-1][3] = True
                # mark this chunk as tile
                is_tile = True

            idxs_in_chunk = indices_np[last_pos: pos[i]].tolist()  # type: ignore
            last_pos = pos[i]
            chunk_id = encoded[i][0].item()
            row = i
            chunk_infos.append([chunk_id, row, idxs_in_chunk, is_tile])

        return chunk_infos

    def _get_samples(
            self,
            chunk_id: int,
            row: int,
            idxs: List[int],
            index: Index,
            is_polygon: bool,
            aslist: bool,
    ):
        """Get samples from a chunk.

        Args:
            chunk_id (int): Chunk to read samples from. Can be ``None`` in case of tiles.
            row (int): Row of the chunk in the chunk_id_encoder.
            idxs (List[int]): List of global sample indices to read from this chunk.
            index (Index): Original index applied on the tensor.
            is_polygon (bool): Whether the tensor is a polygon tensor.
            aslist (bool): Whether to return a list or numpy array.

        Raises:
            GetChunkError: If a chunk cannot be retrieved from the storage.
            ReadSampleFromChunkError: If a sample cannot be read from a chunk.

        Returns:
            Dict of samples and shape of the last sample encountered in this chunk.
        """
        samples = {}
        last_shape = None

        for idx in idxs:
            if idx in samples:
                continue
            try:
                if not self.is_tiled_sample(idx) and idx < self.num_samples:
                    local_idx = self.translate_to_local_index(idx, row)
                    sample = self.read_basic_sample_from_chunk(
                        chunk_id, local_idx, index
                    )
                else:
                    sample = self.get_single_sample(idx, index)
            except GetChunkError as e:
                raise GetChunkError(e.chunk_key, idx, self.name) from e
            except ReadSampleFromChunkError as e:
                raise ReadSampleFromChunkError(e.chunk_key, idx, self.name) from e
            check_sample_shape(sample.shape, last_shape, self.key, index, aslist)
            last_shape = sample.shape
            if is_polygon:
                sample = [p.__array__() for p in sample]
            samples[idx] = sample
        return samples, last_shape

    def _get_fixed_shape_sample(self, global_sample_index, decompress):
        dtype = self.tensor_meta.dtype
        row, chunk_id = self.get_fixed_shape_chunk_id(
            global_sample_index, self.chunk_id_encoder.array, np.dtype(dtype).itemsize)
        chunk = self.get_chunk_from_chunk_id(chunk_id)
        local_index = self.translate_to_local_index(global_sample_index, row)
        return chunk.read_sample(
                local_index,
                cast=self.tensor_meta.htype != "dicom",
                is_tile=False,
                decompress=decompress
        )

    def _load_chunk(
            self,
            chunk_info: List,
            storages: Dict[int, StorageProvider],
    ):
        """Worker function for chunk retrieval."""
        chunk_key = self.get_chunk_key_for_id(chunk_info[0])
        result = self.cache.get_item_from_cache(chunk_key)
        if result is not None:
            return result, chunk_info
        is_tile = chunk_info[3]
        if is_tile:
            return None, chunk_info
        base_storage = storages.get(threading.get_ident())
        if base_storage is None:
            if isinstance(self.base_storage, MemoryProvider):
                base_storage = self.base_storage
            else:
                base_storage = self.base_storage.copy()
            storages[threading.get_ident()] = base_storage
        chunk = base_storage.__getitem__(chunk_key)
        return chunk, chunk_info

    def _get_chunk_names_from_path(self, chunk_index_path):
        try:
            chunk_ids = self.meta_cache[chunk_index_path]
        except KeyError:
            chunk_ids = ChunkIdEncoder(dtype=np.uint64)
            try:
                self.meta_cache[chunk_index_path] = chunk_ids
            except ReadOnlyModeError:
                pass
        if isinstance(chunk_ids, ChunkIdEncoder):
            ids = chunk_ids.encoded
        else:
            _, ids, _ = deserialize_chunkids(chunk_ids)
        all_chunk_names = [hex(item[0]).split('x')[-1] for item in ids]
        return all_chunk_names

    def _obtain_lengths(self, extending, samples, is_uuid):
        lengths = None
        if extending:
            if self.tensor_meta.htype == "text" and (self.chunk_class != SampleCompressedChunk):
                lengths = np.zeros(len(samples), dtype=np.uint32)
                for i, s in enumerate(samples):
                    try:
                        s = s.numpy()
                    except AttributeError:
                        pass
                    try:
                        if s.dtype.name[:3] == "str":
                            lengths[i] = len(str(s.reshape(())))
                    except AttributeError:
                        try:
                            lengths[i] = s.__len__()
                        except AttributeError:  # None
                            lengths[i] = 0
                        except TypeError:  # Numpy scalar str
                            lengths[i] = str(s).__len__()
            elif is_uuid:
                lengths = np.full(len(samples), 8, dtype=np.uint32)
        if isinstance(lengths, np.ndarray):
            lengths = lengths.tolist()
        return lengths

    def _obtain_current_chunk(self, current_chunk, register, start_chunk_row, extending):
        updated_chunks: List[Optional[str]] = []
        enc_ids: List[Optional[str]] = []
        if current_chunk is None:
            current_chunk = self.create_new_chunk(
                register and start_chunk_row is not None
            )
            current_chunk.update_tensor_meta_length = False
            if not register:
                updated_chunks.append(current_chunk.id)
            if extending:
                enc_ids.append(current_chunk.id)
        else:
            current_chunk.update_tensor_meta_length = False
            if extending:
                enc_ids.append(None)
        return current_chunk, updated_chunks, enc_ids

    def _update_current_chunk(self, register, start_chunk_row, enc_ids, enc_count, updated_chunks):
        current_chunk = self.create_new_chunk(
            register and start_chunk_row is not None, row=start_chunk_row
        )
        current_chunk.update_tensor_meta_length = False
        if start_chunk_row is not None:
            start_chunk_row += 1
        elif register:
            enc_ids.append(current_chunk.id)
            enc_count.append(0)
        if not register:
            updated_chunks.append(current_chunk.id)
        return current_chunk, updated_chunks, enc_ids, enc_count, start_chunk_row

    def _update_enc_ids_and_count(self, enc, enc_ids, enc_count, incoming_num_samples):
        if enc_ids[0] is None:
            enc_ids.pop(0)
            start_chunk_incr = enc_count.pop(0)
            enc.encoded[-1, 1] += start_chunk_incr
            enc.is_dirty = True
        if enc_count:
            enc_arr = enc.encoded
            n = len(enc_arr)
            if n:
                enc_count[0] += enc_arr[-1, 1]
            else:
                enc_count[0] -= 1
            enc_last_seen = np.cumsum(enc_count, dtype=np.uint64)
            arr = np.zeros((n + len(enc_ids), 2), dtype=np.uint64)
            if n:
                arr[:n] = enc_arr
            new = arr[n:]
            new[:, 0] = enc_ids
            new[:, 1] = enc_last_seen
            enc.encoded = arr
            enc.is_dirty = True
        self.tensor_meta.update_length(incoming_num_samples)


def check_sample_shape(shape, last_shape, key, index, aslist):
    """Check the sample shape. """
    if not aslist and last_shape is not None and shape != last_shape:
        raise DynamicTensorNumpyError(key, index, "shape")
    
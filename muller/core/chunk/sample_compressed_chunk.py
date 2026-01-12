# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/core/chunk/sample_compressed_chunk.py
#
# Modifications Copyright (c) 2026 Xueling Lin

from typing import List, Optional, Tuple

from muller.core.chunk.base_chunk import BaseChunk, InputSample, catch_chunk_read_error
from muller.core.compression import decompress_bytes, decompress_array
from muller.core.sample import Sample  # type: ignore
from muller.core.serialize import (
    bytes_to_text,
    check_sample_shape,
)
from muller.core.tiling.sample_tiles import SampleTiles


class SampleCompressedChunk(BaseChunk):

    @staticmethod
    def _normalize_index(index, nframes):
        """Normalize video frame indexes to standard format."""
        if index is None:
            return 0, nframes, 1, False
        if isinstance(index, int):
            start = index if index >= 0 else nframes + index
            return start, start + 1, 1, False
        if isinstance(index, slice):
            step = index.step or 1
            reverse = step < 0
            if reverse:
                step = abs(step)

            # process start
            start = index.start
            if start is None:
                start = nframes if reverse else 0
            elif start < 0:
                start = nframes + start

            # process stop
            stop = index.stop
            if stop is None:
                stop = -1 if reverse else nframes
            elif stop < 0:
                stop = nframes + stop

            # solve the bounding problem
            if reverse:
                start, stop = stop + 1, start + 1

            return start, stop, step, reverse
        if isinstance(index, list):
            raise IndexError("List indices not allowed. Use slice notation like [5:10] or [0:100:5]")
        raise IndexError(f"Invalid index type: {type(index)}. Must be int or slice")

    def extend_if_has_space(self, incoming_samples: List[InputSample], update_tensor_meta: bool = True,
                            lengths: Optional[List[int]] = None, ignore_errors: bool = False, **kwargs) -> float:
        self.prepare_for_write()
        num_samples: float = 0.0
        dtype = self.dtype if self.is_byte_compression else None
        compr = self.compression
        skipped: List[int] = []

        for i, incoming_sample in enumerate(incoming_samples):
            try:
                serialized_sample, shape = self.serialize_sample(incoming_sample, compr)
                if shape is not None:
                    self.num_dims = self.num_dims or len(shape)
                    check_sample_shape(shape, self.num_dims)
            except Exception:
                if ignore_errors:
                    skipped.append(i)
                    continue
                raise

            if isinstance(serialized_sample, SampleTiles):
                incoming_samples[i] = serialized_sample  # type: ignore
                if self.is_empty:
                    self.write_tile(serialized_sample)
                    num_samples += 0.5
                break
            sample_nbytes = len(serialized_sample)
            if self.is_empty or self.can_fit_sample(sample_nbytes):
                self.data_bytes += serialized_sample  # type: ignore

                self.register_in_meta_and_headers(
                    sample_nbytes,
                    shape,
                    update_tensor_meta=update_tensor_meta,
                )
                num_samples += 1.0
            else:
                if serialized_sample:
                    sample = Sample(
                        buffer=serialized_sample, compression=compr, shape=shape, dtype=dtype  # type: ignore
                    )
                    sample.htype = self.htype
                    incoming_samples[i] = sample
                break

        for i in reversed(skipped):
            incoming_samples.pop(i)
        return num_samples

    def register_in_meta_and_headers(
        self,
        sample_nbytes: Optional[int],
        shape,
        update_tensor_meta: bool = True,
        num_samples: int = 1,
    ):
        self.register_sample_to_headers(sample_nbytes, shape, num_samples)
        if update_tensor_meta:
            self.update_tensor_meta(shape, num_samples)

    def update_tensor_meta(self, shape, num_samples):
        if self._update_tensor_meta_length:
            self.tensor_meta.update_length(num_samples)
        if shape is not None:
            self.tensor_meta.update_shape_interval(shape)

    def register_sample_to_headers(
        self,
        incoming_num_bytes: Optional[int],
        sample_shape: Tuple[int],
        num_samples: int = 1,
    ):
        if incoming_num_bytes is not None:
            self.byte_positions_encoder.register_samples(
                incoming_num_bytes, num_samples
            )
        if sample_shape is not None:
            if self.shapes_encoder.is_empty():
                padding = self.byte_positions_encoder.num_samples - num_samples
                self._fill_empty_shapes(sample_shape, padding)
            self.shapes_encoder.register_samples(sample_shape, num_samples)

    @catch_chunk_read_error
    def read_sample(  # type: ignore
        self,
        local_index: int,
        cast: bool = True,
        copy: bool = False,
        decompress: bool = True,
        **kwargs,
    ):
        sub_index = kwargs.get("sub_index", None)
        to_pil = kwargs.get("to_pil", False)
        self.check_empty_before_read()
        partial_sample_tile = self._get_partial_sample_tile()
        if partial_sample_tile is not None:
            return partial_sample_tile
        buffer = self.memoryview_data
        bps = self.byte_positions_encoder
        bps_empty = bps.is_empty()
        if not bps_empty:
            sb, eb = bps[local_index]
            buffer = buffer[sb:eb]
        if not decompress:
            return bytes(buffer) if copy else buffer
        try:
            shape = self.shapes_encoder[local_index]
        except IndexError as e:
            if not bps_empty:
                self.num_dims = self.num_dims or len(self.tensor_meta.max_shape)
                shape = (0,) * self.num_dims
            else:
                raise e

        nframes = shape[0]
        if self.is_text_like:
            buffer = decompress_bytes(buffer, compression=self.compression)
            buffer = bytes(buffer)
            return bytes_to_text(buffer, self.htype)

        squeeze = isinstance(sub_index, int)

        start, stop, step, reverse = self._normalize_index(sub_index, nframes)

        if start > nframes:
            raise IndexError("Start index out of bounds.")

        sample = decompress_array(
            buffer,
            shape,
            self.dtype,
            self.compression,
            start_idx=start,
            end_idx=int(stop),
            step=step,
            reverse=reverse,
            to_pil=to_pil,
        )
        if to_pil:
            return sample

        if squeeze:
            sample = sample.squeeze(0)

        if cast and sample.dtype != self.dtype:
            sample = sample.astype(self.dtype)
        elif copy and not sample.flags["WRITEABLE"]:
            sample = sample.copy()
        return sample

    def update_sample(self, local_index: int, new_sample: InputSample):
        self.prepare_for_write()
        serialized_sample, shape = self.serialize_sample(
            new_sample, self.compression, break_into_tiles=False
        )

        self.check_shape_for_update(shape)
        old_data = self.data_bytes
        self.data_bytes = self.create_updated_data(
            local_index, old_data, serialized_sample
        )

        # update encoders and meta
        new_nb = (
            None if self.byte_positions_encoder.is_empty() else len(serialized_sample)
        )
        self.update_in_meta_and_headers(local_index, new_nb, shape)

    def _fill_empty_shapes(self, shape, num_samples):
        dims = len(shape)
        self.num_dims = self.num_dims or dims
        if num_samples > 0:
            empty_shape = (0,) * dims
            self.shapes_encoder.register_samples(empty_shape, num_samples)
            self.tensor_meta.update_shape_interval(empty_shape)

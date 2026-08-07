# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

"""Dedicated unit tests for lz4 / jpeg compress ↔ decompress round-trips.

Compression is exercised implicitly across the suite, but there was no focused
coverage of the byte (lz4) and image (jpeg) codecs themselves.
"""

import numpy as np
import pytest

from muller.core.compression import (
    compress_array,
    compress_bytes,
    decompress_array,
    decompress_bytes,
)
from muller.util.exceptions import (
    SampleCompressionError,
    SampleDecompressionError,
    UnsupportedCompressionError,
)


class TestLz4ByteCompression:
    def test_compress_decompress_bytes_round_trip(self):
        payload = b"muller-lz4-" + bytes(range(256)) * 4
        compressed = compress_bytes(payload, "lz4")
        assert compressed != payload
        assert decompress_bytes(compressed, "lz4") == payload

    def test_empty_bytes_round_trip(self):
        assert compress_bytes(b"", "lz4") == b""
        assert decompress_bytes(b"", "lz4") == b""

    def test_compress_array_lz4_round_trip(self):
        arr = np.arange(64, dtype=np.float64).reshape(8, 8)
        compressed = compress_array(arr, "lz4")
        restored = decompress_array(
            compressed, shape=arr.shape, dtype=str(arr.dtype), compression="lz4"
        )
        np.testing.assert_array_equal(restored, arr)

    def test_lz4_requires_shape_and_dtype_on_decompress(self):
        compressed = compress_array(np.arange(4, dtype=np.int64), "lz4")
        # The ValueError is wrapped in SampleDecompressionError by decompress_array.
        with pytest.raises(SampleDecompressionError, match="dtype and shape"):
            decompress_array(compressed, compression="lz4")


class TestJpegImageCompression:
    def test_jpeg_round_trip_preserves_shape_and_channels(self):
        # Smooth gradient: JPEG is lossy, so assert shape/dtype + bounded MAE.
        ys = np.linspace(0, 255, 32, dtype=np.float32)[:, None]
        xs = np.linspace(0, 255, 48, dtype=np.float32)[None, :]
        arr = np.stack(
            [
                np.broadcast_to(xs, (32, 48)),
                np.broadcast_to(ys, (32, 48)),
                np.full((32, 48), 128.0),
            ],
            axis=-1,
        ).astype(np.uint8)
        compressed = compress_array(arr, "jpeg")
        assert isinstance(compressed, (bytes, memoryview))
        assert len(compressed) > 0
        restored = decompress_array(compressed, compression="jpeg")
        assert restored.shape == arr.shape
        assert restored.dtype == np.uint8
        assert np.mean(np.abs(restored.astype(np.int16) - arr.astype(np.int16))) < 5

    def test_unsupported_compression_raises(self):
        arr = np.zeros((4, 4), dtype=np.uint8)
        with pytest.raises(UnsupportedCompressionError):
            compress_array(arr, "not-a-real-codec")

    def test_jpeg_rejects_non_image_like_float_array(self):
        arr = np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)
        with pytest.raises(SampleCompressionError):
            compress_array(arr, "jpeg")

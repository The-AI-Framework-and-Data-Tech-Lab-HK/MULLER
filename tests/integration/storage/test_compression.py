# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

"""Integration tests for dataset-level lz4 / jpeg compression round-trips."""

import numpy as np

import muller
from muller.compression import COMPRESSION_ALIASES

from tests.constants import SAMPLE_FILES, TEST_COMPRESSION_PATH


def _fresh_dataset():
    return muller.dataset(path=TEST_COMPRESSION_PATH, overwrite=True)


def test_chunk_compression_lz4_numeric_round_trip():
    ds = _fresh_dataset()
    ds.create_tensor("readings", dtype="float64", chunk_compression="lz4")
    values = np.arange(50, dtype=np.float64).reshape(-1, 1)
    ds.readings.extend(values)

    assert ds.readings.meta.chunk_compression == "lz4"
    np.testing.assert_array_equal(ds.readings.numpy(), values)

    # Reload from disk to ensure compressed chunks survive flush + reopen.
    path = TEST_COMPRESSION_PATH
    del ds
    ds2 = muller.dataset(path=path)
    np.testing.assert_array_equal(ds2.readings.numpy(), values)


def test_sample_compression_jpeg_image_round_trip():
    ds = _fresh_dataset()
    ds.create_tensor("images", htype="image", sample_compression="jpeg")
    sample = muller.read(SAMPLE_FILES["jpg_1"])
    ds.images.append(sample)
    ds.images.append(muller.read(SAMPLE_FILES["jpg_2"]))

    assert ds.images.meta.sample_compression == "jpeg"
    assert len(ds.images) == 2
    first = ds.images[0].numpy()
    assert first.ndim == 3 and first.shape[2] == 3
    assert first.dtype == np.uint8


def test_jpg_alias_normalized_to_jpeg():
    """``jpg`` is accepted at create_tensor time and stored as ``jpeg``."""
    assert COMPRESSION_ALIASES["jpg"] == "jpeg"
    ds = _fresh_dataset()
    ds.create_tensor("images", htype="image", sample_compression="jpg")
    assert ds.images.meta.sample_compression == "jpeg"
    ds.images.append(muller.read(SAMPLE_FILES["jpg_1"]))
    assert ds.images[0].numpy().shape[2] == 3


def test_sample_compression_lz4_text_round_trip():
    ds = _fresh_dataset()
    ds.create_tensor("notes", htype="text", sample_compression="lz4")
    texts = ["hello", "压缩测试", "lz4 round trip"]
    ds.notes.extend(texts)

    assert ds.notes.meta.sample_compression == "lz4"
    assert ds.notes.numpy(aslist=True) == texts

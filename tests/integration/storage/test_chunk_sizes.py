# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import numpy as np
import pytest

import muller
from muller.constants import KB
from tests.utils import official_path, official_creds
from tests.constants import TEST_APPEND_PATH, TEST_EXTEND_PATH, TEST_APPEND_AND_EXTEND_PATH, TEST_CLEAR_PATH


def _assert_num_chunks(tensor, expected_num_chunks):
    chunk_engine = tensor.chunk_engine
    actual_num_chunks = chunk_engine.chunk_id_encoder.num_chunks
    assert actual_num_chunks == expected_num_chunks


def _create_tensors(ds):
    images = ds.create_tensor(
        "images",
        htype="image",
        sample_compression=None,
        max_chunk_size=32 * KB,
        tiling_threshold=16 * KB,
    )
    labels = ds.create_tensor(
        "labels", htype="class_label", max_chunk_size=32 * KB, tiling_threshold=16 * KB
    )
    return images, labels


def _append_tensors(images, labels):
    for i in range(100):
        x = np.ones((28, 28), dtype=np.uint8) * i
        y = np.uint32(i)

        images.append(x)
        labels.append(y)


def _extend_tensors(images, labels):
    images.extend(np.ones((100, 28, 28), dtype=np.uint8))
    labels.extend(np.ones(100, dtype=np.uint32))


def _clear_tensors(images, labels):
    images.clear()
    labels.clear()


def test_append(storage):
    ds = muller.dataset(official_path(storage, TEST_APPEND_PATH), creds=official_creds(storage), overwrite=True)
    images, labels = _create_tensors(ds)

    with ds:
        _append_tensors(images, labels)
        _assert_num_chunks(labels, 1)
        _assert_num_chunks(images, 5)

        _append_tensors(images, labels)
        _assert_num_chunks(labels, 1)
        _assert_num_chunks(images, 10)

        _append_tensors(images, labels)
        _assert_num_chunks(labels, 1)
        _assert_num_chunks(images, 15)

    assert len(ds) == 300


def test_extend(storage):
    ds = muller.dataset(official_path(storage, TEST_EXTEND_PATH), creds=official_creds(storage), overwrite=True)
    images, labels = _create_tensors(ds)

    with ds:
        _extend_tensors(images, labels)
        _assert_num_chunks(labels, 1)
        _assert_num_chunks(images, 5)

        _extend_tensors(images, labels)
        _assert_num_chunks(labels, 1)
        _assert_num_chunks(images, 10)

        _extend_tensors(images, labels)
        _assert_num_chunks(labels, 1)
        _assert_num_chunks(images, 15)

    assert len(ds) == 300


def test_extend_and_append(storage):
    ds = muller.dataset(official_path(storage, TEST_APPEND_AND_EXTEND_PATH),
                       creds=official_creds(storage), overwrite=True)
    images, labels = _create_tensors(ds)

    with ds:
        _extend_tensors(images, labels)
        _assert_num_chunks(labels, 1)
        _assert_num_chunks(images, 5)

        _append_tensors(images, labels)
        _assert_num_chunks(labels, 1)
        _assert_num_chunks(images, 10)

        _extend_tensors(images, labels)
        _assert_num_chunks(labels, 1)
        _assert_num_chunks(images, 15)

        _append_tensors(images, labels)
        _assert_num_chunks(labels, 1)
        _assert_num_chunks(images, 20)

    assert len(ds) == 400


def test_clear(storage):
    ds = muller.dataset(official_path(storage, TEST_CLEAR_PATH), creds=official_creds(storage), overwrite=True)
    images, labels = _create_tensors(ds)

    with ds:
        _clear_tensors(images, labels)
        _assert_num_chunks(labels, 0)
        _assert_num_chunks(images, 0)


if __name__ == "__main__":
    pytest.main(["-s", "test_chunk_sizes.py"])

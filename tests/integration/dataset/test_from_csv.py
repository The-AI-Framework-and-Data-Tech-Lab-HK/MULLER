# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import csv
import os

import numpy as np
import pytest

import muller
from tests.utils import official_path, official_creds
from tests.constants import (
    TEST_CSV_DATASET_PATH,
    CSV_WITH_PATHS_FILE,
    CSV_TEXT_ONLY_FILE,
    CIFAR10_TRAIN_DIR,
    NUM_CIFAR10_SAMPLES,
)


def _read_label(txt_path):
    with open(txt_path, "r") as f:
        return int(f.read().strip())


def _ensure_csv_with_paths():
    """Create the CSV file with image paths and labels if it doesn't already exist."""
    if os.path.exists(CSV_WITH_PATHS_FILE):
        return
    with open(CSV_WITH_PATHS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])
        for i in range(NUM_CIFAR10_SAMPLES):
            img_path = os.path.join(CIFAR10_TRAIN_DIR, f"{i}.jpeg")
            label = _read_label(os.path.join(CIFAR10_TRAIN_DIR, f"{i}.txt"))
            writer.writerow([img_path, str(label)])


def _ensure_csv_text_only():
    """Create the CSV file with text-only data if it doesn't already exist."""
    if os.path.exists(CSV_TEXT_ONLY_FILE):
        return
    with open(CSV_TEXT_ONLY_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "score"])
        for i in range(NUM_CIFAR10_SAMPLES):
            writer.writerow([f"sample_{i}", str(i * 10)])


def test_from_csv_text_and_generic(storage):
    """CSV with plain text and numeric columns, using from_csv to create a new dataset."""
    _ensure_csv_text_only()
    schema = {
        "name": ("text", "", "lz4"),
        "score": ("text", "", "lz4"),
    }
    ds = muller.from_csv(
        csv_path=CSV_TEXT_ONLY_FILE,
        muller_path=official_path(storage, TEST_CSV_DATASET_PATH),
        schema=schema,
        workers=0,
    )
    assert len(ds) == NUM_CIFAR10_SAMPLES
    assert ds["name"][0].numpy() == "sample_0"
    assert ds["name"][9].numpy() == "sample_9"
    assert ds["score"][0].numpy() == "0"
    assert ds["score"][9].numpy() == "90"


def test_from_csv_image_read(storage):
    """CSV with image paths loaded via muller.read(), creating a new dataset."""
    _ensure_csv_with_paths()
    schema = {
        "image_path": ("image", "uint8", "jpeg"),
        "label": ("text", "", "lz4"),
    }
    path_columns = {"image_path": "read"}
    ds = muller.from_csv(
        csv_path=CSV_WITH_PATHS_FILE,
        muller_path=official_path(storage, TEST_CSV_DATASET_PATH),
        schema=schema,
        path_columns=path_columns,
        workers=0,
    )
    assert len(ds) == NUM_CIFAR10_SAMPLES
    # Verify images are stored correctly (CIFAR-10 images are 32x32 RGB)
    img = ds["image_path"][0].numpy()
    assert img.shape == (32, 32, 3)
    assert img.dtype == np.uint8


def test_from_csv_path_as_text(storage):
    """CSV with image paths stored as text strings instead of loading the file."""
    _ensure_csv_with_paths()
    schema = {
        "image_path": ("text", "", "lz4"),
        "label": ("text", "", "lz4"),
    }
    path_columns = {"image_path": "text"}
    ds = muller.from_csv(
        csv_path=CSV_WITH_PATHS_FILE,
        muller_path=official_path(storage, TEST_CSV_DATASET_PATH),
        schema=schema,
        path_columns=path_columns,
        workers=0,
    )
    assert len(ds) == NUM_CIFAR10_SAMPLES
    # The stored value should be the relative path string
    expected_path = os.path.join(CIFAR10_TRAIN_DIR, "0.jpeg")
    assert ds["image_path"][0].numpy() == expected_path


def test_from_csv_mixed_columns(storage):
    """CSV with image path (read mode) + label (direct text)."""
    _ensure_csv_with_paths()
    schema = {
        "image_path": ("image", "uint8", "jpeg"),
        "label": ("text", "", "lz4"),
    }
    path_columns = {"image_path": "read"}
    ds = muller.from_csv(
        csv_path=CSV_WITH_PATHS_FILE,
        muller_path=official_path(storage, TEST_CSV_DATASET_PATH),
        schema=schema,
        path_columns=path_columns,
        workers=0,
    )
    assert len(ds) == NUM_CIFAR10_SAMPLES
    # Verify image
    img = ds["image_path"][0].numpy()
    assert img.shape == (32, 32, 3)
    # Verify label
    expected_label = str(_read_label(os.path.join(CIFAR10_TRAIN_DIR, "0.txt")))
    assert ds["label"][0].numpy() == expected_label


def test_add_data_from_csv_to_existing_dataset(storage):
    """Create a dataset with tensors first, then append data from CSV."""
    _ensure_csv_with_paths()

    # Create dataset with tensors
    ds = muller.dataset(
        path=official_path(storage, TEST_CSV_DATASET_PATH),
        creds=official_creds(storage),
        overwrite=True,
    )
    ds.create_tensor("image_path", htype="image", sample_compression="jpeg")
    ds.create_tensor("label", htype="text", sample_compression="lz4")
    assert len(ds) == 0

    # Append data from CSV
    path_columns = {"image_path": "read"}
    ds.add_data_from_csv(
        csv_path=CSV_WITH_PATHS_FILE,
        path_columns=path_columns,
        workers=0,
    )
    assert len(ds) == NUM_CIFAR10_SAMPLES
    img = ds["image_path"][0].numpy()
    assert img.shape == (32, 32, 3)
    assert img.dtype == np.uint8

    expected_label = str(_read_label(os.path.join(CIFAR10_TRAIN_DIR, "0.txt")))
    assert ds["label"][0].numpy() == expected_label


def test_add_data_from_csv_append_twice(storage):
    """Append CSV data twice to verify data accumulates correctly."""
    _ensure_csv_with_paths()

    ds = muller.dataset(
        path=official_path(storage, TEST_CSV_DATASET_PATH),
        creds=official_creds(storage),
        overwrite=True,
    )
    ds.create_tensor("image_path", htype="image", sample_compression="jpeg")
    ds.create_tensor("label", htype="text", sample_compression="lz4")

    path_columns = {"image_path": "read"}
    ds.add_data_from_csv(csv_path=CSV_WITH_PATHS_FILE, path_columns=path_columns, workers=0)
    assert len(ds) == NUM_CIFAR10_SAMPLES

    ds.add_data_from_csv(csv_path=CSV_WITH_PATHS_FILE, path_columns=path_columns, workers=0)
    assert len(ds) == NUM_CIFAR10_SAMPLES * 2


def test_from_csv_missing_file(storage):
    """Should raise ValueError for non-existent CSV file."""
    with pytest.raises(ValueError):
        muller.from_csv(
            csv_path="nonexistent/path/to/file.csv",
            muller_path=official_path(storage, TEST_CSV_DATASET_PATH),
        )


def test_from_csv_empty_path(storage):
    """Should raise ValueError when csv_path is empty."""
    with pytest.raises(ValueError, match="csv_path and muller_path cannot be empty"):
        muller.from_csv(
            csv_path="",
            muller_path=official_path(storage, TEST_CSV_DATASET_PATH),
        )


def test_add_data_from_csv_empty_path(storage):
    """Should raise ValueError when csv_path is empty on instance method."""
    ds = muller.dataset(
        path=official_path(storage, TEST_CSV_DATASET_PATH),
        creds=official_creds(storage),
        overwrite=True,
    )
    ds.create_tensor("col1", htype="text")
    with pytest.raises(ValueError, match="csv_path cannot be empty"):
        ds.add_data_from_csv(csv_path="")


def test_add_data_from_csv_mismatched_columns(storage):
    """Should raise ValueError when CSV columns don't match dataset tensors."""
    _ensure_csv_text_only()

    ds = muller.dataset(
        path=official_path(storage, TEST_CSV_DATASET_PATH),
        creds=official_creds(storage),
        overwrite=True,
    )
    ds.create_tensor("col_a", htype="text")
    ds.create_tensor("col_b", htype="text")

    with pytest.raises(ValueError, match="do not match"):
        ds.add_data_from_csv(csv_path=CSV_TEXT_ONLY_FILE, workers=0)

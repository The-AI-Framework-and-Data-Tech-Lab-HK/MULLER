# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import csv
import os
import shutil
import tempfile

import numpy as np
import pytest

import muller

CIFAR10_TRAIN_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "cifar10", "shared", "train"
)

CSV_DATASET_PATH = "results/csv_dataset_test"


def _read_label(txt_path):
    with open(txt_path, "r") as f:
        return int(f.read().strip())


def _make_csv_with_paths(csv_path, num_samples=10):
    """Create a CSV file with image paths and labels from cifar10 test data."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])
        for i in range(num_samples):
            img_path = os.path.abspath(os.path.join(CIFAR10_TRAIN_DIR, f"{i}.jpeg"))
            label = _read_label(os.path.join(CIFAR10_TRAIN_DIR, f"{i}.txt"))
            writer.writerow([img_path, str(label)])


def _make_csv_text_only(csv_path, num_samples=10):
    """Create a CSV file with text-only data (names and numeric values)."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "score"])
        for i in range(num_samples):
            writer.writerow([f"sample_{i}", str(i * 10)])


def _cleanup(path):
    if os.path.exists(path):
        shutil.rmtree(path)


class TestFromCsvDirectValues:
    """Test importing CSV with direct text/numeric values."""

    def setup_method(self):
        _cleanup(CSV_DATASET_PATH)

    def teardown_method(self):
        _cleanup(CSV_DATASET_PATH)

    def test_from_csv_text_and_generic(self):
        """CSV with plain text and numeric columns, using from_csv to create a new dataset."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_path = f.name
        try:
            _make_csv_text_only(csv_path, num_samples=5)
            schema = {
                "name": ("text", "", "lz4"),
                "score": ("text", "", "lz4"),
            }
            ds = muller.from_csv(
                csv_path=csv_path,
                muller_path=CSV_DATASET_PATH,
                schema=schema,
                workers=0,
            )
            assert len(ds) == 5
            assert ds["name"][0].numpy() == "sample_0"
            assert ds["name"][4].numpy() == "sample_4"
            assert ds["score"][0].numpy() == "0"
            assert ds["score"][4].numpy() == "40"
        finally:
            os.unlink(csv_path)


class TestFromCsvPathColumnsRead:
    """Test importing CSV where columns contain file paths loaded with muller.read()."""

    def setup_method(self):
        _cleanup(CSV_DATASET_PATH)

    def teardown_method(self):
        _cleanup(CSV_DATASET_PATH)

    def test_from_csv_image_read(self):
        """CSV with image paths loaded via muller.read(), creating a new dataset."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_path = f.name
        try:
            _make_csv_with_paths(csv_path, num_samples=5)
            schema = {
                "image_path": ("image", "uint8", "jpeg"),
                "label": ("text", "", "lz4"),
            }
            path_columns = {"image_path": "read"}
            ds = muller.from_csv(
                csv_path=csv_path,
                muller_path=CSV_DATASET_PATH,
                schema=schema,
                path_columns=path_columns,
                workers=0,
            )
            assert len(ds) == 5
            # Verify images are stored correctly (CIFAR-10 images are 32x32 RGB)
            img = ds["image_path"][0].numpy()
            assert img.shape == (32, 32, 3)
            assert img.dtype == np.uint8
        finally:
            os.unlink(csv_path)


class TestFromCsvPathColumnsText:
    """Test importing CSV where path columns are stored as plain text strings."""

    def setup_method(self):
        _cleanup(CSV_DATASET_PATH)

    def teardown_method(self):
        _cleanup(CSV_DATASET_PATH)

    def test_from_csv_path_as_text(self):
        """CSV with image paths stored as text strings instead of loading the file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_path = f.name
        try:
            _make_csv_with_paths(csv_path, num_samples=5)
            schema = {
                "image_path": ("text", "", "lz4"),
                "label": ("text", "", "lz4"),
            }
            path_columns = {"image_path": "text"}
            ds = muller.from_csv(
                csv_path=csv_path,
                muller_path=CSV_DATASET_PATH,
                schema=schema,
                path_columns=path_columns,
                workers=0,
            )
            assert len(ds) == 5
            # The stored value should be the path string
            expected_path = os.path.abspath(os.path.join(CIFAR10_TRAIN_DIR, "0.jpeg"))
            assert ds["image_path"][0].numpy() == expected_path
        finally:
            os.unlink(csv_path)


class TestFromCsvMixed:
    """Test importing CSV with mixed columns: some paths (read), some direct values."""

    def setup_method(self):
        _cleanup(CSV_DATASET_PATH)

    def teardown_method(self):
        _cleanup(CSV_DATASET_PATH)

    def test_from_csv_mixed_columns(self):
        """CSV with image path (read mode) + label (direct text)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_path = f.name
        try:
            _make_csv_with_paths(csv_path, num_samples=10)
            schema = {
                "image_path": ("image", "uint8", "jpeg"),
                "label": ("text", "", "lz4"),
            }
            path_columns = {"image_path": "read"}
            ds = muller.from_csv(
                csv_path=csv_path,
                muller_path=CSV_DATASET_PATH,
                schema=schema,
                path_columns=path_columns,
                workers=0,
            )
            assert len(ds) == 10
            # Verify image
            img = ds["image_path"][0].numpy()
            assert img.shape == (32, 32, 3)
            # Verify label
            expected_label = str(_read_label(os.path.join(CIFAR10_TRAIN_DIR, "0.txt")))
            assert ds["label"][0].numpy() == expected_label
        finally:
            os.unlink(csv_path)


class TestAddDataFromCsv:
    """Test ds.add_data_from_csv() instance method on an existing dataset."""

    def setup_method(self):
        _cleanup(CSV_DATASET_PATH)

    def teardown_method(self):
        _cleanup(CSV_DATASET_PATH)

    def test_add_data_from_csv_to_existing_dataset(self):
        """Create a dataset with tensors first, then append data from CSV."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_path = f.name
        try:
            _make_csv_with_paths(csv_path, num_samples=5)

            # Create dataset with tensors
            ds = muller.dataset(path=CSV_DATASET_PATH, overwrite=True)
            ds.create_tensor("image_path", htype="image", sample_compression="jpeg")
            ds.create_tensor("label", htype="text", sample_compression="lz4")
            assert len(ds) == 0

            # Append data from CSV
            path_columns = {"image_path": "read"}
            ds.add_data_from_csv(
                csv_path=csv_path,
                path_columns=path_columns,
                workers=0,
            )
            assert len(ds) == 5
            img = ds["image_path"][0].numpy()
            assert img.shape == (32, 32, 3)
            assert img.dtype == np.uint8

            expected_label = str(_read_label(os.path.join(CIFAR10_TRAIN_DIR, "0.txt")))
            assert ds["label"][0].numpy() == expected_label
        finally:
            os.unlink(csv_path)

    def test_add_data_from_csv_append_twice(self):
        """Append CSV data twice to verify data accumulates correctly."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_path = f.name
        try:
            _make_csv_with_paths(csv_path, num_samples=3)

            ds = muller.dataset(path=CSV_DATASET_PATH, overwrite=True)
            ds.create_tensor("image_path", htype="image", sample_compression="jpeg")
            ds.create_tensor("label", htype="text", sample_compression="lz4")

            path_columns = {"image_path": "read"}
            ds.add_data_from_csv(csv_path=csv_path, path_columns=path_columns, workers=0)
            assert len(ds) == 3

            ds.add_data_from_csv(csv_path=csv_path, path_columns=path_columns, workers=0)
            assert len(ds) == 6
        finally:
            os.unlink(csv_path)


class TestFromCsvErrorHandling:
    """Test error handling for CSV import."""

    def setup_method(self):
        _cleanup(CSV_DATASET_PATH)

    def teardown_method(self):
        _cleanup(CSV_DATASET_PATH)

    def test_from_csv_missing_file(self):
        """Should raise ValueError for non-existent CSV file."""
        with pytest.raises(ValueError):
            muller.from_csv(
                csv_path="/nonexistent/path/to/file.csv",
                muller_path=CSV_DATASET_PATH,
            )

    def test_from_csv_empty_path(self):
        """Should raise ValueError when csv_path is empty."""
        with pytest.raises(ValueError, match="csv_path and muller_path cannot be empty"):
            muller.from_csv(csv_path="", muller_path=CSV_DATASET_PATH)

    def test_add_data_from_csv_empty_path(self):
        """Should raise ValueError when csv_path is empty on instance method."""
        ds = muller.dataset(path=CSV_DATASET_PATH, overwrite=True)
        ds.create_tensor("col1", htype="text")
        with pytest.raises(ValueError, match="csv_path cannot be empty"):
            ds.add_data_from_csv(csv_path="")

    def test_add_data_from_csv_mismatched_columns(self):
        """Should raise ValueError when CSV columns don't match dataset tensors."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_path = f.name
        try:
            _make_csv_text_only(csv_path, num_samples=3)

            ds = muller.dataset(path=CSV_DATASET_PATH, overwrite=True)
            ds.create_tensor("col_a", htype="text")
            ds.create_tensor("col_b", htype="text")

            with pytest.raises(ValueError, match="do not match"):
                ds.add_data_from_csv(csv_path=csv_path, workers=0)
        finally:
            os.unlink(csv_path)

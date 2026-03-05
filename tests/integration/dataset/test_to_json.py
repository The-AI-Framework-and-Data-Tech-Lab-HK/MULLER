# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import os
import posixpath
import pytest

import muller
from muller.util.exceptions import InvalidJsonFileName, InvalidNumWorkers
from tests.constants import SAMPLE_FILES, TEST_JSON_PATH, TEST_JSON_FILE, TEST_JSONL_FILE
from tests.utils import official_path, official_creds


def create_dataset(storage):
    """ Create dataset """
    values = ["白日依山尽，黄河入海流，欲穷千里目，更上一层楼", "窗前明月光，疑是地上霜，举头邀明月，低头思故乡",
              "真正的勇士，敢于直面惨淡的人生，敢于正视淋漓的鲜血这是怎样的哀痛者和幸福者？",
              "All happy families are happy alike, all unhappy families are unhappy in their own way."]
    ds = muller.empty(official_path(storage, TEST_JSON_PATH), creds=official_creds(storage), overwrite=True)
    with ds:
        ds.create_tensor('value', htype='text')
        ds.create_tensor('label', htype='generic', dtype='int')
        ds.create_tensor('mul_values', htype='list')

    with ds:
        for i in range(0, 8):
            ds.value.append(values[i % 4])
            ds.label.append(i % 4)
            if i // 4 == 0:
                ds.mul_values.append([values[i % 4]])
            else:
                ds.mul_values.append([values[i % 4], values[(i + 1) % 4]])

    return ds


def test_to_full_jsonl(storage):
    """ Test the feasibility of exporting a full dataset to a specified jsonl file. """
    ds = create_dataset(storage)
    ds.to_json(TEST_JSONL_FILE)
    with open(TEST_JSONL_FILE) as f:
        data = f.read()

    with open(posixpath.join(os.getcwd(), SAMPLE_FILES["expected_jsonl"])) as f:
        expected_data = f.read()

    assert expected_data == data


def test_to_full_json(storage):
    """ Test the feasibility of exporting a full dataset to a specified json file. """
    ds = create_dataset(storage)
    ds.to_json(TEST_JSON_FILE)
    with open(TEST_JSON_FILE) as f:
        data = f.read()

    with open(posixpath.join(os.getcwd(), SAMPLE_FILES["expected_json"])) as f:
        expected_data = f.read()

    assert expected_data == data


def test_to_partial_jsonl_mul_workers(storage):
    """ Test the feasibility of exporting a partial dataset (with tensors selected) to a jsonl file,
        with specified num of multiple workers.
    """
    ds = create_dataset(storage)
    ds.to_json(TEST_JSONL_FILE, tensors=["value", "label"], num_workers=3)
    with open(TEST_JSONL_FILE) as f:
        data = f.read()
    with open(posixpath.join(os.getcwd(), SAMPLE_FILES["expected2_jsonl"])) as f:
        expected_data = f.read()

    assert expected_data == data


def test_to_partial_json_mul_workers(storage):
    """ Test the feasibility of exporting a partial dataset (with tensors selected) to a json file,
        with specified num of multiple workers.
    """
    ds = create_dataset(storage)
    ds.to_json(TEST_JSON_FILE, tensors=["value", "label"], num_workers=5)
    with open(TEST_JSON_FILE) as f:
        data = f.read()
    with open(posixpath.join(os.getcwd(), SAMPLE_FILES["expected2_json"])) as f:
        expected_data = f.read()

    assert expected_data == data


def test_invalid_json_name(storage):
    """ Validate that illegal parameters with lead to exceptions. """
    ds = create_dataset(storage)
    try:
        ds.to_json("data/test.txt")
        assert False, "No exception raises"
    except InvalidJsonFileName as e:
        assert True, f"Raises InvalidJsonFileName {e}"

    try:
        ds.to_json("data/test.csv")
        assert False, "No exception raises"
    except InvalidJsonFileName as e:
        assert True, f"Raises InvalidJsonFileName {e}"


def test_invalid_num_workers(storage):
    """ Validate that illegal parameters with lead to exceptions. """
    ds = create_dataset(storage)
    try:
        ds.to_json(TEST_JSON_FILE, num_workers=-1)
        assert False, "No exception raises"
    except InvalidNumWorkers as e:
        assert True, f"Raises InvalidNumWorkers {e}"

    try:
        ds.to_json(TEST_JSON_FILE, num_workers=0)
        assert False, "No exception raises"
    except InvalidNumWorkers as e:
        assert True, f"Raises InvalidNumWorkers {e}"

    try:
        ds.to_json(TEST_JSON_FILE, num_workers=1.5)
        assert False, "No exception raises"
    except TypeError as e:
        assert True, f"Raises TypeError {e}"


if __name__ == '__main__':
    pytest.main(["-s", "test_to_json.py"])

# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

from time import time
import logging

import numpy as np
import pytest

import muller
from muller.util.exceptions import (FilterVectorizedConditionError,
                                   FilterVectorizedConnectorListError,
                                   FilterOperatorNegationUnsupportedError,
                                   InvertedIndexUnsupportedError,
                                   InvertedIndexNotExistsError)
from tests.constants import TEST_FILTER_VECTORIZED_PATH
from tests.utils import official_path, official_creds

logging.basicConfig(level=logging.INFO)


def test_generic_vectorized_filter_1(storage):
    """ A test case of tensor column of generic dtype, using tensor.extend() to append data."""
    ds = muller.dataset(path=official_path(storage, TEST_FILTER_VECTORIZED_PATH),
                       creds=official_creds(storage), overwrite=True)
    # You do not need to specify sample_compression="lz4" if there is no requirement of
    # compression to save storage space. It it much faster to query data without compression.
    ds.create_tensor(name="test", htype="generic")
    ds.test.extend(np.random.randint(5, size=10000))

    t0 = time()
    ds_filter_1 = ds.filter_vectorized([("test", ">=", 2)])
    t1 = time()
    ds_filter_2 = ds.filter(lambda sample: sample["test"].data()['value'] >= 2)
    t2 = time()
    logging.info(f'Function filter_vectorized executed in {(t1 - t0):.8f}s')
    logging.info(f'Function filter_normal executed in {(t2 - t1):.8f}s')
    assert len(ds_filter_1) == len(ds_filter_2)


def test_generic_vectorized_filter_2(storage):
    """ A test case of tensor column of generic dtype, using tensor.append() to append data."""
    ds = muller.dataset(path=official_path(storage, TEST_FILTER_VECTORIZED_PATH),
                       creds=official_creds(storage), overwrite=True)
    ds.create_tensor(name="test", htype="generic")
    count = 1
    with ds:
        while count < 5000:
            ds.test.append(1)
            ds.test.append(2)
            count += 1

    t0 = time()
    ds_filter_1 = ds.filter_vectorized([("test", ">=", 2)])
    t1 = time()
    ds_filter_2 = ds.filter(lambda sample: sample["test"].data()['value'] >= 2)
    t2 = time()
    logging.info(f'Function filter_vectorized executed in {(t1 - t0):.8f}s')
    logging.info(f'Function filter_normal executed in {(t2 - t1):.8f}s')
    assert len(ds_filter_1) == len(ds_filter_2)


def test_text_vectorized_filter(storage):
    """ A test case of tensor column of text dtype, using tensor.extend() to append data."""
    ds = muller.dataset(path=official_path(storage, TEST_FILTER_VECTORIZED_PATH),
                       creds=official_creds(storage), overwrite=True)
    ds.create_tensor(name="test", htype="text")
    count = 1
    with ds:
        while count < 2000:
            ds.test.append("hi")
            ds.test.append("bye")
            ds.test.append("oops")
            ds.test.append("hello")
            ds.test.append("world")
            count += 1

    t0 = time()
    ds_filter_1 = ds.filter_vectorized([("test", "==", "hi")])
    t1 = time()
    ds_filter_2 = ds.filter(lambda sample: sample["test"].data()['value'] == "hi")
    t2 = time()
    logging.info(f'Function filter_vectorized executed in {(t1 - t0):.8f}s')
    logging.info(f'Function filter_normal executed in {(t2 - t1):.8f}s')
    assert len(ds_filter_1) == len(ds_filter_2)


def test_generic_vectorized_filter_with_connector_1(storage):
    """ A test case of tensor column of generic dtype, using tensor.extend() to append data, with AND condition."""
    ds = muller.dataset(path=official_path(storage, TEST_FILTER_VECTORIZED_PATH),
                       creds=official_creds(storage), overwrite=True)
    ds.create_tensor(name="test", htype="generic")
    ds.test.extend(np.random.randint(5, size=10000))
    ds.create_tensor(name="test2", htype="generic")
    ds.test2.extend(np.random.randint(100, size=10000))

    t0 = time()
    ds_1 = ds.filter_vectorized([("test", ">", 2), ("test", "<=", 4), ("test2", "<", 60, False, "NOT")], ["AND", "OR"])
    t1 = time()
    ds_2 = ds.filter(lambda sample: 2 < sample.test.data()["value"] <= 4 or not sample.test2.data()["value"] < 60)
    t2 = time()
    logging.info(f'Function filter_vectorized executed in {(t1 - t0):.8f}s')
    logging.info(f'Function filter_normal executed in {(t2 - t1):.8f}s')
    assert len(ds_1) == len(ds_2)


def test_generic_vectorized_filter_with_connector_2(storage):
    """ A test case of tensor column of generic dtype, using tensor.extend() to append data, with AND condition."""
    ds = muller.dataset(path=official_path(storage, TEST_FILTER_VECTORIZED_PATH),
                       creds=official_creds(storage), overwrite=True)
    ds.create_tensor(name="test", htype="generic")
    ds.test.extend(np.random.randint(100, size=10000))

    t0 = time()
    ds_filter_1 = ds.filter_vectorized([("test", ">", 50, False, "NOT"), ("test", ">=", 20)], ["AND"])
    t1 = time()
    ds_filter_2 = ds.filter(lambda sample: not (sample.test.data()["value"] > 50) and sample.test.data()["value"] >= 20)
    t2 = time()
    logging.info(f'Function filter_vectorized executed in {(t1 - t0):.8f}s')
    logging.info(f'Function \'filter_normal\' executed in {(t2 - t1):.8f}s')
    assert len(ds_filter_1) == len(ds_filter_2)


def test_with_limit_and_offset(storage):
    """ A test case of filter function with limit and offset."""
    ds = muller.dataset(path=official_path(storage, TEST_FILTER_VECTORIZED_PATH),
                       creds=official_creds(storage), overwrite=True)
    ds.create_tensor(name="test", htype="generic")
    ds.test.extend(np.arange(0, 100))

    ds_filter_1 = ds.filter_vectorized([("test", ">", 50), ("test", ">=", 20)], ["AND"], offset=60, limit=10)
    assert ds_filter_1.filtered_index == list(np.arange(60, 70))

    ds_filter_2 = ds.filter_vectorized([("test", ">", 50), ("test", ">=", 20)], ["AND"],
                                       offset=ds_filter_1.filtered_index[-1] + 1, limit=10)
    assert ds_filter_2.filtered_index == list(np.arange(70, 80))

    ds_filter_3 = ds.filter_vectorized([("test", ">", 50), ("test", ">=", 20)], ["AND"], offset=0, limit=10)
    assert ds_filter_3.filtered_index == list(np.arange(51, 61))


def test_generic_inverted_index_exact_match(storage):
    """ A test case of filter function with inverted index, exact match. """
    ds = muller.dataset(path=official_path(storage, TEST_FILTER_VECTORIZED_PATH),
                       creds=official_creds(storage), overwrite=True)
    ds.create_tensor(name="test", htype="generic")

    ds.test.extend(list(range(0, 100)))
    ds.commit()

    ds.create_index(["test"])
    ds_1 = ds.filter_vectorized([("test", ">", 50, False), ("test", "==", 70, True)], ["AND"], use_local_index=False)
    assert ds_1.filtered_index == [70]

    ds_2 = ds.filter_vectorized([("test", ">", 50, False), ("test", "==", 20, True)], ["OR"], use_local_index=False)
    temp = [20]
    temp.extend(list(np.arange(51, 100)))
    assert ds_2.filtered_index == temp

    ds_3 = ds.filter_vectorized([("test", ">", 50, False), ("test", "==", 70, True)], ["AND"], offset=60,
                                use_local_index=False)
    assert ds_3.filtered_index == [70]

    ds_4 = ds.filter_vectorized([("test", ">", 50, False), ("test", "==", 20, True)], ["OR"], limit=10,
                                use_local_index=False)
    temp = [20]
    temp.extend(list(np.arange(51, 60)))
    assert ds_4.filtered_index == temp

    ds_5 = ds.filter_vectorized([("test", "BETWEEN", [13, 15])], use_local_index=False)
    assert ds_5.filtered_index == [13, 14, 15]


def test_regular_expression(storage):
    """ A test case of filter function with regular expressions. """
    ds = muller.dataset(path=official_path(storage, TEST_FILTER_VECTORIZED_PATH),
                       creds=official_creds(storage), overwrite=True)
    ds.create_tensor(name="test", htype="text")
    ds.test.extend(['A0', 'A1', 'A2', 'A3', 'A4', 'B0', 'B1', 'C0'])

    ds_1 = ds.filter_vectorized([("test", "LIKE", "A[0-2]")])
    assert ds_1.filtered_index == [0, 1, 2]

    ds_2 = ds.filter_vectorized([("test", "LIKE", "B")])
    assert ds_2.filtered_index == [5, 6]

    ds_3 = ds.filter_vectorized([("test", "LIKE", "[a-zA-Z]+0")])
    assert ds_3.filtered_index == [0, 5, 7]

    ds_4 = ds.filter_vectorized([("test", "LIKE", "A[0-2]", False, "NOT")])
    assert ds_4.filtered_index == [3, 4, 5, 6, 7]


def test_cache(storage):
    """ A test case of evaluating the correctness of cache in filter. """
    ds = muller.dataset(path=official_path(storage, TEST_FILTER_VECTORIZED_PATH),
                       creds=official_creds(storage), overwrite=True)
    ds.create_tensor(name="test", htype="generic")

    ds.test.extend(list(range(0, 100)))
    ds_1 = ds.filter_vectorized([("test", ">", 50)])
    assert ds.append_only is True
    assert ds_1.filtered_index == list(np.arange(51, 100))

    # Conduct the same query again.
    ds_2 = ds.filter_vectorized([("test", ">", 50)])
    assert ds.append_only is True
    assert ds_2.filtered_index == list(np.arange(51, 100))

    # Conduct the same subquery again.
    ds_3 = ds.filter_vectorized([("test", ">", 50), ("test", "<=", 90)], ["AND"], offset=60, limit=10)
    assert ds.append_only is True
    assert ds_3.filtered_index == list(np.arange(60, 70))

    # In non pop-only scenario, we will not use the cache again. Instead, we recompute the result.
    # Case 1: update
    ds[51].update({"test": 0})
    ds_4 = ds.filter_vectorized([("test", ">", 50)])
    assert ds.append_only is False
    assert ds_4.filtered_index == list(np.arange(52, 100))

    # Case 2: update - another method
    ds.test[52] = 0
    ds_5 = ds.filter_vectorized([("test", ">", 50)])
    assert ds.append_only is False
    assert ds_5.filtered_index == list(np.arange(53, 100))

    # Case 3: pop
    ds.pop([53])
    ds_6 = ds.filter_vectorized([("test", ">", 50)])
    assert ds.append_only is False
    assert ds_6.filtered_index == list(np.arange(53, 99))


def test_filter_with_different_branches(storage):
    """ A test case that checks the correctness when filtering with different branches. """
    ds = muller.dataset(path=official_path(storage, TEST_FILTER_VECTORIZED_PATH),
                       creds=official_creds(storage), overwrite=True)

    ds.create_tensor('labels', htype='generic', dtype='int')
    ds.create_tensor('categories', htype='text')
    ds.labels.extend([10, 2, 30, 4, 50, 6, 70, 8, 90, 100] * 2)
    ds.categories.extend(['agent', '情感', '生成', '写作', '情感', 'agent', '生成', '写作', '情感', '写作'] * 2)

    # Checkout to branch dev, append data, and query
    ds.checkout('dev', create=True)
    ds.labels.extend([10, 2, 30, 4, 50, 6, 70, 8, 90, 100] * 2)
    ds.categories.extend(['agent2', '情感2', '生成2', '写作2', '情感2', 'agent2', '生成2', '写作2', '情感2', '写作2'] * 2)
    assert ds.append_only is True
    ds_1 = ds.filter_vectorized([("labels", ">", 50, False), ("categories", "==", '写作2', False)], ["OR"])
    assert len(ds_1.filtered_index) == 16

    # Checkout to branch dev-2, append data, and query
    ds.checkout('dev-2', create=True)
    ds.labels.extend([10, 2, 30, 4, 50, 6, 70, 8, 90, 100] * 2)
    ds.categories.extend(
        ['agent2', '情感2', '生成2', '写作2', '情感2', 'agent2', '生成2', '写作2', '情感2', '写作2'] * 2)
    ds.commit()
    assert ds.append_only is True
    ds_5 = ds.filter_vectorized([("labels", ">", 50, False), ("categories", "==", '写作2', False)], ["OR"])
    assert len(ds_5.filtered_index) == 26

    # Append data and query
    ds.labels.extend([10, 2, 30, 4, 50, 6, 70, 8, 90, 100] * 2)
    ds.categories.extend(
        ['agent2', '情感2', '生成2', '写作2', '情感2', 'agent2', '生成2', '写作2', '情感2', '写作2'] * 2)
    ds.commit()
    assert ds.append_only is True
    ds_6 = ds.filter_vectorized([("labels", ">", 50, False), ("categories", "==", '写作2', False)], ["OR"])
    assert len(ds_6.filtered_index) == 36

    # Modify data and query (via the first method of data update)
    ds[78].update({"labels": 0})
    assert ds.append_only is False
    ds_7 = ds.filter_vectorized([("labels", ">", 50, False), ("categories", "==", '写作2', False)], ["OR"])
    assert len(ds_7.filtered_index) == 35

    # Modify data and query (via the second method of data update)
    ds.labels[79] = 0
    ds.categories[79] = "写作1"
    assert ds.append_only is False
    ds_8 = ds.filter_vectorized([("labels", ">", 50, False), ("categories", "==", '写作2', False)], ["OR"])
    assert len(ds_8.filtered_index) == 34

    # Delete data and query
    ds.pop([63, 66, 67])
    assert ds.append_only is False
    ds_9 = ds.filter_vectorized([("labels", ">", 50, False), ("categories", "==", '写作2', False)], ["OR"])
    assert len(ds_9.filtered_index) == 31

    # Checkout to the main branch and query
    ds.checkout('main')
    assert ds.append_only is False
    ds_2 = ds.filter_vectorized([("labels", ">", 50, False), ("categories", "==", '写作2', False)], ["OR"])
    assert len(ds_2.filtered_index) == 6

    # Checkout to the dev branch, delete data, and query
    ds.checkout('dev')
    ds.pop([5, 6, 7, 8, 9])
    assert ds.append_only is False
    ds_3 = ds.filter_vectorized([("labels", ">", 50, False), ("categories", "==", '写作2', False)], ["OR"])
    assert len(ds_3.filtered_index) == 13

    # Checkout to the main branch, and query
    ds.checkout('main')
    assert ds.append_only is False
    ds_4 = ds.filter_vectorized([("labels", ">", 50, False), ("categories", "==", '写作2', False)], ["OR"])
    assert len(ds_4.filtered_index) == 6


def test_exception(storage):
    """ A test case that checks if exceptions are correctly raised. """
    ds = muller.dataset(path=official_path(storage, TEST_FILTER_VECTORIZED_PATH),
                       creds=official_creds(storage), overwrite=True)
    ds.create_tensor(name="test1", htype="text")
    ds.test1.extend(['A0', 'A1', 'A2', 'A3', 'A4', 'B0', 'B1', 'C0'])
    ds.create_tensor(name="test2", htype="generic")
    ds.test2.extend(list(range(0, 8)))
    ds.create_tensor(name="test3", htype="generic")
    ds.test3.extend([True, True, True, True, False, False, True, False])
    ds.summary()

    try:
        ds.filter_vectorized([("test1", "LIKE", 1)])
        assert False, "No exception raises"
    except FilterVectorizedConditionError as e:
        assert True, f"Filter values caused exception {e}"

    try:
        ds.filter_vectorized([("test2", ">", "5")])
        assert False, "No exception raises"
    except FilterVectorizedConditionError as e:
        assert True, f"Filter values caused exception {e}"

    ds_1 = ds.filter_vectorized([("test2", ">", 5.1)])
    assert ds_1.filtered_index == [6, 7]

    ds_2 = ds.filter_vectorized([("test3", "==", True, False, None)])
    assert ds_2.filtered_index == [0, 1, 2, 3, 6]

    try:
        ds.filter_vectorized([("test1", "==", [])])
        assert False, "No exception raises"
    except FilterVectorizedConditionError as e:
        assert True, f"Filter values caused exception {e}"

    try:  # Before creating inverted index
        ds.filter_vectorized([("test1", "CONTAINS", "A", True)])
        assert False, "No exception raises"
    except InvertedIndexNotExistsError as e:
        assert True, f"Filter values caused exception {e}"

    ds.commit()
    ds.create_index(["test1", "test2"])
    try:  # Use inverted index in non-string columns
        ds.filter_vectorized([("test2", "CONTAINS", 1)])
        assert False, "No exception raises"
    except FilterVectorizedConditionError as e:
        assert True, f"Filter values caused exception {e}"

    try:  # Use the keyword NOT with CONTAINS
        ds.filter_vectorized([("test1", "CONTAINS", "A", True, "NOT")])
        assert False, "No exception raises"
    except FilterOperatorNegationUnsupportedError as e:
        assert True, f"Filter values caused exception {e}"

    try:  # User the keyword NOT with ==, !=, and inverted index
        ds.filter_vectorized([("test2", "==", 1, True, "NOT")])
        assert False, "No exception raises"
    except FilterOperatorNegationUnsupportedError as e:
        assert True, f"Filter values caused exception {e}"

    try:  # Use the keyword BETWEEN on string-typed columns
        ds.filter_vectorized([("test1", "BETWEEN", ["A" "B"], True)])
        assert False, "No exception raises"
    except FilterVectorizedConditionError as e:
        assert True, f"Filter values caused exception {e}"

    try:  # Use the keyword BETWEEN together with NOT
        ds.filter_vectorized([("test2", "BETWEEN", [1, 2], True, "NOT")])
        assert False, "No exception raises"
    except FilterOperatorNegationUnsupportedError as e:
        assert True, f"Filter values caused exception {e}"

    try:  # The connector list
        ds.filter_vectorized([("test2", "BETWEEN", [1, 2], True, "NOT")], ["AND"])
        assert False, "No exception raises"
    except FilterVectorizedConnectorListError as e:
        assert True, f"Filter values caused exception {e}"

    try: # Using illegal operator with inverted index
        ds.filter_vectorized([("test2", ">", 1, True)])
        assert False, "No exception raises"
    except InvertedIndexUnsupportedError as e:
        assert True, f"Filter values caused exception {e}"

    try: # The length of condition list is illegal
        ds.filter_vectorized([("test2", ">", 1, True, False, False)])
        assert False, "No exception raises"
    except FilterVectorizedConditionError as e:
        assert True, f"Filter values caused exception {e}"

if __name__ == '__main__':
    pytest.main(["-s", "test_filter_vectorized.py"])

# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin
#
import numpy as np
import pytest

import muller
from muller.util.exceptions import FilterOperatorNegationUnsupportedError
from muller.util.sensitive_config import SensitiveConfig
from tests.constants import TEST_INDEX_PATH
from tests.utils import official_path, official_creds


def test_inverted_index(storage):
    ds = muller.empty(official_path(storage, TEST_INDEX_PATH), creds=official_creds(storage), overwrite=True)
    values = ["白日依山尽，黄河入海流，欲穷千里目，更上一层楼",
              "床前明月光，疑是地上霜，举头邀明月，低头思故乡",
              "京口瓜洲一水间，钟山只隔数重山。 春风又绿江南岸，明月何时照我还？",
              "真正的勇士，敢于直面惨淡的人生，敢于正视淋漓的鲜血。这是怎样的哀痛者和幸福者？",
              "All happy families are happy alike, all unhappy families are unhappy in their own way."]

    with ds:
        # Create the tensors with names of your choice.
        ds.create_tensor('value', htype='text')
        ds.create_tensor('label', htype='generic', dtype='int')

    with ds:
        ds.value.extend(values * 2)
        ds.label.extend(list(np.arange(0, 5)) * 2)

    ds.commit()

    # 1. Normal query
    ds.create_index(['value'], use_uuid=True)

    ds_test_0 = ds.filter_vectorized([("label", "==", 1)], use_local_index=False)
    assert ds_test_0.filtered_index == [1, 6]

    ds_test_1 = ds.filter_vectorized([("value", "CONTAINS", "明月")], use_local_index=False)
    assert ds_test_1.filtered_index == [1, 2, 6, 7]

    ds_test_2 = ds.filter_vectorized([("label", "==", 1), ("value", "CONTAINS", "明月")], ["AND"],
                                     use_local_index=False)
    assert ds_test_2.filtered_index == [1, 6]

    ds_test_3 = ds.filter_vectorized([("label", "==", 1), ("value", "CONTAINS", "明月")], ["OR"], use_local_index=False)
    assert ds_test_3.filtered_index == [1, 2, 6, 7]

    ds_test_4 = ds.filter_vectorized([("value", "CONTAINS", "明月"), ("value", "CONTAINS", "春风")], ["AND"],
                                     use_local_index=False)
    assert ds_test_4.filtered_index == [2, 7]

    ds_test_5 = ds.filter_vectorized([("label", "==", 1), ("value", "CONTAINS", "明月"), ("value", "CONTAINS", "春风")],
                                     ["OR", "AND"],
                                     use_local_index=False)
    assert ds_test_5.filtered_index == [2, 7]

    # Note: de_test_6 is different from de_test_5 because of the different execution sequence of AND, OR.
    ds_test_6 = ds.filter_vectorized([("value", "CONTAINS", "明月"), ("value", "CONTAINS", "春风"), ("label", "==", 1)],
                                     ["AND", "OR"],
                                     use_local_index=False)
    assert ds_test_6.filtered_index == [1, 2, 6, 7]

    # 2. Query with version. How to update the data?
    ds[1].update({"value": "update data", "label": 1})
    ds.pop([0, 4])
    with ds:
        ds.value.append("add data")
        ds.label.append(-1)

    ds.commit()
    ds.create_index(['value'], use_uuid=True)

    ds_test_7 = ds.filter_vectorized([("value", "CONTAINS", "春风")],
                                     use_local_index=False)
    assert ds_test_7.filtered_index == [1, 5]

    ds_test_8 = ds.filter_vectorized([("value", "CONTAINS", "update")],
                                     use_local_index=False)
    assert ds_test_8.filtered_index == [0]

    ds_test_9 = ds.filter_vectorized([("value", "CONTAINS", "明月"), ("value", "CONTAINS", "春风")], ["OR"],
                                     use_local_index=False)
    assert ds_test_9.filtered_index == [1, 4, 5]

    # 3. Query with version. How to append the data?
    with ds:
        ds.value.append("白日依山尽，黄河入海流，欲穷千里目，更上一层楼")
        ds.label.append(0)
    ds.commit()

    ds.create_index(["value"], use_uuid=True)
    ds_test_10 = ds.filter_vectorized([("value", "CONTAINS", "黄河")],
                                      use_local_index=False)
    assert ds_test_10.filtered_index == [3, 9]

    # 4. Query with version. How to append, update, and pop the data?
    ds[1].update({"value": "update data", "label": 1})
    ds[2].update({"value": "update data", "label": 1})
    ds.commit()
    ds.pop(7)
    with ds:
        ds.value.append("add data")
        ds.label.append(-1)
    ds.commit()
    ds.commit(allow_empty=True)

    ds.create_index(['value', 'label'], use_uuid=True)
    ds_test_10 = ds.filter_vectorized([("value", "CONTAINS", "data")],
                                      use_local_index=False)
    assert ds_test_10.filtered_index == [0, 1, 2, 7, 9]

    indexed_tensors = ds.indexed_tensors
    assert set(indexed_tensors) == {'value', 'label'}

    ds_test_11 = ds.filter_vectorized([("label", "==", 1, True)],
                                      use_local_index=False)
    assert ds_test_11.filtered_index == [0, 1, 2, 4]

    ds_test_12 = ds.filter_vectorized([("label", "==", 1, True), ("value", "CONTAINS", "data")], ["AND"],
                                      use_local_index=False)
    assert ds_test_12.filtered_index == [0, 1, 2]

    ds_test_13 = ds.filter_vectorized([("label", "BETWEEN", [1, 3])],
                                      use_local_index=False)
    assert ds_test_13.filtered_index == [0, 1, 2, 4, 5, 6]

    ds_test_14 = ds.filter_vectorized([("label", "BETWEEN", [1, 3]), ("value", "CONTAINS", "data")], ["OR"],
                                      use_local_index=False)
    assert ds_test_14.filtered_index == [0, 1, 2, 4, 5, 6, 7, 9]

    try:
        ds.filter_vectorized([("label", "==", 1, True, "NOT")],
                             use_local_index=False)
        assert False, "No exception raises"
    except FilterOperatorNegationUnsupportedError as e:
        assert True, f"Filter types caused exception {e}"

    # 5. limit and offset
    ds_test_16 = ds.filter_vectorized([("value", "CONTAINS", "data")], offset=4, limit=1,
                                      use_local_index=False)
    assert ds_test_16.filtered_index == [7]

    ds_test_161 = ds.filter_vectorized([("label", ">=", 1)], offset=4, limit=10,
                                       use_local_index=False)
    assert ds_test_161.filtered_index == [4, 5, 6]

    ds_test_162 = ds.filter_vectorized([("value", "CONTAINS", "明月")], offset=4, limit=10,
                                       use_local_index=False)
    assert ds_test_162.filtered_index == [4, 5]

    ds_test_17 = ds.filter_vectorized([("label", ">=", 1), ("value", "CONTAINS", "明月")], ["AND"], offset=4, limit=1,
                                      use_local_index=False)
    assert ds_test_17.filtered_index == [4]

    ds_test_18 = ds.filter_vectorized([("label", ">=", 1), ("value", "CONTAINS", "明月")], ["AND"],
                                      offset=ds_test_17.filtered_index[-1] + 1, limit=1,
                                      use_local_index=False)
    assert ds_test_18.filtered_index == [5]

    ds_test_19 = ds.filter_vectorized([("label", ">=", 1), ("value", "CONTAINS", "明月")],
                                      ["OR"], offset=4, limit=2,
                                      use_local_index=False)
    assert ds_test_19.filtered_index == [4, 5]

    ds_test_20 = ds.filter_vectorized([("label", ">=", 1), ("value", "CONTAINS", "明月")],
                                      ["OR"], offset=ds_test_19.filtered_index[-1] + 1, limit=2,
                                      use_local_index=False)
    assert ds_test_20.filtered_index == [6]

    # 6. The current commit version of dataset and index is not the same.
    ds[3].update({"value": "update data", "label": 3})
    ds.commit()
    ds_test_21 = ds.filter_vectorized([("label", "==", 3, True)],
                                      use_local_index=False)
    assert ds_test_21.filtered_index == [6]


def test_inverted_index_with_multi_user(storage):
    """test inverted index with multi user"""
    values = ["白日依山尽，黄河入海流，欲穷千里目，更上一层楼",
              "床前明月光，疑是地上霜，举头邀明月，低头思故乡",
              "京口瓜洲一水间，钟山只隔数重山。 春风又绿江南岸，明月何时照我还？",
              "真正的勇士，敢于直面惨淡的人生，敢于正视淋漓的鲜血。这是怎样的哀痛者和幸福者？",
              "All happy families are happy alike, all unhappy families are unhappy in their own way."]
    SensitiveConfig().uid = "A"
    ds = muller.dataset(path=official_path(storage, TEST_INDEX_PATH),
                       creds=official_creds(storage), overwrite=True)
    with ds:
        # Create the tensors with names of your choice.
        ds.create_tensor('value', htype='text')
        ds.create_tensor('label', htype='generic', dtype='int')
    with ds:
        ds.value.extend(values)
        ds.label.extend(list(np.arange(0, 5)))
    ds.commit()
    ds.create_index(['value'], use_uuid=True)
    ds_test_1 = ds.filter_vectorized([("value", "CONTAINS", "明月")], use_local_index=False)
    assert ds_test_1.filtered_index == [1, 2]
    with ds:
        ds.value.extend(values)
        ds.label.extend(list(np.arange(0, 5)))
    ds.commit()

    SensitiveConfig().uid = "B"
    ds = muller.load(path=official_path(storage, TEST_INDEX_PATH),
                    creds=official_creds(storage))
    ds_test_1 = ds.filter_vectorized([("value", "CONTAINS", "明月")], use_local_index=False)
    assert ds_test_1.filtered_index == [1, 2]

    SensitiveConfig().uid = "A"
    ds = muller.load(path=official_path(storage, TEST_INDEX_PATH),
                    creds=official_creds(storage))
    ds.create_index(['value'], use_uuid=True)
    ds_test_1 = ds.filter_vectorized([("value", "CONTAINS", "明月")], use_local_index=False)
    assert ds_test_1.filtered_index == [1, 2, 6, 7]


if __name__ == '__main__':
    pytest.main(["-s", "test_inverted_index.py"])

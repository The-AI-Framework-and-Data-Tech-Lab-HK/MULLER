# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Bingyu Liu

import logging
import os
import multiprocessing as mp

import numpy as np
import pytest
import muller
from muller.util.exceptions import FilterOperatorNegationUnsupportedError, InvertedIndexUnsupportedError, \
    InvertedIndexNotExistsError, FilterVectorizedConditionError, UnsupportedMethod, UpdateIndexFailError
from muller.util.sensitive_config import SensitiveConfig
from tests.constants import TEST_INDEX_PATH
from tests.utils import official_path, official_creds


def is_ci_environment():
    """Check if it is running in CI online."""
    return os.getenv('JENKINS_URL') is not None or os.getenv('CI') == 'true'


def get_test_params():
    """Dynamic return test parameters: Only return [False] in CI environment, otherwise return [True, False]"""
    if is_ci_environment():
        return [False]
    return [True, False]


@pytest.mark.skipif(is_ci_environment(),reason="Skip CI online, because it cannot use cpp yet.")
def test_cpp_python_mix(storage):
    ds = muller.empty(official_path(storage, TEST_INDEX_PATH), creds=official_creds(storage), overwrite=True)

    stop_words_list = ["data/stop_words/baidu_stopwords.txt",
                       "data/stop_words/cn_stopwords.txt",
                       "data/stop_words/hit_stopwords.txt",
                       "data/stop_words/scu_stopwords.txt",
                       "data/stop_words/common_stopwords.txt",
                       "data/stop_words/stopwords-en.txt", ]
    compulsory_words = "data/compulsory_words/compulsory_words.txt"
    values = ["白日依山尽，黄河入海流，欲穷千里目，更上一层楼",
              "床前明月光，疑是地上霜，举头邀明月，低头思故乡",
              "京口瓜洲一水间，钟山只隔数重山。 春风又绿江南岸，明月何时照我还？",
              "我是deepseek，迅雷不及掩耳盗铃儿响叮当仁不让世界充满爱之势!你是谁？",
              "All happy families are happy alike, all unhappy families are unhappy in their own way."]
    label = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    float_list = [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1, 0.1]
    str_list = ["abc", "体育", "cbc", "abc", "bbc", "cbc", "abc", "bbc", "cbc", "z"]
    bool_list = [True, True, True, True, False, False, True, True, True, True]
    with ds:
        # Create the tensors with names of your choice.
        ds.create_tensor('value', htype='text')
        ds.create_tensor('label', htype='generic', dtype='int')
        ds.create_tensor('float_list', htype='generic')
        ds.create_tensor('str_list', htype='text')
        ds.create_tensor('bool_list', htype='generic')

    with ds:
        ds.value.extend(values * 2)
        ds.label.extend(label)
        ds.float_list.extend(float_list)
        ds.str_list.extend(str_list)
        ds.bool_list.extend(bool_list)

    ds.commit()

    # You cannot create index for the same column, unless using force_create=True
    ds.create_index_vectorized("value", num_of_shards=2, max_workers=4,
                               stop_words_list=stop_words_list, compulsory_words=compulsory_words, use_cpp=False)

    try:
        ds.create_index_vectorized("value", num_of_shards=2, max_workers=4,
                                   stop_words_list=stop_words_list, compulsory_words=compulsory_words, use_cpp=True)
        assert False, "No exception raises"
    except UpdateIndexFailError as e:
        assert True, f"There is an exception {e}"

    ds.create_index_vectorized("value", num_of_shards=2, max_workers=4,
                               stop_words_list=stop_words_list, compulsory_words=compulsory_words, use_cpp=True,
                               force_create=True)

    # You may consider different index type for different columns
    ds.create_index_vectorized("label", index_type="exact_match", use_cpp=False)

    # The CPP modules is not supported for exact_match
    try:
        ds.create_index_vectorized("bool_list", index_type="exact_match", max_workers=3, num_of_batches=2, use_cpp=True)
        assert False, "No exception raises"
    except UnsupportedMethod as e:
        assert True, f"There is an exception {e}"

    ds.create_index_vectorized("bool_list", index_type="exact_match", max_workers=3, num_of_batches=2, use_cpp=False)

    # Hybrid search on index created by python and cpp
    ds_test_1 = ds.filter_vectorized([("value", "CONTAINS", "明月"), ("label", "==", 1, True)],
                                     ["AND"])
    assert ds_test_1.filtered_index == [1, 6]

    # Hybrid search on index created by python and cpp, with normal query
    ds_test_2 = ds.filter_vectorized([("value", "CONTAINS", "明月"), ("label", "==", 1, True), ("label", "==", 0)],
                                     ["AND", "OR"])
    assert ds_test_2.filtered_index == [0, 1, 5, 6]

    # Optimized exact_match: key0=exact_match, key1=fuzzy_match or in pre_res
    ds_test_3 = ds.filter_vectorized([("label", "==", 2, True), ("value", "CONTAINS", "京口")],
                                     ["AND"])
    assert ds_test_3.filtered_index == [2, 7]

    ds_test_4 = ds.filter_vectorized([("label", "==", 2, True), ("value", "CONTAINS", "京口"), ("label", "==", 0)],
                                     ["AND", "OR"])
    assert ds_test_4.filtered_index == [0, 2, 5, 7]

    ds_test_5 = ds.filter_vectorized([("label", "==", 2, True), ("bool_list", "==", True), ("label", "==", 0)],
                                     ["AND", "OR"])
    assert ds_test_5.filtered_index == [0, 2, 5, 7]

    # Optimized exact_match: exact_match!=0, and, key-1
    ds_test_6 = ds.filter_vectorized([("value", "CONTAINS", "京口"), ("bool_list", "==", True, True)],
                                     ["AND"])
    assert ds_test_6.filtered_index == [2, 7]

    ds_test_7 = ds.filter_vectorized([("value", "CONTAINS", "京口"), ("label", "==", 2),
                                      ("bool_list", "==", True, True)],
                                     ["AND", "AND"])
    assert ds_test_7.filtered_index == [2, 7]

    ds_test_8 = ds.filter_vectorized([("label", "==", 2), ("label", "==", 3), ("bool_list", "==", True, True)],
                                     ["OR", "AND"])
    assert ds_test_8.filtered_index == [2, 3, 7, 8]

    ds_test_9 = ds.filter_vectorized([("label", "==", 2), ("label", "==", 3), ("bool_list", "==", False, True)],
                                     ["OR", "AND"])
    assert ds_test_9.filtered_index == []

    # No optimize exact_match
    ds_test_10 = ds.filter_vectorized([("value", "CONTAINS", "京口"), ("label", "==", 2),
                                      ("bool_list", "==", True, True)],
                                     ["AND", "OR"])
    assert ds_test_10.filtered_index == [0, 1, 2, 3, 6, 7, 8, 9]



@pytest.mark.parametrize("use_cpp", get_test_params())
def test_inverted_index(storage, use_cpp):
    ds = muller.empty(official_path(storage, TEST_INDEX_PATH), creds=official_creds(storage), overwrite=True)

    orig_create = ds.create_index_vectorized

    def patched_create_index_vec(tensor_name, *args, **kwargs):
        if kwargs.get("index_type") == "exact_match":
            kwargs["use_cpp"] = False
        else:
            kwargs.setdefault("use_cpp", use_cpp)
        return orig_create(tensor_name, *args, **kwargs)

    ds.create_index_vectorized = patched_create_index_vec

    stop_words_list = ["data/stop_words/baidu_stopwords.txt",
                       "data/stop_words/cn_stopwords.txt",
                       "data/stop_words/hit_stopwords.txt",
                       "data/stop_words/scu_stopwords.txt",
                       "data/stop_words/common_stopwords.txt",
                       "data/stop_words/stopwords-en.txt", ]
    compulsory_words = "data/compulsory_words/compulsory_words.txt"
    values = ["白日依山尽，黄河入海流，欲穷千里目，更上一层楼",
              "床前明月光，疑是地上霜，举头邀明月，低头思故乡",
              "京口瓜洲一水间，钟山只隔数重山。 春风又绿江南岸，明月何时照我还？",
              "我是deepseek，迅雷不及掩耳盗铃儿响叮当仁不让世界充满爱之势!你是谁？",
              "All happy families are happy alike, all unhappy families are unhappy in their own way."]
    label = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    float_list = [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1, 0.1]
    str_list = ["abc", "体育", "cbc", "abc", "bbc", "cbc", "abc", "bbc", "cbc", "z"]
    bool_list = [True, True, True, True, False, False, True, True, True, True]
    with ds:
        # Create the tensors with names of your choice.
        ds.create_tensor('value', htype='text')
        ds.create_tensor('label', htype='generic', dtype='int')
        ds.create_tensor('float_list', htype='generic')
        ds.create_tensor('str_list', htype='text')
        ds.create_tensor('bool_list', htype='generic')

    with ds:
        ds.value.extend(values * 2)
        ds.label.extend(label)
        ds.float_list.extend(float_list)
        ds.str_list.extend(str_list)
        ds.bool_list.extend(bool_list)

    ds.commit()

    # 0. Conduct optimize index and query before creating index
    try:
        ds.optimize_index("label", max_workers=1)
        assert False, "No exception raises"
    except InvertedIndexNotExistsError as e:
        assert True, f"There is an exception {e}"

    try:
        ds.filter_vectorized([("value", "CONTAINS", "明月")])
        assert False, "No exception raises"
    except InvertedIndexNotExistsError as e:
        assert True, f"There is an exception {e}"

    try:
        ds.filter_vectorized([("label", "==", 1, True)])
        assert False, "No exception raises"
    except InvertedIndexNotExistsError as e:
        assert True, f"There is an exception {e}"

    # It should raise exception when the index type does not match the column type
    try:
        ds.filter_vectorized([("bool_list", "==", "hi", True)])
        assert False, "No exception raises"
    except FilterVectorizedConditionError as e:
        assert True, f"There is an exception {e}"

    try:
        ds.filter_vectorized([("str_list", "==", 1.0, True)])
        assert False, "No exception raises"
    except FilterVectorizedConditionError as e:
        assert True, f"There is an exception {e}"

    try:
        ds.filter_vectorized([("float_list", "==", "Hi", True)])
        assert False, "No exception raises"
    except FilterVectorizedConditionError as e:
        assert True, f"There is an exception {e}"

    # 1. Normal query (text)
    ds.create_index_vectorized("value", num_of_shards=2, max_workers=4,
                               stop_words_list=stop_words_list, compulsory_words=compulsory_words)
    ds_test_01 = ds.filter_vectorized([("value", "CONTAINS", "我是deepseek")])
    assert ds_test_01.filtered_index == [3, 8]

    ds_test_02 = ds.filter_vectorized([("value", "CONTAINS", "不让世界充满爱")])
    assert ds_test_02.filtered_index == [3, 8]

    ds_test_03 = ds.filter_vectorized([("value", "CONTAINS", "您好")])
    assert ds_test_03.filtered_index == []

    ds_test_04 = ds.filter_vectorized([("value", "CONTAINS", "？")])
    assert ds_test_04.filtered_index == []

    ds_test_1 = ds.filter_vectorized([("value", "CONTAINS", "明月")])
    assert ds_test_1.filtered_index == [1, 2, 6, 7]

    ds_test_2 = ds.filter_vectorized([("value", "CONTAINS", "明月"), ("value", "CONTAINS", "春风")],
                                     ["AND"])
    assert ds_test_2.filtered_index == [2, 7]

    # 2. Normal query (scalar)
    ds_test_3 = ds.filter_vectorized([("label", "==", 1)])
    assert ds_test_3.filtered_index == [1, 6]
    ds.create_index_vectorized("label", index_type="exact_match")
    ds_test_4 = ds.filter_vectorized([("label", "==", 1, True)])
    assert ds_test_4.filtered_index == ds_test_3.filtered_index
    ds_test_41 = ds.filter_vectorized([("label", "==", 100, True)])
    assert ds_test_41.filtered_index == []

    ds_test_3 = ds.filter_vectorized([("float_list", "==", 0.1)])
    assert ds_test_3.filtered_index == [0, 4, 8, 9]
    ds.create_index_vectorized("float_list", index_type="exact_match", num_of_shards=2)
    ds_test_4 = ds.filter_vectorized([("float_list", "==", 0.1, True)])
    assert ds_test_4.filtered_index == ds_test_3.filtered_index
    ds_test_41 = ds.filter_vectorized([("float_list", "==", 0.001, True)])
    assert ds_test_41.filtered_index == []

    ds_test_3 = ds.filter_vectorized([("str_list", "==", "bbc")])
    assert ds_test_3.filtered_index == [4, 7]
    ds.create_index_vectorized("str_list", index_type="exact_match", num_of_shards=2, max_workers=3, num_of_batches=4)
    ds_test_4 = ds.filter_vectorized([("str_list", "==", "bbc", True)])
    assert ds_test_4.filtered_index == [4, 7]
    ds_test_41 = ds.filter_vectorized([("str_list", "==", "mmc", True)])
    assert ds_test_41.filtered_index == []

    ds_test_3 = ds.filter_vectorized([("bool_list", "==", False)])
    assert ds_test_3.filtered_index == [4, 5]
    ds.create_index_vectorized("bool_list", index_type="exact_match", max_workers=3, num_of_batches=2)
    ds_test_4 = ds.filter_vectorized([("bool_list", "==", False, True)])
    assert ds_test_4.filtered_index == [4, 5]

    ds_test_5 = ds.filter_vectorized([("label", "==", 1, True), ("str_list", "==", "bbc", True)],
                                     ["AND"])
    assert ds_test_5.filtered_index == []

    ds_test_6 = ds.filter_vectorized([("label", "==", 1, True), ("float_list", "==", 0.1, True)],
                                     ["OR"])
    assert ds_test_6.filtered_index == [0, 1, 4, 6, 8, 9]

    ds_test_7 = ds.filter_vectorized([("float_list", "==", 0.1, True), ("str_list", "==", "bbc", True)],
                                     ["OR"])
    assert ds_test_7.filtered_index == [0, 4, 7, 8, 9]

    ds_test_8 = ds.filter_vectorized([("float_list", "==", 0.1, True), ("str_list", "==", "bbc", True),
                                      ("bool_list", "==", False, True)],
                                     ["OR", "AND"])
    assert ds_test_8.filtered_index == [4]

    # 3. Normal query（text+scalar）
    ds_test_9 = ds.filter_vectorized([("value", "CONTAINS", "明月"), ("label", "==", 1, True)],
                                     ["AND"])
    assert ds_test_9.filtered_index == [1, 6]

    ds_test_10 = ds.filter_vectorized([("label", "==", 1, True), ("value", "CONTAINS", "明月")], ["AND"])
    assert ds_test_10.filtered_index == [1, 6]

    ds_test_11 = ds.filter_vectorized([("label", "==", 1, True), ("value", "CONTAINS", "明月")], ["OR"])
    assert ds_test_11.filtered_index == [1, 2, 6, 7]

    ds_test_12 = ds.filter_vectorized([("value", "CONTAINS", "明月"), ("value", "CONTAINS", "春风"),
                                       ("str_list", "==", "bbc", True)],
                                      ["AND", "AND"])
    assert ds_test_12.filtered_index == [7]

    ds_test_13 = ds.filter_vectorized([("value", "CONTAINS", "明月"), ("value", "CONTAINS", "春风"),
                                       ("str_list", "==", "bbc", True)],
                                      ["AND", "OR"])
    assert ds_test_13.filtered_index == [2, 4, 7]

    ds_test_14 = ds.filter_vectorized(
        [("label", "==", 1, True), ("value", "CONTAINS", "明月"), ("value", "CONTAINS", "春风")],
        ["OR", "AND"])
    assert ds_test_14.filtered_index == [2, 7]

    # Note: de_test_15 is different from de_test_14 because of the different execution sequence of AND, OR.
    ds_test_15 = ds.filter_vectorized(
        [("value", "CONTAINS", "明月"), ("value", "CONTAINS", "春风"), ("label", "==", 1, True)],
        ["AND", "OR"])
    assert ds_test_15.filtered_index == [1, 2, 6, 7]

    # 4. Query with version. How to update and delete data? Need index reconstruction.
    ds[1].update({"value": "update data", "label": 1})
    ds.pop([0, 4])
    with ds:
        ds.value.append("add data")
        ds.label.append(-1)
        ds.float_list.append(-0.1)
        ds.bool_list.append(False)
        ds.str_list.append("ccf")

    ds.commit()

    ds.create_index_vectorized("value", stop_words_list=stop_words_list, compulsory_words=compulsory_words,
                               force_create=True)

    ds_test_17 = ds.filter_vectorized([("value", "CONTAINS", "春风")])
    assert ds_test_17.filtered_index == [1, 5]

    ds_test_18 = ds.filter_vectorized([("value", "CONTAINS", "update")])
    assert ds_test_18.filtered_index == [0]

    ds_test_19 = ds.filter_vectorized([("value", "CONTAINS", "明月"), ("value", "CONTAINS", "春风")], ["OR"])
    assert ds_test_19.filtered_index == [1, 4, 5]

    ds.create_index_vectorized("bool_list", index_type="exact_match", force_create=True)
    ds_test_20 = ds.filter_vectorized([("bool_list", "==", False, True)])
    assert ds_test_20.filtered_index == [3, 8]

    # 5. Query with version. How to append data?
    with ds:
        ds.value.append("白日依山尽，黄河入海流，欲穷千里目，更上一层楼")
        ds.label.append(-2)
        ds.float_list.append(-0.2)
        ds.bool_list.append(False)
        ds.str_list.append("cse")
    ds.commit()

    ds.create_index_vectorized("value", stop_words_list=stop_words_list, compulsory_words=compulsory_words,
                               force_create=True)
    ds.create_index_vectorized("bool_list", index_type="exact_match", force_create=True)

    ds_test_21 = ds.filter_vectorized([("value", "CONTAINS", "黄河")])
    assert ds_test_21.filtered_index == [3, 9]

    ds_test_22 = ds.filter_vectorized([("value", "CONTAINS", "黄河"), ("bool_list", "==", False, True)],
                                      ["OR"])
    assert ds_test_22.filtered_index == [3, 8, 9]

    # 6. Query with version. How to append, update and delete data?
    ds[1].update({"value": "update data", "label": 1})
    ds[2].update({"value": "update data", "label": 1})
    ds.commit()
    ds.pop(7)
    with ds:
        ds.value.append("add data 2")
        ds.label.append(-3)
        ds.float_list.append(-0.3)
        ds.bool_list.append(False)
        ds.str_list.append("cba")
    ds.commit()
    ds.commit(allow_empty=True)

    ds.create_index_vectorized("value", stop_words_list=stop_words_list, compulsory_words=compulsory_words,
                               force_create=True)
    ds.create_index_vectorized("label", index_type="exact_match", force_create=True)

    ds_test_23 = ds.filter_vectorized([("value", "CONTAINS", "data")])
    assert ds_test_23.filtered_index == [0, 1, 2, 7, 9]

    assert ds.indexed_tensors_vec == {'value', 'label'}

    ds_test_24 = ds.filter_vectorized([("label", "==", 1, True)])
    assert ds_test_24.filtered_index == [0, 1, 2, 4]

    ds_test_25 = ds.filter_vectorized([("label", "==", 1, True), ("value", "CONTAINS", "data")], ["AND"])
    assert ds_test_25.filtered_index == [0, 1, 2]

    try:
        ds.filter_vectorized([("label", "==", 1, True, "NOT")])
        assert False, "No exception raises"
    except FilterOperatorNegationUnsupportedError as e:
        assert True, f"Filter types caused exception {e}"

    try:
        ds.filter_vectorized([("label", ">=", 1, True)])
        assert False, "No exception raises"
    except InvertedIndexUnsupportedError as e:
        assert True, f"Filter types caused exception {e}"

    # 7. limit and offset:
    ds_test_26 = ds.filter_vectorized([("value", "CONTAINS", "data")], offset=4, limit=1)
    assert ds_test_26.filtered_index == [7]

    ds_test_261 = ds.filter_vectorized([("label", ">=", 1)], offset=4, limit=10)
    assert ds_test_261.filtered_index == [4, 5, 6]

    ds_test_262 = ds.filter_vectorized([("value", "CONTAINS", "明月")], offset=4, limit=10)
    assert ds_test_262.filtered_index == [4, 5]

    ds_test_27 = ds.filter_vectorized([("label", ">=", 1), ("value", "CONTAINS", "明月")], ["AND"],
                                      offset=4, limit=1)
    assert ds_test_27.filtered_index == [4]

    ds_test_28 = ds.filter_vectorized([("label", ">=", 1), ("value", "CONTAINS", "明月")], ["AND"],
                                      offset=ds_test_27.filtered_index[-1] + 1, limit=1)
    assert ds_test_28.filtered_index == [5]

    ds_test_29 = ds.filter_vectorized([("label", ">=", 1), ("value", "CONTAINS", "明月")],
                                      ["OR"], offset=4, limit=2)
    assert ds_test_29.filtered_index == [4, 5]

    ds_test_30 = ds.filter_vectorized([("label", ">=", 1), ("value", "CONTAINS", "明月")],
                                      ["OR"],
                                      offset=ds_test_19.filtered_index[-1] + 1, limit=2)
    assert ds_test_30.filtered_index == [6]

    ds_test_31 = ds.filter_vectorized([("label", "==", 1, True), ("value", "CONTAINS", "明月")], ["AND"],
                                      offset=4, limit=1)
    assert ds_test_31.filtered_index == [4]

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

    ds.create_index_vectorized("value")
    ds_test_1 = ds.filter_vectorized([("value", "CONTAINS", "明月")])
    assert ds_test_1.filtered_index == [1, 2]
    with ds:
        ds.value.extend(values)
        ds.label.extend(list(np.arange(0, 5)))
    ds.commit()

    SensitiveConfig().uid = "B"
    ds = muller.load(path=official_path(storage, TEST_INDEX_PATH),
                    creds=official_creds(storage))
    ds_test_1 = ds.filter_vectorized([("value", "CONTAINS", "明月")])
    assert ds_test_1.filtered_index == [1, 2]

    SensitiveConfig().uid = "A"
    ds = muller.load(path=official_path(storage, TEST_INDEX_PATH),
                    creds=official_creds(storage))
    ds.create_index_vectorized("value")
    ds_test_1 = ds.filter_vectorized([("value", "CONTAINS", "明月")])
    assert ds_test_1.filtered_index == [1, 2, 6, 7]


@pytest.mark.parametrize("use_cpp", get_test_params())
def test_show_progress(storage, caplog, use_cpp):
    values = ["白日依山尽，黄河入海流，欲穷千里目，更上一层楼",
              "床前明月光，疑是地上霜，举头邀明月，低头思故乡",
              "京口瓜洲一水间，钟山只隔数重山。 春风又绿江南岸，明月何时照我还？",
              "真正的勇士，敢于直面惨淡的人生，敢于正视淋漓的鲜血。这是怎样的哀痛者和幸福者？",
              "All happy families are happy alike, all unhappy families are unhappy in their own way."]
    ds = muller.dataset(path=official_path(storage, TEST_INDEX_PATH),
                       creds=official_creds(storage), overwrite=True)

    orig_create = ds.create_index_vectorized

    def patched_create_index_vec(tensor_name, *args, **kwargs):
        if kwargs.get("index_type") == "exact_match":
            kwargs["use_cpp"] = False
        else:
            kwargs.setdefault("use_cpp", use_cpp)
        return orig_create(tensor_name, *args, **kwargs)

    ds.create_index_vectorized = patched_create_index_vec

    with ds:
        ds.create_tensor('value', htype='text')
        ds.create_tensor('label', htype='generic', dtype='int')
    with ds:
        ds.value.extend(values)
        ds.label.extend(list(np.arange(0, 5)))
    ds.commit()

    ds.create_index_vectorized("value")

    with caplog.at_level(logging.INFO):
        ds.filter_vectorized([("value", "CONTAINS", "明月"), ("value", "CONTAINS", "春风"),
                             ("value", "CONTAINS", "黄河"), ("value", "CONTAINS", "故乡")],
                             ["OR", "OR", "OR"],
                             show_progress=True)
        assert "Computing the result of ('value', 'CONTAINS', '明月')" in caplog.text
        assert "Computing the result of ('value', 'CONTAINS', '故乡')" in caplog.text

        ds.filter_vectorized([("label", ">", 2), ("label", "<=", 4)], ["AND"], show_progress=True)
        assert "Computing the result of ('label', '>', 2, None)" in caplog.text


@pytest.mark.parametrize("use_cpp", get_test_params())
def test_complex_fuzzy_match(storage, use_cpp):
    values = ["白日依山尽，黄河入海流，欲穷千里目，更上一层楼",
              "床前明月光，疑是地上霜，举头邀明月，低头思故乡",
              "京口瓜洲一水间，钟山只隔数重山。 春风又绿江南岸，明月何时照我还？",
              "真正的勇士，敢于直面惨淡的人生，敢于正视淋漓的鲜血。这是怎样的哀痛者和幸福者？",
              "All happy families are happy alike, all unhappy families are unhappy in their own way."]
    ds = muller.dataset(path=official_path(storage, TEST_INDEX_PATH),
                       creds=official_creds(storage), overwrite=True)

    orig_create = ds.create_index_vectorized

    def patched_create_index_vec(tensor_name, *args, **kwargs):
        if kwargs.get("index_type") == "exact_match":
            kwargs["use_cpp"] = False
        else:
            kwargs.setdefault("use_cpp", use_cpp)
        return orig_create(tensor_name, *args, **kwargs)


    ds.create_index_vectorized = patched_create_index_vec

    with ds:
        ds.create_tensor('value', htype='text')
        ds.create_tensor('label', htype='generic', dtype='int')
    with ds:
        ds.value.extend(values)
        ds.label.extend(list(np.arange(0, 5)))
    ds.commit()

    ds.create_index_vectorized("value")

    ds_1 = ds.filter_vectorized([("value", "CONTAINS", "床前明月光"), ("value", "CONTAINS", "春风"),
                         ("value", "CONTAINS", "黄河"), ("value", "CONTAINS", "故乡")],
                         ["OR", "OR", "OR"],
                         )
    assert ds_1.filtered_index == [0, 1, 2]

    ds_2 = ds.filter_vectorized([("value", "CONTAINS", "床前明月光||春风||黄河||故乡")])
    assert ds_2.filtered_index == [0, 1, 2]

    ds_3 = ds.filter_vectorized([("value", "CONTAINS", "别墅"), ("value", "CONTAINS", "唱k"),
                                 ("value", "CONTAINS", "水池"), ("value", "CONTAINS", "银龙鱼")],
                                ["OR", "OR", "OR"],
                                )
    assert ds_3.filtered_index == []

    ds_4 = ds.filter_vectorized([("value", "CONTAINS", "别墅||唱k||水池||银龙鱼")])
    assert ds_4.filtered_index == []


def test_create_new_index_while_using_old_index(storage):
    values = ["白日依山尽，黄河入海流，欲穷千里目，更上一层楼",
              "床前明月光，疑是地上霜，举头邀明月，低头思故乡",
              "京口瓜洲一水间，钟山只隔数重山。 春风又绿江南岸，明月何时照我还？",
              "真正的勇士，敢于直面惨淡的人生，敢于正视淋漓的鲜血。这是怎样的哀痛者和幸福者？",
              "All happy families are happy alike, all unhappy families are unhappy in their own way."]
    ds = muller.dataset(path=official_path(storage, TEST_INDEX_PATH),
                       creds=official_creds(storage), overwrite=True)

    with ds:
        ds.create_tensor('value', htype='text')
        ds.value.extend(values)
    ds.commit()

    ds.create_index_vectorized("value")
    ds_0 = ds.filter_vectorized([("value", "CONTAINS", "床前明月光")])
    assert ds_0.filtered_index == [1]

    ds.value.extend(values * 100)
    ds.commit()

    def task1():
        ds.create_index_vectorized("value")

    def task2():
        ds_1 = ds.filter_vectorized([("value", "CONTAINS", "床前明月光")])
        assert ds_1.filtered_index == [1]

    ctx = mp.get_context('fork')
    p1 = ctx.Process(target=task1)
    p2 = ctx.Process(target=task2)
    p1.start()  # Start subprocess 1
    p2.start()  # Start subprocess 2
    # (but currently the index construction of task 1 may not be finished, so the query of task 2 may be on the old index)
    p1.join()  # Wait for subprocess 1 to be killed
    p2.join()  # Wair for subprocess 2 to be killed

    ds_2 = ds.filter_vectorized([("value", "CONTAINS", "床前明月光")])
    assert ds_2.filtered_index == [i * 5 + 1 for i in range(101)]

def _task_create_index(storage):
    ds = muller.load(path=official_path(storage, TEST_INDEX_PATH))
    ds.create_index_vectorized("value", use_cpp=True, force_create=True)

def _task_query(storage):
    ds = muller.load(path=official_path(storage, TEST_INDEX_PATH))
    result = ds.filter_vectorized([("value", "CONTAINS", "床前明月光")])
    assert result.filtered_index == [1]

def _generate_ds(storage):
    values = ["白日依山尽，黄河入海流，欲穷千里目，更上一层楼",
              "床前明月光，疑是地上霜，举头邀明月，低头思故乡",
              "京口瓜洲一水间，钟山只隔数重山。 春风又绿江南岸，明月何时照我还？",
              "真正的勇士，敢于直面惨淡的人生，敢于正视淋漓的鲜血。这是怎样的哀痛者和幸福者？",
              "All happy families are happy alike, all unhappy families are unhappy in their own way."]
    ds = muller.dataset(path=official_path(storage, TEST_INDEX_PATH),
                       creds=official_creds(storage), overwrite=True)

    with ds:
        ds.create_tensor('value', htype='text')
        ds.value.extend(values)
    ds.commit()

    ds.create_index_vectorized("value", use_cpp=True)
    ds_0 = ds.filter_vectorized([("value", "CONTAINS", "床前明月光")])
    assert ds_0.filtered_index == [1]

    ds.value.extend(values * 100)
    ds.commit()


@pytest.mark.skipif(is_ci_environment(),reason="Skip CI online, because it cannot use cpp yet.")
def test_create_new_index_while_using_old_index_cpp(storage):
    ctx = mp.get_context('spawn')
    p0 = ctx.Process(target=_generate_ds, args=(storage,))
    p0.start()
    p0.join()

    p1 = ctx.Process(target=_task_create_index, args=(storage,))
    p2 = ctx.Process(target=_task_query, args=(storage,))
    p1.start()  # Start subprocess 1
    p2.start()  # Start subprocess 2
    # When subprocess 2 start, the index construction in subprocess 1 is not finished yet, so the query is based on
    # the old index
    p1.join()
    p2.join()

    ds_2 = muller.load(path=official_path(storage, TEST_INDEX_PATH))
    ds_2_filter = ds_2.filter_vectorized([("value", "CONTAINS", "床前明月光")])
    assert ds_2_filter.filtered_index == [i * 5 + 1 for i in range(101)]



if __name__ == '__main__':
    pytest.main(["-s", "test_inverted_index_local.py"])

# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import numpy as np

import muller
from tests.constants import TEST_AGGREGATE_VECTORIZED_PATH
from tests.utils import official_path, official_creds


def get_raw_data():
    """Return test data."""
    return [
        ['1．使用if-else语句，实现比较功能', 'if in1 > in2:', '{"通顺性评分": 5.0, "完整性评分": 3.0}]}', '综合质量评分:6分', 6.0, 5.0],
        ['编写一个Python程序，使用正则表达式匹配手机号码。', '首先，我们需要导入re模块。', '{"通顺性评分": 5.0, "完整性评分": 2.0}]}', '综合质量评分:0分', 0.0, 4.0],
        ['用C语言编译：计算圆柱的表面积与体积', '定义了四个变量:`r`,`h`,`表面积`,`体积`', '{"通顺性评分": 5.0, "完整性评分": 4.0}]}', '综合质量评分:6分', 6.0, 6.0],
        ['怎么将rgb三个通道的值合成一个rgb值', 'Java中，将提取出的R、G、B通道的值相加来合成一个RGB值。', '{"通顺性评分": 5.0, "完整性评分": 4.0}]}', '综合质量评分:7分',
            7.0, 4.0],
        ['怎么将rgb三个通道的值合成一个rgb值', 'Java中，将提取出的R、G、B通道的值相加来合成一个RGB值。', '{"通顺性评分": 5.0, "完整性评分": 4.0}]}', '综合质量评分:7分',
            7.0, 4.0],
        ['写一段基于深度学习的智能客服系统设计', '基于深度学习的智能客服系统，包括数据预处理、模型构建、模型训练和预测等步骤。', '{"通顺性评分": 5.0, "完整性评分": 5.0}]}', '综合质量评分:6分',
            6.0, 3.0]]


def create_muller_dataset(muller_path, creds, overwrite=True, n=1):
    """Create muller dataset for testing."""
    ds = muller.dataset(path=muller_path, creds=creds, overwrite=overwrite)
    tensors = ["ori_query", "ori_response", "query_analysis", "result", "score", "type"]
    ds.create_tensor("ori_query", htype="text", exist_ok=True)
    ds.create_tensor("ori_response", htype="text", exist_ok=True)
    ds.create_tensor("query_analysis", htype="text", exist_ok=True)
    ds.create_tensor("result", htype="text", exist_ok=True)
    ds.create_tensor("score", htype="generic", exist_ok=True, dtype="float64")
    ds.create_tensor("type", htype="generic", exist_ok=True, dtype="float64")
    np_data = np.array(get_raw_data() * n)
    for i, item in enumerate(tensors):
        ds[item].extend(np_data[:, i].astype(ds[item].dtype))
    return ds


def test_count_star(storage):
    """
    select ori_query, ori_response, count(*)
    from dataset
    group by ori_query, ori_response
    order by ori_query;
    """
    ds = create_muller_dataset(official_path(storage, TEST_AGGREGATE_VECTORIZED_PATH), official_creds(storage), n=1)
    result1 = ds.aggregate(
        group_by_tensors=['ori_query', 'ori_response'],
        selected_tensors=['ori_query', 'ori_response'],
        aggregate_tensors=["*"],
        order_by_tensors=['ori_query'],
        )
    result2 = ds.aggregate_vectorized(
        group_by_tensors=['ori_query', 'ori_response'],
        selected_tensors=['ori_query', 'ori_response'],
        order_by_tensors=['ori_query'],
        aggregate_tensors=["*"],
        )
    assert result1.shape == result2.shape


def test_count_column(storage):
    """
    select ori_query, ori_response, count(score)
    from dataset
    where score >= 4
    group by ori_query, ori_response
    order by score;
    """
    ds = create_muller_dataset(official_path(storage, TEST_AGGREGATE_VECTORIZED_PATH), official_creds(storage), n=1)
    result1 = ds.aggregate(
        group_by_tensors=['ori_query', 'ori_response'],
        selected_tensors=['ori_query', 'ori_response'],
        aggregate_tensors=['score'],
        order_by_tensors=['score'],
        )
    result2 = ds.aggregate_vectorized(
        group_by_tensors=['ori_query', 'ori_response'],
        selected_tensors=['ori_query', 'ori_response'],
        order_by_tensors=['score'],
        aggregate_tensors=['score'],
        )
    assert result1.shape == result2.shape


def test_sum_column(storage):
    """
    select ori_query, ori_response, sum(score)
    from dataset
    where score >= 4
    group by ori_query, ori_response
    order by ori_query;
    """
    ds = create_muller_dataset(official_path(storage, TEST_AGGREGATE_VECTORIZED_PATH), official_creds(storage), n=1)
    result1 = ds.aggregate(
        group_by_tensors=['ori_query', 'ori_response'],
        selected_tensors=['ori_query', 'ori_response'],
        aggregate_tensors=['score'],
        order_by_tensors=['ori_query'],
        method='sum'
        )
    result2 = ds.aggregate_vectorized(
        group_by_tensors=['ori_query', 'ori_response'],
        selected_tensors=['ori_query', 'ori_response'],
        order_by_tensors=['ori_query'],
        aggregate_tensors=['score'],
        method='sum'
        )
    assert result1.shape == result2.shape


def test_function(storage):
    """Test different aggregate functions."""
    ds = muller.dataset(official_path(storage, TEST_AGGREGATE_VECTORIZED_PATH), official_creds(storage), overwrite=True)
    ds.create_tensor("name", htype="text", exist_ok=True)
    ds.create_tensor("type", htype="text", exist_ok=True)
    ds.create_tensor("price", htype="generic", exist_ok=True, dtype="float64")
    with ds:
        ds.name.extend(["apple", "banana", "burger", "apple", "coke"])
        ds.type.extend(["fruits", "fruits", "food", "fruits", "drinks"])
        ds.price.extend([2.0, 1.0, 5.0, 2.0, 3])

    result1 = ds.aggregate_vectorized(
                    group_by_tensors=['type', 'name'],
                    selected_tensors=['type', 'name'],
                    order_by_tensors=['price'],
                    aggregate_tensors=['price'],
                    method='sum'
                )
    assert result1.shape == (4, 3)

    result2 = ds.aggregate_vectorized(
                    group_by_tensors=['type'],
                    selected_tensors=['type'],
                    order_by_tensors=['price'],
                    aggregate_tensors=['price'],
                    method='sum',
                    order_direction='ASC'
                )
    assert result2.shape == (3, 2)
    assert result2[0][1] == "3.0"

    result3 = ds.aggregate_vectorized(
                    group_by_tensors=['type'],
                    selected_tensors=['type'],
                    order_by_tensors=['price'],
                    aggregate_tensors=['price'],
                    method='avg',
                )
    assert result3.shape == (3, 2)
    assert result3[0][1] == "5.0"

    result4 = ds.aggregate_vectorized(
                    group_by_tensors=['type'],
                    selected_tensors=['type'],
                    order_by_tensors=['price'],
                    aggregate_tensors=['price'],
                    method='min',
                    order_direction='ASC'
                )
    assert result4.shape == (3, 2)
    assert result4[0][1] == "1.0"

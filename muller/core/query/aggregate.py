# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

from collections import Counter
from functools import reduce
from itertools import product
from time import time
from typing import Callable, List, Optional, Sequence, Dict

import numpy as np

import muller
from muller.constants import (
    TRANSFORM_PROGRESSBAR_UPDATE_INTERVAL,
)
from muller.util.compute import get_compute_provider
from muller.util.exceptions import AggregateError


def _get_query_index_pair(query_labels: list, num_workers):
    """
        return: [[0,1,2], [3,4,5], [6,7]]
    """
    split_indices = np.array_split(range(len(query_labels)), min(num_workers, len(query_labels)))
    return [item.tolist() for item in split_indices]


def _is_suitable_accelerate_count(dataset: muller.Dataset, group_by_tensors: List[str], aggregate_tensors: List[str],
                                  filter_function: Callable, method: str = "count"):
    """
        scenarios for example:
            1、select level_1, count(*) from xx group by level_1;
            2、select level_1, level_2, count(*) from xx group by level_1, level_2;
            3、select level_1, level_2, level_3, count(*) from xx group by level_1, level_2, level_3;
        specify:
            1、suitable only for count
            2、has no where clause
            3、htype for all group_by tensor must be 'class_label'
            4、inverted index must be created for all group_by tensors before 'ds.aggregate'
    """
    indexed_tensors = dataset.indexed_tensors
    return (method == 'count'
            and filter_function is None
            and all(x == "*" for x in aggregate_tensors)
            and all(dataset[tensor].htype == 'class_label' for tensor in group_by_tensors)
            and all(tensor in indexed_tensors for tensor in group_by_tensors))


def _get_query_labels(dataset: muller.Dataset, group_by_tensors: List[str], is_acc_count: bool):
    """
        for example:
            # tensor level_1 with htype 'class_label'
            ds.level_1.info['class_names'] == ['多任务','创意生成', '多轮AIGC', 'WizardLM']
            return:
            [('level_1', 0), ('level_1', 1), ('level_1', 2), ('level_1', 3)]
    """
    result = []
    if is_acc_count:
        for tensor in group_by_tensors:
            class_names = dataset[tensor].info['class_names']
            for label_index, _ in enumerate(class_names):
                result.append((tensor, label_index))
    return result


def _get_tensor_index_indices(result_dict: dict, dataset: muller.Dataset, index_query):
    """
    This is an example of result_dict:
        {
            ("level_1", 0): [5, 8, 12, 25],
            ("level_1", 1): [6, 9, 15, 22],
            ("level_1", 2): [0, 10, 11, 21],
            ("level_1", 3): [3, 4, 7, 10],
            ("level_2", 0): [5, 8, 25, 26],
            ("level_2", 1): [1, 9, 15, 29],
            ("level_2", 2): [7, 13, 14, 16]
        }
    """
    tensor_name, class_value = index_query
    ids_set = dataset.query(tensor_name, class_value)
    result_dict[index_query] = list(ids_set)


def _build_agg_counter(counter_dict: dict,
                       sample_in: muller.Dataset,
                       label_list: list,
                       selected_tensors: List[str],
                       group_by_tensors: List[str],
                       aggregate_tensors: list,
                       method: str = 'count'):
    selected_indices_dict = {group_by_tensor: i
                             for i, group_by_tensor in enumerate(group_by_tensors)
                             if group_by_tensor in selected_tensors}

    def get_sample_data(tensor_name):
        sample_value = sample_in[tensor_name].data()['value']
        sample_data = sample_value.flatten()[0] if isinstance(sample_value, np.ndarray) else sample_value
        return label_list[selected_indices_dict[tensor_name]][sample_data] \
            if sample_in[tensor_name].htype == 'class_label' \
            else sample_data

    group_key = tuple(get_sample_data(tensor_name) for tensor_name in group_by_tensors)
    for tensor_name in aggregate_tensors:
        counter = counter_dict[tensor_name]
        counter.setdefault(group_key, 0)
        if method == 'count':
            if tensor_name == '*':
                # Note: count(*)
                counter.update((group_key,))
            else:
                # Note: count(#{column})
                if sample_in[tensor_name].data()['value']:
                    counter.update((group_key,))
        if method == 'sum':
            # Note: sum(#{column})
            counter[group_key] = counter[group_key] + sample_in[tensor_name].data()['value'].flatten()[0]


def aggregate_by_inplace(dataset: muller.Dataset,
                         filter_function: Callable[[muller.Dataset], bool],
                         progressbar: bool,
                         selected_tensors: List[str],
                         group_by_tensors: List[str],
                         aggregate_tensors: List[str],
                         order_by_tensors: Optional[list] = None,
                         order_direction: str = "DESC",
                         method: str = 'count',
                         ):
    """Aggregate function of inplace."""
    it = enumerate(dataset)
    if progressbar:
        from tqdm import tqdm
        it = tqdm(it, total=len(dataset))
    is_acc_count = _is_suitable_accelerate_count(dataset, group_by_tensors, aggregate_tensors, filter_function,
                                                 method)
    query_labels = _get_query_labels(dataset, group_by_tensors, is_acc_count)
    label_list = [dataset[tensor].info['class_names']
                  for tensor in group_by_tensors
                  if dataset[tensor].htype == 'class_label']

    def filter_dataset_slice():
        counter_dict = {tensor: Counter() for tensor in aggregate_tensors}
        is_sample_match = True
        for _, sample_in in it:
            if filter_function is not None:
                is_sample_match = filter_function(sample_in)
            if is_sample_match:
                _build_agg_counter(counter_dict, sample_in, label_list, selected_tensors, group_by_tensors,
                                   aggregate_tensors, method)
            else:
                continue
        return counter_dict

    def query_tensor_index():
        result_dict = {}
        for index_query in query_labels:
            _get_tensor_index_indices(result_dict, dataset, index_query)
        return result_dict

    collected_dicts: Dict[str:Counter] | Dict[tuple:list]
    try:
        fun = query_tensor_index if is_acc_count else filter_dataset_slice
        collected_dicts = fun()
        if is_acc_count:
            return _reduce_collected_query_indices_dict([collected_dicts], label_list, selected_tensors,
                                                        group_by_tensors, aggregate_tensors, order_by_tensors,
                                                        order_direction)

        return _reduce_collected_counter_dict([collected_dicts], selected_tensors, group_by_tensors,
                                                  aggregate_tensors, order_by_tensors, order_direction)
    except Exception as e:
        raise AggregateError from e


def _turn_agg_counter_ndarray(counter_dict: dict, group_by_tensors: List[str], selected_tensors: List[str]):
    """
        This is an example of counter_dict: (示例)
        {
            "*": Counter(("key11", "key12",): 2,
                         ("key21", "key22"): 1),
            "#{agg_tensor_name}": Counter(("key11", "key12",): 1,
                                          ("key21", "key22"): 1)
        }
        return example:
        [["key11", "key12", 2, 1],
                 ["key21", "key22", 1, 1]]
    """
    agg_append_list = []
    most_common_keys = []
    for counter_item in list(counter_dict.values()):
        if not agg_append_list:
            # reverse order
            agg_append_list = counter_item.most_common()
            most_common_keys = [row[0] for row in agg_append_list]
            continue
        for i, row in enumerate(agg_append_list):
            agg_append_list[i] = agg_append_list[i] + (counter_item[row[0]],)
    # return none when dataset is empty
    if len(most_common_keys) == 0:
        return None
    # indices for selected_tensor in group_by_tensors
    selected_indices = [i for i, group_by_item in enumerate(group_by_tensors) if group_by_item in selected_tensors]
    # prepare selected tensors
    agg_np_data = np.array(most_common_keys)[:, selected_indices]
    # prepare aggregate tensors
    np_append_data = np.array([row[1:] for row in agg_append_list])
    # combine above tensors
    agg_np_data = np.column_stack((agg_np_data, np_append_data))
    return agg_np_data


def _get_agg_orderby_result(agg_result_data: np.ndarray,
                            selected_tensors: List[str],
                            aggregate_tensors: List[str],
                            order_by_tensors: Optional[list] = None,
                            order_direction: str = "DESC",
                            ):
    if order_by_tensors and agg_result_data is not None:
        # indices for order_by_tensor in the selected_tensors
        order_by_indices = [i
                            for i, order_item in enumerate(selected_tensors + aggregate_tensors)
                            if order_item in order_by_tensors]

        # lexicographical order
        indices = np.lexsort(tuple(
            [np.char.strip(agg_result_data[:, index], "'") for index in order_by_indices[::-1]]))
        if order_direction.lower() == 'ASC'.lower():
            return agg_result_data[indices]
        return agg_result_data[indices[::-1]]
    return agg_result_data


def _reduce_collected_counter_dict(collected_dicts,
                                   selected_tensors: List[str],
                                   group_by_tensors: List[str],
                                   aggregate_tensors: List[str],
                                   order_by_tensors: Optional[list] = None,
                                   order_direction: str = "DESC",
                                   ):
    reduced_dicts = {tensor: Counter() for tensor in aggregate_tensors}
    for raw_tensor_agg_dict in collected_dicts:
        for tensor, counter_item in raw_tensor_agg_dict.items():
            tensor_counter = reduced_dicts[tensor]
            tensor_counter.update(counter_item)
    agg_result_data = _turn_agg_counter_ndarray(reduced_dicts, group_by_tensors, selected_tensors)
    return _get_agg_orderby_result(agg_result_data, selected_tensors, aggregate_tensors,
                                   order_by_tensors, order_direction)


def _turn_agg_query_ndarray(queried_dict: dict, label_list: list, group_by_tensors: List[str],
                            selected_tensors: List[str]):
    """
        This is an example of queried_dict: (示例)
        {
            ("level_1", 0): [5, 8, 12, 25],
            ("level_1", 1): [6, 9, 15, 22],
            ("level_1", 2): [0, 10, 11, 21],
            ("level_1", 3): [3, 4, 7, 10],
            ("level_2", 0): [5, 8, 25, 26],
            ("level_2", 1): [1, 9, 15, 29],
            ("level_2", 2): [7, 13, 14, 16]
        }
        label_list: [['多任务', '开放问答', '多轮AIGC', '多任务'],
                    ['实时翻译工具', '智能助手', '聊天机器人']]
        return: [['多任务', '实时翻译工具', '3'],
                ['开放问答', '智能助手', '2'],
                ['多轮AIGC', '聊天机器人', '1'],
                ['多任务', '聊天机器人', '1']]
    """
    label_len_list = [range(len(class_label)) for class_label in label_list]

    # list full permutation with product
    product_group = list(product(*label_len_list))
    selected_indices = [i
                        for i, group_by_item in enumerate(group_by_tensors)
                        if group_by_item in selected_tensors]

    def intersect_array(left_arr, right_arr):
        # intersect with all group_by tensors
        return np.intersect1d(left_arr, right_arr)

    agg_group_counter = Counter()
    for group_key in product_group:
        # prepare the intersect list from left to right based on group_key
        index_group_lists = []
        for index, label_value in enumerate(group_key):
            key = (group_by_tensors[index], label_value)
            value = queried_dict.get(key)  # 使用 get() 方法避免 KeyError
            if value is not None:
                index_group_lists.append(value)
        total = reduce(intersect_array, index_group_lists)
        count = len(total)
        if count > 0:
            # mapping class_label int to class_names
            label_group_key = tuple(label_list[index][label_value] for index, label_value in enumerate(group_key)
                                    if index in selected_indices)
            agg_group_counter[label_group_key] = count
    sorted_arr = agg_group_counter.most_common()

    # merge selected_tensors and count part
    return np.column_stack((
        np.array([row[0] for row in sorted_arr])[:, selected_indices],  # for selected_tensors part
        np.array([row[1:] for row in sorted_arr]) # for count part
    ))


def _reduce_collected_query_indices_dict(collected_dicts,
                                         label_list: list,
                                         selected_tensors: List[str],
                                         group_by_tensors: List[str],
                                         aggregate_tensors: List[str],
                                         order_by_tensors: Optional[list] = None,
                                         order_direction: str = "DESC",
                                         ):
    queried_dict = {key: list_item
                    for dict_item in collected_dicts
                    for key, list_item in dict_item.items()}
    agg_result_data = _turn_agg_query_ndarray(queried_dict, label_list, group_by_tensors, selected_tensors)
    return _get_agg_orderby_result(agg_result_data, selected_tensors, aggregate_tensors,
                                   order_by_tensors, order_direction)


def aggregate_by_with_compute(dataset: muller.Dataset,
                              filter_function,
                              num_workers,
                              scheduler,
                              progressbar,
                              selected_tensors: List[str],
                              group_by_tensors: List[str],
                              aggregate_tensors: List[str],
                              order_by_tensors: Optional[list] = None,
                              order_direction: str = "DESC",
                              method: str = "count",
                              ):
    """Aggregate function of compute."""
    dataset.is_iteration = True
    # Removed SampleStreaming which creates IO Blocks to get dataset index,
    # instead, used dataset.index below to get index.
    compute = get_compute_provider(scheduler=scheduler, num_workers=num_workers)
    is_acc_count = _is_suitable_accelerate_count(dataset, group_by_tensors, aggregate_tensors, filter_function, method)
    query_labels = _get_query_labels(dataset, group_by_tensors, is_acc_count)
    label_list = [dataset[tensor].info['class_names']
                  for tensor in group_by_tensors
                  if dataset[tensor].htype == 'class_label']

    def filter_slice(indices: Sequence[int]):
        counter_dict = {tensor: Counter() for tensor in aggregate_tensors}
        is_sample_match = True
        for i in indices:
            sample_in = dataset[i]
            if filter_function is not None:
                is_sample_match = filter_function(sample_in)
            if is_sample_match:
                _build_agg_counter(counter_dict, sample_in, label_list, selected_tensors, group_by_tensors,
                                   aggregate_tensors, method)
            else:
                continue
        return counter_dict

    def query_tensor_index(indices: Sequence[int]):
        result_dict = {}
        for i in indices:
            index_query = query_labels[i]
            _get_tensor_index_indices(result_dict, dataset, index_query)
        return result_dict

    def filter_slice_with_progress(pg_callback, indices: Sequence[int]):
        counter_dict = {tensor: Counter() for tensor in aggregate_tensors}
        progress = 0
        start = time()
        is_sample_match = True
        for i in indices:
            sample_in = dataset[i]
            if filter_function is not None:
                is_sample_match = filter_function(sample_in)
            if is_sample_match:
                _build_agg_counter(counter_dict, sample_in, label_list, selected_tensors, group_by_tensors,
                                   aggregate_tensors, method)
            else:
                continue
            progress += 1

            if time() - start > TRANSFORM_PROGRESSBAR_UPDATE_INTERVAL:
                pg_callback(progress)
                progress = 0
                start = time()
        if progress > 0:
            pg_callback(progress)
        return counter_dict

    def query_tensor_index_with_progress(pg_callback, indices: Sequence[int]):
        result_dict = {}
        progress = 0
        start = time()
        for i in indices:
            index_query = query_labels[i]
            _get_tensor_index_indices(result_dict, dataset, index_query)
            progress += 1

            if time() - start > TRANSFORM_PROGRESSBAR_UPDATE_INTERVAL:
                pg_callback(progress)
                progress = 0
                start = time()
        if progress > 0:
            pg_callback(progress)
        return result_dict

    try:
        sub_worker_indices = _get_query_index_pair(query_labels, num_workers) \
            if is_acc_count \
            else get_worker_dataset_indices(dataset, num_workers)

        collected_dicts: Sequence[Dict[str:Counter] | Dict[tuple:list]]
        if progressbar:
            fun = query_tensor_index_with_progress if is_acc_count else filter_slice_with_progress
            total_length = len(query_labels) if is_acc_count else len(dataset)
            collected_dicts = compute.map_with_progress_bar(fun, sub_worker_indices,
                                                            total_length=total_length)
        else:
            fun = query_tensor_index if is_acc_count else filter_slice
            collected_dicts = compute.map(fun, sub_worker_indices)

        if is_acc_count:
            return _reduce_collected_query_indices_dict(collected_dicts, label_list, selected_tensors,
                                                        group_by_tensors, aggregate_tensors, order_by_tensors,
                                                        order_direction)

        return _reduce_collected_counter_dict(collected_dicts, selected_tensors, group_by_tensors,
                                                  aggregate_tensors, order_by_tensors, order_direction)
    except Exception as e:
        raise AggregateError from e
    finally:
        compute.close()


def get_worker_dataset_indices(dataset: muller.Dataset, num_workers: int):
    """Get the workers of dataset indices."""
    version_state = dataset.version_state
    tensor_lengths = [
        len(version_state["full_tensors"][version_state["tensor_names"][tensor]])
        for tensor in dataset.tensors.keys()
    ]

    length = min(tensor_lengths, default=0)

    # get indices from dataset, split by num_workers
    indices = list(dataset.index.values[0].indices(length))
    split_indices = np.array_split(indices, min(num_workers, len(indices)))
    return [split_indices[i].tolist() for i in range(len(split_indices))]


def _check_incoming_param(dataset: muller.Dataset,
                          selected_tensors: List[str],
                          group_by_tensors: List[str],
                          order_by_tensors: Optional[list] = None,
                          aggregate_tensors: Optional[list] = None,
                          method: str = 'count',
                          ):
    if any(item not in group_by_tensors for item in selected_tensors):
        raise KeyError("Syntax error, selected_tensor not in group_by_tensors")

    if (order_by_tensors
            and not any(item in selected_tensors or item in aggregate_tensors for item in order_by_tensors)):
        raise KeyError("Syntax error, order_by_tensor not in selected_tensors or aggregate_tensors")

    if (method == 'sum'
            and not aggregate_tensors):
        raise KeyError("aggregate_tensors is empty, please check again.")

    def is_valid_aggregate_tensors(tmp_aggregate_tensors, tmp_dataset):
        """检查 aggregate_tensors 中的所有元素是否为数值类型，且不包含 '*'。"""
        return not tmp_aggregate_tensors or "*" in tmp_aggregate_tensors or \
            not all(np.issubdtype(tmp_dataset[item].dtype, np.number) for item in tmp_aggregate_tensors)

    if method == 'sum' and is_valid_aggregate_tensors(aggregate_tensors, dataset):
        raise ValueError("for aggregate_tensors, only the numeric type is supported.")


def aggregate_dataset(
        dataset: muller.Dataset,
        filter_function: Callable[[muller.Dataset], bool],
        group_by_tensors: List[str],
        selected_tensors: List[str],
        order_by_tensors: Optional[list] = None,
        aggregate_tensors: Optional[list] = None,
        order_direction: Optional[str] = 'DESC',
        num_workers: int = 0,
        scheduler: str = "thread",
        progressbar: bool = True,
        method: str = "count",
):
    """Aggregate on the target dataset."""
    _check_incoming_param(dataset, selected_tensors, group_by_tensors, order_by_tensors, aggregate_tensors, method)
    if method == 'count' and not aggregate_tensors:
        aggregate_tensors = ['*']
    try:
        if num_workers > 0:
            agg_result = aggregate_by_with_compute(
                dataset,
                filter_function,
                num_workers,
                scheduler,
                progressbar,
                selected_tensors,
                group_by_tensors,
                aggregate_tensors,
                order_by_tensors,
                order_direction,
                method,
            )
        else:
            agg_result = aggregate_by_inplace(
                dataset,
                filter_function,
                progressbar,
                selected_tensors,
                group_by_tensors,
                aggregate_tensors,
                order_by_tensors,
                order_direction,
                method,
            )
    except Exception as e:
        raise e

    return agg_result

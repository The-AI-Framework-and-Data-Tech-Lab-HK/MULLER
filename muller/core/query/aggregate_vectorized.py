# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

from typing import List, Optional

from concurrent.futures import ProcessPoolExecutor
import numpy as np

import muller


def to_numpy(tensor_obj):
    """Convert tensor to numpy data."""
    return tensor_obj.numpy_full()


def get_numpy_data(ds, tensors: List[str]):
    """Convert tensors to numpy data multiprocessed."""
    futures_list = []
    with ProcessPoolExecutor(len(tensors)) as executor:
        for tensor in tensors:
            futures_list.append(executor.submit(to_numpy, ds[tensor]))
    results = [future.result() for future in futures_list]
    return np.column_stack((results))


def aggregate_vectorized_dataset(
        dataset: muller.Dataset,
        group_by_tensors: List[str],
        selected_tensors: List[str],
        order_by_tensors: Optional[list] = None,
        aggregate_tensors: Optional[list] = None,
        order_direction: Optional[str] = 'DESC',
        method: str = 'count',
):
    """
    A vectorized aggregate function accelerated by the parallel computing supported by numpy.
    :param group_by_tensors(List[str]) - names of the tensors, as the groupby clause in aggregate statement.
    :param selected_tensors(List[str]) - names of the tensors, as the select clause in aggregate statement.
    :param order_by_tensors(Optional, list[str]) - names of the tensors, as the orderby clause in aggregate statement.
    :param aggregate_tensors(Optional, list[str]) - names of the tensors, as the aggregate function clause in aggregate
            statement. support '*' only for count.
    :param order_direction(str) - 'DESC' or 'ASC', 'DESC' for default.
    :param method(str) - 'count' 'avg' 'min' 'max' 'sum'. 'count' for default.
    :return: A numpy array of aggregated result.
    """
    # 1. groupby
    _, indices, inverse_indices, counts = np.unique(get_numpy_data(dataset, group_by_tensors),
                                                    axis=0, return_index=True, return_inverse=True, return_counts=True)

    # 2. aggregate sum or count
    if aggregate_tensors:
        aggregate_col = get_agg_col(dataset, aggregate_tensors, order_direction, method, inverse_indices, counts)
    else:
        aggregate_col = None

    # 3. Concat the selected tensors and aggregate tensors
    selected = np.concatenate([dataset[tensor][list(indices)].numpy() for tensor in selected_tensors], axis=1)
    selected = np.column_stack((selected, aggregate_col)) if aggregate_col is not None else selected

    if order_by_tensors:
        # 4. Order by
        index_order = np.argsort(indices)  # Sort the indices based on `order by list`
        selected = selected[index_order]
        indices = indices[index_order]
        aggregate_col = aggregate_col[index_order]

        for tensor in order_by_tensors[::-1]:  # Reverse
            if tensor in aggregate_tensors:  # Sort by the aggregated results
                index_order = np.argsort(aggregate_col.transpose()[aggregate_tensors.index(tensor)])
            else:  # Sort based on the original context in tensors
                index_order = np.argsort(dataset[tensor][list(indices)].numpy().flatten())
            if order_direction == 'DESC':
                index_order = index_order[::-1]
            selected = selected[index_order]
    return selected


def get_one_hot_matrix(inverse_indices):
    """Return an one hot matrix, while each row is a class and each column is the original data.
    1 denotes that the data belongs to this class. 0 denotes the reverse case. """
    num_categories = np.max(inverse_indices) + 1  # Number of categories
    one_hot_matrix = np.zeros((num_categories, len(inverse_indices)))  # Initialize all the elements as 0
    one_hot_matrix[inverse_indices, np.arange(len(inverse_indices))] = 1  # Set the target element as 1
    return one_hot_matrix


def get_agg_col(
        dataset,
        aggregate_tensors,
        order_direction,
        method,
        inverse_indices,
        counts
):
    """Returns aggregated result."""
    if method == 'count':
        aggregate_col = np.repeat(counts[:, None], len(aggregate_tensors), axis=1)
    elif method == 'sum':
        one_hot_matrix = get_one_hot_matrix(inverse_indices)
        aggregate_col = np.concatenate(
            [np.matmul(one_hot_matrix, dataset[tensor].numpy())  # sum = one hot matrix * tensor
                for tensor in aggregate_tensors
            ], axis=1
        )
    elif method == 'avg':
        one_hot_matrix = get_one_hot_matrix(inverse_indices)
        count_arr = counts.reshape(len(counts), 1)  # Reshape as (n, 1)
        aggregate_col = np.concatenate(
            [np.divide(np.matmul(one_hot_matrix, dataset[tensor].numpy()), count_arr)  # avg = sum / count
                    for tensor in aggregate_tensors
            ], axis=1
        )
    elif method == 'min':
        one_hot_matrix = get_one_hot_matrix(inverse_indices)
        aggregate_col = None
        for tensor in aggregate_tensors:
            grouped_arr = np.multiply(one_hot_matrix, dataset[tensor].numpy().flatten())
            ma_data = np.ma.masked_equal(grouped_arr, 0)  # Discard the 0 in the array
            min_col = np.min(ma_data, axis=1).data.reshape(len(counts), 1)
            if not aggregate_col:
                aggregate_col = min_col
            else:
                aggregate_col = np.column_stack((aggregate_col, min_col))  # Concat
    elif method == 'max':
        one_hot_matrix = get_one_hot_matrix(inverse_indices)
        aggregate_col = np.concatenate(
            [np.max(
                np.multiply(one_hot_matrix, dataset[tensor].numpy().flatten()), axis=1).reshape(len(counts), 1)
                for tensor in aggregate_tensors
            ], axis=1
        )
    else:
        aggregate_col = None
    return aggregate_col

# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import json
import logging
import os
import re
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import numpy as np

import muller
from muller.constants import FILTER_CACHE_SIZE, REGEX_BATCH_SIZE
from muller.util.exceptions import (FilterVectorizedConditionError,
                                   FilterVectorizedConnectorListError,
                                   FilterOperatorNegationUnsupportedError,
                                   InvertedIndexUnsupportedError, InvertedIndexNotExistsError)


def filter_based_on_numpy(
        tensor,
        target: np.array,
        length: int,
        operator: str,
        negation: Optional[str] = None

):
    """
        A vectorized filtering function accelerated by the parallel computing supported by numpy.
        :param source: the source np array, from the tensor column.
        :param target: the target np array that represents the target values.
        :param length: the length of the tensor.
        :param operator: the operator which could be ">", ">=", "<", "<=", "==", "!=".
        :param negation: to indicator of logical complment for this condition, default to be None, could be "NOT".
        :return: a list of index which satisfies the filter condition.
    """
    source = tensor.numpy_continuous()
    compare_result = np.full((length, 1), False)

    if operator == ">=":
        compare_result = source >= target  # Note: this is a boolean numpy array
    elif operator == ">":
        compare_result = source > target
    elif operator == "==":
        compare_result = source == target
    elif operator == "<=":
        compare_result = source <= target
    elif operator == "<":
        compare_result = source < target
    elif operator == "!=":
        compare_result = source != target
    elif operator == "LIKE":
        source = source.flatten()
        indices = filter_like(source, target)
        if negation:
            return np.setdiff1d(np.arange(0, length), indices)
        return indices

    if negation:
        # negation=True, return the results that does not satisfy the condition
        result_index = np.where(compare_result == 0)
    else:
        # negation=False, return the results that satisfy the condition
        result_index = np.where(compare_result == 1)

    return result_index[0]


def filter_vectorized_dataset(
        dataset: muller.Dataset,
        condition_list: list,
        connector_list: Optional[List] = None,
        offset: Optional[int] = 0,
        limit: Optional[int] = None,
        compute_future: Optional[bool] = True,
        use_local_index: bool = True,
        max_workers: Optional[int] = 16,
        show_progress: bool = False,
):
    """
    A vectorized filtering function accelerated by the parallel computing supported by numpy.
    :param dataset: a MULLER dataset.
    :param condition_list: the list of filtering functions, while each element in the list should be
            (tensor_column, filter_condition, filter_value, use_inverted_index, negation),
            e.g., ("test", ">", 2, False, "NOT"), ("test", "==", "2", True, None),
            ("test", "BETWEEN", [3, 5], True, None), ("test", "LIKE", "A[0-2]", False, None).
            Note:
            (1) legal operators: >, <, >=, <=, ==, !=, CONTAINS, BETWEEN, LIKE
            (2) use_inverted_index defaults to be False if not specified.
                However, BETWEEN and CONTAINS must use inverted_index for searching.
            (3) negation defaults to be None if not specified.
                However, we do not support using inverted_index and negation for the same time.
            (4) for the CONTAINS operator, the value must be string.
            (5) for the LIKE operator, the value must be string or regex pattern.
            (6) for the ">", "<", ">=", "<=" operator, the value must be int or float.
            (7) for the "==" and "!=" operator, the value must be bool, int, str or float.
            (8) for the BETWEEN operator, the value must be a list or tuple which contains two int or two float.
                We only deal with >= or <= in the BETWEEN operator.
    :param connector_list: a list of connectors to connect each element in the filter list, while each of the element
            can be "OR" or "AND".
            Note: len(connector_list) = len(condition_list) - 1
    :param offset: The start index of dataset to filter. Default to 0.
    :param limit: The number of filtered results. Default to none, which means unlimited.
    :param compute_future: If True, calculate the content of the next page in advance.
            Default to True.
    :return: a MULLER dataset which is a subset of the original dataset which satisfies the filter conditions.
    """
    length = len(dataset)  # Length: # of rows in the dataset. We assume all columns have the same length.
    if not limit:
        limit = length

    # Retrieve the results
    filtered_index = _obtain_final_index(dataset, condition_list, connector_list, offset, limit, use_local_index,
                                         max_workers, show_progress)

    # Insert to upper cache
    _insert_to_cache(dataset, condition_list, connector_list, offset, limit, filtered_index)

    # Compute some results in asyn way in advance
    if limit != len(dataset) and compute_future:
        if len(filtered_index) > 0 and filtered_index[-1] != len(dataset) - 1:  # Not the last row yet
            _filter_vectorized_next(dataset,
                                    condition_list=condition_list,
                                    connector_list=connector_list,
                                    offset=int(filtered_index[-1] + 1),
                                    limit=limit,
                                    use_local_index=use_local_index,
                                    max_workers=max_workers,
                                    )

    new_filtered_index = list(filtered_index)
    dataset = dataset[new_filtered_index]
    dataset.filtered_index = new_filtered_index

    return dataset


def _obtain_final_index(dataset,
                        condition_list,
                        connector_list,
                        offset,
                        limit,
                        use_local_index,
                        max_workers,
                        show_progress: bool = False,):
    """
    Obtain a list of global indexes that match the query conditions from the dataset.
    """
    partial_filtered_index = np.array([])
    settings = {
        'offset': offset,
        'use_local_index': use_local_index,
        'max_workers': max_workers,
        'show_progress': show_progress
    }
    # Step 1：Fetch results from cache
    result, recompute_flag, recompute_range = _fetch_from_cache(dataset, condition_list, connector_list,
                                                                settings['offset'], limit)
    if result is not None:
        # If there is existing query in the query, no need to recompute. Return the results directly.
        if not recompute_flag:
            return result
        partial_filtered_index = result

    # Step 2：Legality verification
    legality_verification(dataset, condition_list, connector_list)

    # Step 3：Use array-based full scan or inverted index?
    filter_list = assign_compute_method(dataset, condition_list) # filter_inverted_index_list, filter_vec_list
    if filter_list[0]: # filter_inverted_index_list
        # Since it is difficult to set range while using inverted index, we do not consider fetch results from cache.
        recompute_range = (0, len(dataset))

    # Step 4: Execution of array-based full scan and inverted index (parallel)
    final_index_array = filter_with_inverted_index(dataset, filter_list[0], connector_list,
                                                   filter_with_numpy_vec(dataset[recompute_range[0]:recompute_range[1]],
                                                                         filter_list[1], settings['offset'],
                                                                         settings['show_progress']), settings)

    if len(partial_filtered_index):
        return np.unique(np.concatenate((partial_filtered_index,
                                                   final_index_array[:limit] + recompute_range[0])))

    return final_index_array[:limit]


def legality_verification(dataset, condition_list, connector_list):
    """
    Check the legality of each condition and connector.
    """
    # Need to check the legality of connector_list and condition_list
    # The length of connector_list
    if connector_list is None:
        if len(condition_list) > 1:
            raise FilterVectorizedConnectorListError(connector_list)
    else:
        if len(connector_list) != len(condition_list) - 1:
            raise FilterVectorizedConnectorListError(connector_list)
        # The illegal keywords in connector_list
        unique_connectors = set(connector_list)
        if unique_connectors not in [{"AND"}, {"OR"}, {"AND", "OR"}]:
            raise FilterVectorizedConnectorListError(connector_list)


def assign_compute_method(dataset, condition_list):
    """
    Put them in the corresponding filter_vec_list (for filter based on numpy) and
       filter_inverted_index_list (for filtered based on inverted index)
    """
    filter_vec_list = {}  # Need to use array-based full scan
    filter_inverted_index_list = {}  # Need to use inverted index

    for i, tmp_tuple in enumerate(condition_list):  # Process each independent filter condition
        if len(tmp_tuple) == 3:  # Not specify whether using inverted index
            tmp_tuple += (False,)
        if len(tmp_tuple) == 4:  # Not specify whether it is negation
            tmp_tuple += (None,)
        if len(tmp_tuple) != 5:  # Illegal number of items with the above specification
            raise FilterVectorizedConditionError(tmp_tuple)

        (tensor, operator, value, use_inverted_index, negation) = tmp_tuple
        if operator == 'CONTAINS':
            if not isinstance(value, str):
                raise FilterVectorizedConditionError(tmp_tuple)
            if negation:
                raise FilterOperatorNegationUnsupportedError(operator)
            filter_inverted_index_list.update({i: (tensor, operator, value)})
        elif operator == 'LIKE':
            if dataset[tensor].htype != "text":
                raise FilterVectorizedConditionError(tmp_tuple)
            if not isinstance(value, (str, re.Pattern)):
                raise FilterVectorizedConditionError(tmp_tuple)
            filter_vec_list.update({i: (tensor, operator, value, negation)})
        elif operator == "BETWEEN":
            if dataset[tensor].htype != "generic":
                raise FilterVectorizedConditionError(tmp_tuple)
            if not isinstance(value, (list, tuple)):
                raise FilterVectorizedConditionError(tmp_tuple)
            if len(value) != 2 or not isinstance(value[0], (int, float)) or not isinstance(value[1], (int, float)):
                raise FilterVectorizedConditionError(tmp_tuple)
            if not use_inverted_index:
                use_inverted_index = True
            if use_inverted_index:
                if negation:
                    raise FilterOperatorNegationUnsupportedError(operator)
                filter_inverted_index_list.update({i: (tensor, operator, value)})
        else:
            if dataset[tensor].htype == "text":
                if not isinstance(value, str):
                    raise FilterVectorizedConditionError(tmp_tuple)
            if dataset[tensor].htype == "generic":
                if not isinstance(value, (bool, float, int)):
                    raise FilterVectorizedConditionError(tmp_tuple)
            if operator in [">", "<", ">=", "<="]:
                if not isinstance(value, (int, float)):
                    raise FilterVectorizedConditionError(tmp_tuple)
                if use_inverted_index:
                    raise InvertedIndexUnsupportedError(operator)
                filter_vec_list.update({i: (tensor, operator, value, negation)})
            elif operator in ["==", "!="]:
                if not isinstance(value, (bool, int, str, float)):
                    raise FilterVectorizedConditionError(tmp_tuple)
                if use_inverted_index:
                    if operator == "==":
                        if negation:
                            raise FilterOperatorNegationUnsupportedError(operator)
                        filter_inverted_index_list.update({i: (tensor, operator, value)})
                    else:
                        raise InvertedIndexUnsupportedError(operator)
                else:
                    filter_vec_list.update({i: (tensor, operator, value, negation)})

    return filter_inverted_index_list, filter_vec_list


def optimize_connector(target_idx, tensor, target_value, dataset):
    """Find indices in the given list where tensor values equal target_value."""

    tensor_values = np.array(dataset[tensor].numpy_batch_random_access(index_list=target_idx, parallel="threaded"))
    target_idx_array = np.array(target_idx)

    if tensor_values.ndim > 1:
        # If this is a 2-d array with only one value in each element, flatten it into 1-d array
        if tensor_values.shape[1] == 1:
            tensor_values = tensor_values.ravel()
        else:
            # If this is a multi-dimension array, then we need to compare element by element.
            # However, this case would not happen in normal situation.
            raise ValueError(f"Tensor values have unexpected shape: {tensor_values.shape}")

    mask = tensor_values == target_value

    matched_indices = target_idx_array[mask]
    return matched_indices


def filter_with_inverted_index(dataset,
                               filter_inverted_index_list,
                               connector_list,
                               pre_result,
                               settings):
    """
    Function to handle the key of inverted index with the result of numpy vectorize to get the final result.
    """

    final_index_array = None
    results_cache = pre_result if pre_result else {}
    skip_next_key = False

    for idx, key in enumerate(sorted(set(filter_inverted_index_list.keys()) | set(pre_result.keys()))):
        if skip_next_key and key == 1:
            skip_next_key = False
            continue

        # Process key from filter_inverted_index_list
        if key in filter_inverted_index_list:
            params = filter_inverted_index_list[key] # tensor, operator, value

            if settings['show_progress']:
                logging.getLogger().setLevel(logging.INFO)
                logging.info(f"Computing the result of {(params[0], params[1], params[2])}...")

            search_type = (
                "complex_fuzzy_match" if (params[1] == "CONTAINS" and "||" in params[2])
                else "fuzzy_match" if params[1] == "CONTAINS"
                else "range_match" if params[1] == "BETWEEN"
                else "exact_match"
            )

            is_and_connector = key > 0 and key - 1 < len(connector_list) and connector_list[key - 1] == "AND"

            if search_type == "exact_match" and is_and_connector and final_index_array is not None:
                # Use the old final_index_array as prev_result
                final_index_array = _optimize_and_condition(
                    final_index_array, params[0], params[2], dataset, settings['offset']
                )
                continue

            is_first_key_and = (
                    key == 0 and
                    search_type == "exact_match" and
                    (len(filter_inverted_index_list) + len(pre_result)) > 1 and
                    connector_list[key] == "AND"
            )

            if is_first_key_and:
                final_index_array, skip_next_key = _optimize_first_key(
                    dataset, filter_inverted_index_list, results_cache, key,
                    params[0], params[2], settings['offset'], settings['use_local_index'], settings['max_workers']
                )
                if final_index_array is not None:
                    continue

            # Normal case (no optimization)
            if key not in results_cache:
                key_individual_result = np.sort(np.array(list(
                    matching(dataset, params[0], params[2], search_type, settings['use_local_index'],
                             settings['max_workers'])
                )))
                if settings['offset']:
                    key_individual_result = key_individual_result[key_individual_result >= settings['offset']]
                results_cache[key] = key_individual_result

        # Fetch the result of the current key (may com from filter_inverted_index_list or pre_result)
        key_individual_result = results_cache[key]
        if final_index_array is None:
            final_index_array = key_individual_result
        else:
            # Determine the operation based on the former connector
            if idx > 0 and idx - 1 < len(connector_list):
                final_index_array = _apply_connector(
                    final_index_array, key_individual_result, connector_list[idx - 1]
                )
    return final_index_array



def filter_with_numpy_vec(dataset, filter_vec_list, offset, show_progress):
    """
    Filter based on array-based SIMD computation.
    Note: Only need to start from the offset.
    """
    result_index_list = {}
    if len(filter_vec_list):
        single_ds = dataset[offset:]  # Only need to start from the offset.
        # TODO: Previously we use multi-thread here, but a little bit weird since it is a compute-intensive task.
        # Should modify to multi-process here.
        for key, (tensor, operator, value, negation) in filter_vec_list.items():
            if show_progress:
                logging.getLogger().setLevel(logging.INFO)
                logging.info(f"Computing the result of {(tensor, operator, value, negation)}...")
            target = np.full((len(single_ds), 1), value) if operator != 'LIKE' else value  # the target value.
            # Note：need to add offset here, Since the filter_based_on_numpy above is without offset
            res = filter_based_on_numpy(single_ds[tensor], target, len(single_ds), operator,
                                    negation) + offset
            result_index_list.update({key: res})
    return result_index_list


def matching(ds, tensor, value, search_type="fuzzy_match", use_local_index=True, max_workers=16):
    """
    Load the inverted_index and conduct searching.
    """
    # 1. Combine the path and metadata related variables
    branch = ds.version_state["branch"]
    try:
        meta_json = json.loads(ds.storage[os.path.join("inverted_index_dir_vec" if use_local_index else
                                                       "inverted_index_dir", branch, "meta.json")].decode('utf-8'))
    except KeyError as e:
        raise InvertedIndexNotExistsError(tensor) from e

    # 2. Get inveted index
    inverted_index = ds.get_inverted_index(tensor, vectorized=use_local_index)

    # 3. Perform search
    if use_local_index:
        try:
            use_cpp = meta_json[tensor]['use_cpp']
        except KeyError as e:
            raise InvertedIndexNotExistsError(tensor) from e
        ids = _perform_search(inverted_index, value, search_type,
                                 use_cpp,
                                 max_workers)
    else:
        ids = inverted_index.search(value, search_type=search_type)

    # 4. Post process
    return _post_process_results(ds, tensor, inverted_index, meta_json, ids)


def filter_like_batch(source, pattern, start_index):
    """
        Filter function with regular expression with a batch.
        :param source: the source np array, from the tensor column.
        :param pattern: the regex pattern.
        :param start_index: start index of this batch.
    """
    source = '@@'.join(source)
    index = re.finditer(pattern, source)
    index = [match.span()[0] for match in index]  # The start index of each element in source
    at = re.finditer(r'@@', source)
    at = [match.span()[0] for match in at]  # The index where @@ starts
    indices = np.searchsorted(at, index) + start_index  # The index of insertion
    return indices


def filter_like(source, pattern):
    """
        Filter function with regular expression multithreaded.
        :param source: the source np array, from the tensor column.
        :param pattern: the regex pattern.
    """
    length = len(source)
    futures_list = []
    with ThreadPoolExecutor(max(1, length // REGEX_BATCH_SIZE)) as executor:
        for i in range(0, length, REGEX_BATCH_SIZE):
            futures_list.append(executor.submit(filter_like_batch, source[i: i+REGEX_BATCH_SIZE], pattern, i))
    result = np.concatenate([future.result() for future in futures_list])
    return result


def _perform_search(inverted_index, value, search_type, use_cpp, max_workers):
    """Perform search"""
    if search_type == "complex_fuzzy_match":
        return inverted_index.complex_search(value, max_workers=max_workers, use_cpp=use_cpp)
    search_func = inverted_index.search_cpp if use_cpp else inverted_index.search
    return search_func(value, search_type=search_type, max_workers=max_workers)


def _post_process_results(ds, tensor, inverted_index, meta_json, ids):
    """Post process the results"""
    # Verify the consistency of the commit ids
    commit_node = ds.version_state.get("commit_node")
    index_commit_id = meta_json[tensor].get("commit_id", None)

    if commit_node.parent.commit_id != index_commit_id:
        logging.info(f"Dataset has new commit record, searching from commit id {index_commit_id}.\
                         Please update inverted index if needed.")

    if inverted_index.use_uuid:
        # map uuids to global idx
        commit_id = commit_node.parent.commit_id
        uuids = ds.get_tensor_uuids(tensor, commit_id)
        ids = {idx for idx, uuid in enumerate(uuids) if str(uuid) in ids}

    return ids


def _fetch_from_cache(
        dataset,
        condition_list,
        connector_list,
        offset,
        limit
):
    if "filter_vectorized" not in dataset.storage.upper_cache:
        # There is no filter_vec in upper_cache, so we need to create a new key for recording.
        # Note: The value of the key is an OrderedDict()
        dataset.storage.upper_cache["filter_vectorized"] = OrderedDict()
    cache_key = (str(condition_list), str(connector_list), dataset.branch)
    recompute_flag = 1
    recompute_range = (0, len(dataset))
    # The dataset has not been pop or updated, and we have the same query before.
    # Let's check whether the recorded result match the target range.
    if dataset.append_only:
        # We are in the same branch, and we hava the same cache_key with non-empty record
        is_old_query = cache_key[2] == dataset.branch and \
                       cache_key in dataset.storage.upper_cache["filter_vectorized"] and \
                       len(dataset.storage.upper_cache["filter_vectorized"][cache_key]) == 3 and \
                       dataset.storage.upper_cache["filter_vectorized"][cache_key][2] is not None
        if is_old_query:
            old_offset = dataset.storage.upper_cache["filter_vectorized"][cache_key][0]
            old_limit = dataset.storage.upper_cache["filter_vectorized"][cache_key][1]
            old_result = dataset.storage.upper_cache["filter_vectorized"][cache_key][2]

            # We have three cases! Note: The current implementation is naive. More optimization is needed.
            # Case 1: offset and limit are in the previous compute range, so we can fetch the result, no need recompute.
            if old_offset == offset and old_limit >= limit:
                new_result = old_result
                new_result = new_result[new_result >= offset][:limit]
                recompute_flag = 0
                return new_result, recompute_flag, (0, 0)
            # Case 2：partial offset and limit are in the previous compute range, so we can fetch the result.
            # But recompute is needed.
            if old_offset == offset and old_limit < limit:
                partial_result = old_result[:limit]
                recompute_range = (old_limit, limit)
                return partial_result, recompute_flag, recompute_range
            # Case 3: partial offset and limit are beyond the previous compute range. The result cannot be used.

    # A new query condition. We need to create a new key for record.
    dataset.storage.upper_cache["filter_vectorized"][cache_key] = (offset, limit, None)
    if len(dataset.storage.upper_cache["filter_vectorized"]) > FILTER_CACHE_SIZE:
        dataset.storage.upper_cache["filter_vectorized"].popitem()
    return None, recompute_flag, recompute_range


def _insert_to_cache(dataset, condition_list, connector_list, offset, limit, filtered_index):
    cache_key = (str(condition_list), str(connector_list), dataset.branch)
    if dataset.append_only and cache_key in dataset.storage.upper_cache["filter_vectorized"]:
        (old_offset, old_limit, old_index_list) = dataset.storage.upper_cache["filter_vectorized"][cache_key]
        # Case 1：No need to replace the old record in cache
        if old_offset <= offset and old_limit >= limit and old_index_list is not None:
            return
        # Case 2：The record in the current cache can be reused. We can concat it with the new results.
        if old_index_list is not None and offset == old_index_list[-1] + 1:
            new_index_list = old_index_list + filtered_index
            dataset.storage.upper_cache["filter_vectorized"][cache_key] = (old_offset, old_limit, new_index_list)
            return
    # Otherwise, update the cache. # TODO: Can we optimize this workflow?
    dataset.storage.upper_cache["filter_vectorized"][cache_key] = (offset, limit, filtered_index)


def _filter_vectorized_next(dataset, condition_list, connector_list, offset, limit, use_local_index,
                            max_workers):
    with ThreadPoolExecutor() as executor:
        result = executor.submit(
            _obtain_final_index, dataset, condition_list, connector_list, offset, limit, use_local_index,
            max_workers, False)
        _insert_to_cache(dataset, condition_list, connector_list, offset, limit, result.result())


def _optimize_and_condition(final_index_array, tensor, value, dataset, offset):
    """Helper to optimize AND conditions."""
    optimized_result = optimize_connector(final_index_array, tensor, value, dataset)
    if offset:
        optimized_result = optimized_result[optimized_result >= offset]
    return optimized_result


def _optimize_first_key(
    dataset, filter_inverted_index_list, results_cache, key,
    tensor, value, offset, use_local_index, max_workers
):
    """Helper to optimize first key if next is AND."""
    next_result = None
    if key + 1 in filter_inverted_index_list:
        next_tensor, next_op, next_val = filter_inverted_index_list[key + 1]
        next_search_type = (
            "complex_fuzzy_match" if "||" in next_val else "fuzzy_match"
        ) if next_op == "CONTAINS" else "exact_match"

        next_result = np.sort(np.array(list(
            matching(dataset, next_tensor, next_val, next_search_type, use_local_index, max_workers)
        )))
        results_cache[key + 1] = next_result

    elif key + 1 in results_cache:
        next_result = results_cache[key + 1]

    if next_result is not None:
        optimized_result = optimize_connector(next_result, tensor, value, dataset)
        if offset:
            optimized_result = optimized_result[optimized_result >= offset]
        return optimized_result, True  # (final_index_array, skip_next_key)
    return None, False


def _apply_connector(final_index_array, key_individual_result, connector):
    """Helper to apply AND/OR connector logic."""
    if connector == "AND":
        return np.intersect1d(final_index_array, key_individual_result)
    return np.union1d(final_index_array, key_individual_result)

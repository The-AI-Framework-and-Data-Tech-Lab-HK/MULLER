# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import numpy as np
import muller


def get_statistics(dataset: muller.Dataset) -> muller.Dataset:
    statistics_json = create_statistics_json(dataset)
    return statistics_json


def get_histogram(data):
    n, bins = np.histogram(data)
    histogram = {}
    histogram['hist'] = n.tolist()
    histogram['bin_edges'] = bins.tolist()
    return histogram


def get_column_stats_numpy(data):
    column_statistics = {}
    histogram = get_histogram(data)
    length = len(data)
    nan_count = np.count_nonzero(np.isnan(data))
    column_statistics['nan_count'] = nan_count
    column_statistics['nan_proportion'] = nan_count / length
    column_statistics['min'] = int(np.min(data))
    column_statistics['max'] = int(np.max(data))
    column_statistics['mean'] = float(round(np.mean(data), 5))
    column_statistics['median'] = float(np.median(data))
    column_statistics['std'] = float(round(np.std(data), 5))
    column_statistics['histogram'] = histogram

    return column_statistics


def create_statistics_json(ds):
    tensor_dict = ds.tensors
    final_json = {}
    columns = []
    for tensor_name in tensor_dict:
        statistics = {}
        tensor_object = tensor_dict[tensor_name]
        tensor_htype = tensor_object.htype
        tensor_dtype = tensor_object.dtype.name
        statistics['column_name'] = tensor_name

        if tensor_htype == 'text':
            statistics['column_type'] = 'string_text'
            strings_arr = tensor_object.numpy().flatten()
            length_arr = np.char.str_len(strings_arr)
            statistics['column_statistics'] = get_column_stats_numpy(length_arr)
        elif tensor_dtype in ('uint8','uint32'):
            statistics['column_type'] = "int"
            arr = tensor_object.numpy().flatten()
            statistics['column_statistics'] = get_column_stats_numpy(arr)
        elif tensor_dtype == 'float32':
            statistics['column_type'] = 'float'
            arr = tensor_object.numpy().flatten()
            statistics['column_statistics'] = get_column_stats_numpy(arr)
        elif tensor_htype == 'image':
            statistics['column_type'] = 'image'
            arr = [i['shape'][0] for i in ds._images_info.data()['value']]
            statistics['column_statistics'] = get_column_stats_numpy(arr)
        columns.append(statistics)
    final_json['num_examples'] = len(ds)
    final_json['statistics'] = columns
    return final_json

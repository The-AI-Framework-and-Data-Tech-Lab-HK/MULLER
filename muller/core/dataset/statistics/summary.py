# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/core/dataset/dataset.py
#
# Modifications Copyright (c) 2026 Xueling Lin

def _max_array_length(arr_max, arr_to_compare):  # helper for __str__
    for i, _ in enumerate(arr_max):
        str_length = len(arr_to_compare[i])
        if arr_max[i] < str_length:
            arr_max[i] = str_length
    return arr_max


def _get_string(table_array, max_arr):
    temp_str = ""
    for row in table_array:
        temp_str += "\n"
        for col_no, _ in enumerate(row):
            max_col = max_arr[col_no]
            length = len(row[col_no])
            starting_loc = (max_col - length) // 2
            temp_str += (" " * starting_loc + row[col_no] + " " * (max_col - length - starting_loc))
    return temp_str


def summary_dataset(dataset):
    """Produce the summary of the dataset."""
    head = ["tensor", "htype", "shape", "dtype", "compression"]
    divider = ["-------"] * 5
    tensor_dict = dataset.tensors
    max_column_length = [7, 7, 7, 7, 7]
    count = 0
    table_array = [head, divider]
    for tensor_name in tensor_dict:
        tensor_object = tensor_dict[tensor_name]

        tensor_htype = tensor_object.htype
        if tensor_htype is None:
            tensor_htype = "None"

        tensor_shape = str(tensor_object.shape_interval)

        tensor_compression = tensor_object.meta.sample_compression
        if tensor_compression is None:
            tensor_compression = "None"

        if tensor_object.dtype is None:
            tensor_dtype = "None"
        else:
            tensor_dtype = tensor_object.dtype.name

        row_array = [tensor_name, tensor_htype, tensor_shape, tensor_dtype, tensor_compression]

        table_array.append(row_array)
        max_column_length = _max_array_length(max_column_length, row_array)
        count += 1
    max_column_length = [elem + 2 for elem in max_column_length]

    return _get_string(table_array, max_column_length)

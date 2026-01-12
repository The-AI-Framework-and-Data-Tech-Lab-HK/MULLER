# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import pandas as pd


def to_dataframe(dataset, tensor_list=None, index_list=None):
    """ Export dataset as a pandas dataframe. """
    data = {}
    if tensor_list:
        target_list = tensor_list
    else:
        target_list = list(dataset.tensors)
    if index_list:
        for tensor in target_list:
            data.update({tensor: list(dataset[index_list][tensor].numpy().flatten())})  # 这里可以用numpy_continuous?
    else:
        for tensor in target_list:
            data.update({tensor: list(dataset[tensor].numpy().flatten())})  # 这里可以用numpy_full? 但有进程lock风险
    df = pd.DataFrame(data, columns=target_list)
    return df

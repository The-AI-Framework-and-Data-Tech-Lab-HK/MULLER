# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

from functools import wraps
from typing import Callable, List
import muller


def suppress_iteration_warning(my_callable: Callable):
    """Suppresses iteration warning."""
    @wraps(my_callable)
    def inner(x, *args, **kwargs):
        iteration_warning_flag = muller.constants.SHOW_ITERATION_WARNING
        muller.constants.SHOW_ITERATION_WARNING = False
        res = my_callable(x, *args, **kwargs)
        muller.constants.SHOW_ITERATION_WARNING = iteration_warning_flag
        return res

    return inner


def check_if_iteration(indexing_history: List[int], item):
    """Check if item is iteration warning."""
    is_iteration = False
    if len(indexing_history) == 10:
        step = indexing_history[1] - indexing_history[0]
        for i in range(2, len(indexing_history)):
            if indexing_history[i] - indexing_history[i - 1] != step:
                indexing_history.pop(0)
                indexing_history.append(item)
                break
        else:
            is_iteration = True
    else:
        indexing_history.append(item)
    return is_iteration

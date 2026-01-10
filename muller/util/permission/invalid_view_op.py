# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

from functools import wraps
from typing import Callable

import muller
from muller.util.exceptions import InvalidOperationError


def invalid_view_op(func: Callable):
    """Check whether a view operation is invalid."""
    @wraps(func)
    def inner(x, *args, **kwargs):
        ds = x if isinstance(x, muller.Dataset) else x.dataset
        if not ds.__dict__.get("_allow_view_updates"):
            is_del = func.__name__ == "delete"
            managed_view = ds.view_entry # managed_view = "_view_entry" in ds.__dict__
            has_vds = ds.vds # has_vds = "_vds" in ds.__dict__
            is_view = not x.index.is_trivial() or has_vds or managed_view
            if is_view and not (is_del and (has_vds or managed_view)):
                raise InvalidOperationError(
                    func.__name__,
                    type(x).__name__,
                )
        return func(x, *args, **kwargs)

    return inner

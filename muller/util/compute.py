# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/util/compute.py
#
# Modifications Copyright (c) 2026 Bingyu Liu

from muller.core.compute.provider import ComputeProvider
from muller.util.exceptions import UnsupportedSchedulerError


def get_compute_provider(
        scheduler: str = "threaded", num_workers: int = 0
) -> ComputeProvider:
    """Function to get compute provider."""
    num_workers = max(num_workers, 0)
    if scheduler == "serial" or num_workers == 0:
        from muller.core.compute.serial import SerialProvider

        compute: ComputeProvider = SerialProvider(num_workers)
    elif scheduler == "threaded":
        from muller.core.compute.thread import ThreadProvider

        compute = ThreadProvider(num_workers)
    elif scheduler == "processed":
        from muller.core.compute.process import ProcessProvider

        compute = ProcessProvider(num_workers)
    elif scheduler == "mpprocessed":
        from muller.core.compute.mpprocess import MPProcessProvider

        compute = MPProcessProvider(num_workers)
    elif scheduler == "distributed":
        from muller.core.compute.distributed import DistributedProvider

        compute = DistributedProvider(num_workers)
    else:
        raise UnsupportedSchedulerError(scheduler)
    return compute

# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import os
import logging

try:
    import ray  # type: ignore
    from ray.util.multiprocessing import Pool  # type: ignore
    from ray.util.queue import Queue  # type: ignore
except ImportError:
    logging.info("ray not found")

from muller.client.log import logger
from muller.constants import DEFAULT_MAX_TASK_RETRY_TIMES, DEFAULT_MAX_TASK_WAIT_TIME
from muller.core.compute.provider import ComputeProvider


class DistributedProvider(ComputeProvider):
    def __init__(self, workers):
        super().__init__(workers)

        if not ray.is_initialized():
            address = os.getenv("MULLER_RAY_ADDRESS")
            if address is None or address == "":
                logger.warning(
                    "you are using ray without determining the head cluster "
                    "like \"export MULLER_RAY_ADDRESS=0.0.0.0:6379\"")
                address = "auto"
            ray.init(address=address)
        self.workers = workers
        self.pool = Pool(processes=workers)

    def map(self, func, iterable):
        finished_task = set()
        futures = [self.pool.apply_async(func, args=(args,)) for args in iterable]
        results = [None for _ in range(len(futures))]
        retries = [0 for _ in range(len(futures))]
        while len(finished_task) < len(results) and sum(retries) < DEFAULT_MAX_TASK_RETRY_TIMES * len(retries):
            for index, future in enumerate(futures):
                if index in finished_task:
                    continue
                try:
                    result = future.get(timeout=DEFAULT_MAX_TASK_WAIT_TIME)
                    results[index] = result
                    finished_task.add(index)
                except ray.util.multiprocessing.TimeoutError:
                    continue
                except ray.exceptions.RayError as e:
                    logger.error(f"Task {index} fails {retries[index]} times")
                    retries[index] += 1
                    if retries[index] <= DEFAULT_MAX_TASK_RETRY_TIMES:
                        logger.info(f"Retry task {index}, retry times {retries[index]}")
                        future_recreate = self.pool.apply_async(func, args=(iterable[index],))
                        futures[index] = future_recreate
                    else:
                        logger.error(
                            f"Task {index} retry times exceed the max times, raise exception {type(e).__name__}, "
                            f"exception message\n {str(e)}")
                        raise e
                except Exception as e:
                    raise e
        return results

    def close(self):
        self.pool.close()
        self.pool.join()

    def create_queue(self):
        return Queue()

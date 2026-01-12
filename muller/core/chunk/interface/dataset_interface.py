# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/core/chunk_engine.py
#
# Modifications Copyright (c) 2026 Xueling Lin

import logging
import multiprocessing as mp
from functools import partial
from typing import Dict, List, Optional, Union

import muller
from muller.constants import DATASET_UUID_NAME
from muller.util.dataset import rechunk_one_tensor


def dataset_rechunk(
        ds,
        tensors: Optional[Union[str, List[str]]] = None,
        num_workers: int = 0,
        scheduler: str = "threaded",
        progressbar: bool = True,):
    """Rechunk the dataset."""
    if tensors is None:
        tensors = list(ds.tensors)
    elif isinstance(tensors, str):
        tensors = [tensors]

    @muller.compute
    def rechunking(sample_in, samples_out):
        for tensor in tensors:
            samples_out[tensor].extend(sample_in[tensor])

    rechunking().eval(
        ds,
        num_workers=num_workers,
        scheduler=scheduler,
        progressbar=progressbar,
        skip_ok=True,
        extend_only=True,
        disable_label_sync=True,
        disable_rechunk=True,
    )


def dataset_rechunk_if_necessary(
        ds,
        tensor_spec: Optional[Union[List[str], Dict[str, Optional[int]]]] = None,
        num_workers: int = 1
) -> None:
    """
    Rechunk the data chunks on several tensors.
    Args:
        tensor_spec(None, List[str], Dict[str, int|None]): If it is None, applies rechunking to all tensors. If it
            is a list of str, processes only the specified tensors. If a Dict, keys are tensor names and values are
            their predicted average bytes-per-sample (use 'None' if unknown).
        num_workers(int): Number of worker processes. If it is larger than 1, use multi-process for
            parallel execution.

    Returns:
        None.
    """
    if tensor_spec is None:
        tensors = [t for t in ds.tensors if not t.startswith("_")]
        avg_bps_map: Dict[str, Optional[int]] = {t: None for t in tensors}
    elif isinstance(tensor_spec, list):
        tensors = [t for t in tensor_spec if t in ds.tensors]
        avg_bps_map = {t: None for t in tensors}
    elif isinstance(tensor_spec, dict):
        tensors = [t for t in tensor_spec.keys() if t in ds.tensors]
        avg_bps_map = {t: tensor_spec[t] for t in tensors}
    else:
        raise TypeError("tensor_spec must be None, List[str] or dict[str, int|None]")

    if DATASET_UUID_NAME in ds.meta.hidden_tensors and DATASET_UUID_NAME not in tensors:
        tensors.append("_uuid")
    if DATASET_UUID_NAME in tensors:
        avg_bps_map["_uuid"] = 8  # current uuid design by default
    if not tensors:
        logging.info("No tensor to rechunk.")
        return

    if num_workers <= 1:
        for t in tensors:
            msg = rechunk_one_tensor(ds, t, avg_bps_map.get(t))
            logging.info(msg)
        return
    func = partial(rechunk_one_tensor, ds)
    args_iter = [(t, avg_bps_map.get(t)) for t in tensors]
    with mp.Pool(processes=num_workers) as pool:
        for msg in pool.starmap(func, args_iter):
            logging.info(msg)
    pool.close()
    pool.join()

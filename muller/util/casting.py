# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/util/casting.py
#
# Modifications Copyright (c) 2026 Bingyu Liu

from typing import List, Union, Sequence, Any
from functools import reduce
import numpy as np
from muller.util.exceptions import TensorDtypeMismatchError
from muller.core.sample import Sample  # type: ignore
import muller


def _get_bigger_dtype(d1, d2):
    if np.can_cast(d1, d2):
        if np.can_cast(d2, d1):
            return d1
        else:
            return d2
    else:
        if np.can_cast(d2, d1):
            return d2
        else:
            return np.object


def get_htype(val: Union[np.ndarray, Sequence, Sample]) -> str:
    """Get the htype of a non-uniform mixed dtype sequence of samples."""
    if isinstance(val, muller.core.tensor.Tensor):
        return val.meta.htype
    if hasattr(val, "shape"):  # covers numpy arrays, numpy scalars and hub samples.
        return "generic"
    if (
            isinstance(val, list)
            and len(val) > 0
            and isinstance(val[0], Sample)
    ):
        return "generic"
    types = set((map(type, val)))  # type: ignore
    if dict in types:
        return "json"
    if types == set((str,)):
        return "text"
    if object in [  # type: ignore
        np.array(x).dtype if not isinstance(x, np.ndarray) else x.dtype for x in val if x is not None  # type: ignore
    ]:
        return "json"
    return "generic"


def get_dtype(val: Union[np.ndarray, Sequence, Sample]) -> np.dtype:
    """Get the dtype of a non-uniform mixed dtype sequence of samples."""

    if hasattr(val, "dtype"):
        return np.dtype(val.dtype)  # type: ignore
    elif isinstance(val, int):
        return np.array(0).dtype
    elif isinstance(val, float):
        return np.array(0.0).dtype
    elif isinstance(val, str):
        return np.array("").dtype
    elif isinstance(val, bool):
        return np.dtype(bool)
    elif isinstance(val, Sequence):
        return reduce(_get_bigger_dtype, map(get_dtype, val))
    else:
        raise TypeError(f"Cannot infer numpy dtype for {val}")


def get_incompatible_dtype(
        samples: Union[np.ndarray, Sequence], dtype: Union[str, np.dtype]
):
    if isinstance(samples, np.ndarray):
        if samples.size == 0:
            return None
        elif samples.size == 1:
            samples = samples.reshape(1).tolist()[0]

    if isinstance(samples, (int, float, bool)) or hasattr(samples, "dtype"):
        return (
            None
            if np.can_cast(np.array([samples]).astype(dtype), dtype)
            else getattr(samples, "dtype", np.array(samples).dtype)
        )
    elif isinstance(samples, str):
        return None if dtype == np.dtype(str) else np.dtype(str)
    elif isinstance(samples, Sequence):
        for dt in map(lambda x: get_incompatible_dtype(x, dtype), samples):
            if dt:
                return dt
        return None
    else:
        raise TypeError(
            f"Unexpected object {samples}. Expected np.ndarray, int, float, bool, str or Sequence."
        )


def intelligent_cast(
    sample: Any, dtype: Union[np.dtype, str], htype: str
) -> np.ndarray:
    # TODO: docstring (note: sample can be a scalar)/statictyping
    # TODO: implement better casting here
    if isinstance(sample, Sample):
        sample = sample.array

    if hasattr(sample, "dtype") and sample.dtype == dtype:
        return sample

    err_dtype = get_incompatible_dtype(sample, dtype)
    if err_dtype:
        raise TensorDtypeMismatchError(
            dtype,
            err_dtype,
            htype,
        )

    if hasattr(sample, "astype"):  # covers both ndarrays and scalars
        return sample.astype(dtype)

    return np.array(sample, dtype=dtype)


def get_empty_text_like_sample(htype: str):
    """Get an empty sample of the given htype.

    Args:
        htype: htype of the sample.

    Returns:
        Empty sample.

    Raises:
        ValueError: if htype is not one of 'text', 'json', and 'list'.
    """
    if htype == "text":
        return ""
    elif htype == "json":
        return {}
    elif htype == "list" or htype == "tag":
        return []
    else:
        raise ValueError(
            f"This method should only be used for htypes 'text', 'json' and 'list'. Got {htype}."
        )

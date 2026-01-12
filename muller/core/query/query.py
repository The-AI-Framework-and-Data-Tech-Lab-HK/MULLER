# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/core/query/query.py
#
# Modifications Copyright (c) 2026 Bingyu Liu

import re
from typing import Any, Callable, List, Union, Optional

import numpy as np

from muller.core.dataset import Dataset
from muller.core.index import Index
from muller.core.query.io import IOBlock
from muller.core.tensor import Tensor

NP_RESULT = Union[np.ndarray, List[np.ndarray]]
NP_ACCESS = Callable[[str], NP_RESULT]


def tensor_name_match(tensor_name: str, query: str) -> bool:
    check_list = re.split(r" |=|>|<|!|%|^|&|\+|\(|\)", query)
    if tensor_name in check_list:
        return True
    return False


class DatasetQuery:
    def __init__(
            self,
            dataset,
            query: str,
            progress_callback: Callable[[int, bool], None] = lambda *_: None,
    ):
        self._dataset = dataset
        self._query = query
        self._pg_callback = progress_callback
        self._cquery = compile(query, "", "eval")
        self._tensors_name = [
            tensor
            for tensor in dataset.tensors.keys()
            if tensor_name_match(normalize_query_tensors(tensor), query)
        ]  # normalize_query_tensors(tensor) in query
        self._tensors = {tensor: self._dataset[tensor] for tensor in self._tensors_name}
        self._wrappers = self._export_tensors()
        self._groups = self._export_groups(self._wrappers)

    def execute(self, limit) -> List[int]:
        idx_map: List[int] = list()
        count = 0
        for global_idx in range(len(self._dataset)):
            if count == limit:
                break
            p = {
                tensor: self._wrap_value(tensor, self._tensors[tensor][global_idx])
                for tensor in self._tensors_name
            }
            p.update(self._groups)
            if eval(self._cquery, p):
                idx_map.append(global_idx)
                count += 1
                self._pg_callback(global_idx, True)
            else:
                self._pg_callback(global_idx, False)
        return idx_map

    def _wrap_value(self, tensor, val):
        if tensor in self._wrappers:
            return self._wrappers[tensor].with_value(val)
        return val

    def _export_tensors(self):
        return {
            tensor_key: export_tensor(tensor)
            for tensor_key, tensor in self._dataset.tensors.items()
        }

    def _export_groups(self, wrappers):
        return {
            extract_prefix(tensor_key): GroupTensor(
                self._dataset, wrappers, extract_prefix(tensor_key)
            )
            for tensor_key in self._dataset.tensors.keys()
            if "/" in tensor_key
        }


def normalize_query_tensors(tensor_key: str) -> str:
    return tensor_key.replace("/", ".")


def extract_prefix(tensor_key: str) -> str:
    return tensor_key.split("/")[0]


def export_tensor(tensor: Tensor):
    if tensor.htype == "class_label":
        return ClassLabelsTensor(tensor)

    return EvalObject()


def _get_np(dataset: Dataset, block: IOBlock) -> NP_ACCESS:
    idx = block.indices()

    def f(tensor: str) -> NP_RESULT:
        tensor_obj = dataset.tensors[tensor]
        tensor_obj.index = Index()
        return tensor_obj[idx]

    return f


class EvalObject:
    def __init__(self) -> None:
        self._val: Any = None
        self._numpy: Optional[Union[np.ndarray, List[np.ndarray]]] = None

    def __getitem__(self, item):
        r = EvalObject()
        return r.with_value(self.value[item])

    def __eq__(self, obj: object) -> bool:
        val = self.numpy_value
        obj = self._to_np(obj)
        if isinstance(val, (list, np.ndarray)):
            if isinstance(obj, (list, tuple)):
                return set(obj) == set(val)
            return obj in val
        return val == obj

    def __lt__(self, obj: object) -> bool:
        obj = self._to_np(obj)
        return self.numpy_value < obj

    def __le__(self, obj: object) -> bool:
        obj = self._to_np(obj)
        return self.numpy_value <= obj

    def __gt__(self, obj: object) -> bool:
        obj = self._to_np(obj)
        return self.numpy_value > obj

    def __ge__(self, obj: object) -> bool:
        obj = self._to_np(obj)
        return self.numpy_value >= obj

    def __mod__(self, obj: object):
        obj = self._to_np(obj)
        return self.numpy_value % obj

    def __add__(self, obj: object):
        obj = self._to_np(obj)
        return self.numpy_value + obj

    def __sub__(self, obj: object):
        obj = self._to_np(obj)
        return self.numpy_value - obj

    def __div__(self, obj: object):
        obj = self._to_np(obj)
        return self.numpy_value / obj

    def __floordiv__(self, obj: object):
        obj = self._to_np(obj)
        return self.numpy_value // obj

    def __mul__(self, obj: object):
        obj = self._to_np(obj)
        return self.numpy_value * obj

    def __pow__(self, obj: object):
        obj = self._to_np(obj)
        return self.numpy_value ** obj

    def __contains__(self, obj: object):
        obj = self._to_np(obj)
        return self.contains(obj)

    @property
    def value(self):
        return self._val

    @property
    def numpy_value(self):
        if self._numpy is None:
            self._numpy = self._val.numpy(
                aslist=self._val.is_dynamic, fetch_chunks=True
            )
        return self._numpy

    @property
    def min(self):
        """Returns np.min() for the tensor"""
        return np.amin(self.numpy_value)

    @property
    def max(self):
        """Returns np.max() for the tensor"""
        return np.amax(self.numpy_value)

    @property
    def mean(self):
        """Returns np.mean() for the tensor"""
        return self.numpy_value.mean()

    @property
    def shape(self):
        """Returns shape of the underlying numpy array"""
        return self.value.shape  # type: ignore

    @property
    def size(self):
        """Returns size of the underlying numpy array"""
        return self.value.size  # type: ignore

    @property
    def sample_info(self):
        return self._val.sample_info

    @staticmethod
    def _to_np(obj):
        if isinstance(obj, EvalObject):
            return obj.numpy_value
        return obj

    def with_value(self, v: Any):
        self._val = v
        self._numpy = None
        return self

    def contains(self, v: Any):
        return v in self.numpy_value


class GroupTensor:
    def __init__(self, dataset: Dataset, wrappers, prefix: str) -> None:
        self.prefix = prefix
        self.dataset = dataset
        self.wrappers = wrappers
        self._subgroup = self.expand()

    def __getattr__(self, __name: str) -> Any:
        return self._subgroup[self.normalize_key(__name)]

    def expand(self):
        r = {}
        for tensor in [
            self.normalize_key(t)
            for t in self.dataset.tensors
            if t.startswith(self.prefix)
        ]:
            prefix = self.prefix + "/" + extract_prefix(tensor)
            if "/" in tensor:
                r[tensor] = GroupTensor(self.dataset, self.wrappers, prefix)
            else:
                r[tensor] = self.wrappers[prefix]

        return r

    def normalize_key(self, key: str) -> str:
        return key.replace(self.prefix + "/", "")


class ClassLabelsTensor(EvalObject):
    def __init__(self, tensor: Tensor) -> None:
        super(ClassLabelsTensor, self).__init__()
        _classes = tensor.info["class_names"]  # type: ignore
        self._classes_dict = {v: idx for idx, v in enumerate(_classes)}

    def __eq__(self, obj: object) -> bool:
        try:
            obj = self._norm_labels(obj)
        except KeyError:
            return False
        return super(ClassLabelsTensor, self).__eq__(obj)

    def __lt__(self, obj: object) -> bool:
        obj = self._to_np(obj)
        if isinstance(obj, str):
            raise ValueError("label class is not comparable")
        return self.numpy_value < obj

    def __le__(self, obj: object) -> bool:
        obj = self._to_np(obj)
        if isinstance(obj, str):
            raise ValueError("label class is not comparable")
        return self.numpy_value <= obj

    def __gt__(self, obj: object) -> bool:
        obj = self._to_np(obj)
        if isinstance(obj, str):
            raise ValueError("label class is not comparable")
        return self.numpy_value > obj

    def __ge__(self, obj: object) -> bool:
        obj = self._to_np(obj)
        if isinstance(obj, str):
            raise ValueError("label class is not comparable")
        return self.numpy_value >= obj

    def contains(self, v: Any):
        v = self._to_np(v)
        if isinstance(v, str):
            v = self._classes_dict[v]
        return super(ClassLabelsTensor, self).contains(v)

    def _norm_labels(self, obj: object):
        obj = self._to_np(obj)
        if isinstance(obj, str):
            return self._classes_dict[obj]
        if isinstance(obj, int):
            return obj
        if isinstance(obj, (list, tuple)):
            return obj.__class__(map(self._norm_labels, obj))

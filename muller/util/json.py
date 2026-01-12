# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/util/json.py
#
# Modifications Copyright (c) 2026 Bingyu Liu

from typing import Any, List, Tuple, Union
import numpy as np
from numpy import ndarray
import json
import base64
from muller.util.exceptions import JsonValidationError

Schema = Any

scalars = ["int", "float", "bool", "str", "list", "dict", "ndarray", "Sample"]
types = ["Any", "Dict", "List", "Optional", "Union"]


class HubJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ndarray):
            return {
                "_hub_custom_type": "ndarray",
                "data": base64.b64encode(obj.tobytes()).decode(),
                "shape": obj.shape,
                "dtype": obj.dtype.str,
            }
        elif isinstance(obj, bytes):
            return {
                "_hub_custom_type": "bytes",
                "data": base64.b64encode(obj).decode(),
            }

        return obj


class HubJsonDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        hub_custom_type = obj.get("_hub_custom_type")
        if hub_custom_type == "ndarray":
            return np.frombuffer(
                base64.b64decode(obj["data"]), dtype=obj["dtype"]
            ).reshape(obj["shape"])
        elif hub_custom_type == "bytes":
            return base64.b64decode(obj["data"])
        return obj

def validate_json_schema(schema: str):
    _parse_schema(schema)

def _validate_schema(typ: str, params: List[str]) -> Tuple[str, List[str]]:
    if typ in scalars:
        return typ, params

    if typ not in types:
        raise InvalidJsonSchemaException(f"Unsupported type: {typ}")

    def _err(expected_num_params: int, exact: bool = False):
        raise ArgumentMismatchException(typ, len(params), expected_num_params, exact)

    if typ == "Any":
        if params:
            _err(0)
    elif typ == "Optional":
        if len(params) > 1:
            _err(1)
    elif typ == "Union":
        if len(params) == 0:
            _err(1)
    elif typ == "List":
        if len(params) > 1:
            _err(1)
    elif typ == "Dict":
        if len(params) not in (0, 2):
            _err(2, True)
    return typ, params

def _norm_type(typ: str):
    typ = typ.replace("typing.", "")
    replacements = {
        "numpy.ndarray": "ndarray",
        "np.ndarray": "ndarray",
        "muller.core.sample.Sample": "Sample",
        "muller.Sample": "Sample",
    }
    return replacements.get(typ, typ)

def _parse_schema(schema: Union[str, Schema]) -> Tuple[str, List[str]]:
    if getattr(schema, "__module__", None) == "typing":
        schema = str(schema)
        validate = False
    else:
        validate = True

    if schema in scalars:
        return schema, []

    if "[" not in schema:
        return _norm_type(schema), []

    typ, param_string = schema.split("[", 1)
    typ = _norm_type(typ)
    assert param_string[-1] == "]"
    params = []
    buff = ""
    level = 0
    for c in param_string:
        if c == "[":
            level += 1
            buff += c
        elif c == "]":
            if level == 0:
                if buff:
                    params.append(buff)
                if validate:
                    _validate_schema(typ, params)
                return typ, params
            else:
                buff += c
                level -= 1
        elif c == ",":
            if level == 0:
                params.append(buff)
                buff = ""
            else:
                buff += c
        elif c == " ":
            continue
        else:
            buff += c
    raise InvalidJsonSchemaException()


class InvalidJsonSchemaException(Exception):
    pass

class ArgumentMismatchException(InvalidJsonSchemaException):
    def __init__(self, typ: str, actual: int, expected: int, exact: bool = False):
        assert actual != expected
        gt = actual > expected
        super(ArgumentMismatchException, self).__init__(
            f"Too {'many' if gt else 'few'} parameters for {typ};"
            + f" actual {actual},expected {'exatcly' if exact else ('at most' if gt else 'at least')} {expected}."
        )


def _validate_object(obj: Any, schema: Union[str, Schema]) -> bool:
    typ, params = _parse_schema(schema)
    if typ in scalars:
        return isinstance(obj, eval(typ))
    return globals()[f"_validate_{typ.lower()}"](obj, params)


def _validate_any(obj: Any, params: List[str]):
    assert not params
    return True

def validate_json_object(obj: Any, schema: Union[str, Schema]) -> None:
    if obj and not _validate_object(obj, schema):
        raise JsonValidationError()

def _validate_list(obj: Any, params: List[str]) -> bool:
    assert len(params) <= 1
    if not isinstance(obj, (list, tuple)):
        return False
    if params:
        for item in obj:
            if not _validate_object(item, params[0]):
                return False
    return True

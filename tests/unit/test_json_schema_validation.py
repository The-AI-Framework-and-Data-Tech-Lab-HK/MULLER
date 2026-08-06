# SPDX-License-Identifier: MPL-2.0
#
# Copyright (c) 2026 Xueling Lin

"""Regression tests for muller/util/json.py schema validation.

_validate_object used to resolve scalar type names with eval() and composite
validators with a dynamic globals() lookup; both are now whitelist dicts.
The dynamic lookup also masked the fact that the Dict/Optional/Union
validators had been dropped, so those schemas crashed with KeyError.
"""

import typing

import numpy as np
import pytest

from muller.util.exceptions import JsonValidationError
from muller.util.json import (
    InvalidJsonSchemaException,
    validate_json_object,
    validate_json_schema,
)


# ---------------------------------------------------------------------------
# Scalar whitelist (replaces eval)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "schema,valid,invalid",
    [
        ("int", 3, "x"),
        ("float", 1.5, "x"),
        ("str", "x", 3),
        ("list", [1, 2], {"a": 1}),
        ("dict", {"a": 1}, [1, 2]),
        ("ndarray", np.zeros(2), [0.0, 0.0]),
    ],
)
def test_scalar_schemas(schema, valid, invalid):
    validate_json_object(valid, schema)
    with pytest.raises(JsonValidationError):
        validate_json_object(invalid, schema)


def test_bool_schema():
    # bool objects are falsy-or-truthy scalars; validate_json_object skips
    # falsy objects, so only True exercises the validator.
    validate_json_object(True, "bool")
    with pytest.raises(JsonValidationError):
        validate_json_object("yes", "bool")


def test_sample_schema_resolves():
    # eval("Sample") raised NameError (Sample was never imported here); the
    # whitelist resolves it lazily.
    from muller.core.sample import Sample

    validate_json_object(Sample(array=np.zeros((2, 2))), "Sample")
    with pytest.raises(JsonValidationError):
        validate_json_object("not a sample", "Sample")


def test_non_whitelisted_builtin_name_rejected():
    # With eval() any builtin name was resolvable; now anything outside the
    # whitelist is rejected with a clear error instead of being evaluated.
    for schema in ("object", "eval", "type", "__import__"):
        with pytest.raises(InvalidJsonSchemaException):
            validate_json_object(1, schema)


# ---------------------------------------------------------------------------
# Composite validators (previously missing -> KeyError)
# ---------------------------------------------------------------------------

def test_list_schema():
    validate_json_object([1, 2, 3], "List[int]")
    validate_json_object((1, 2, 3), "List[int]")
    with pytest.raises(JsonValidationError):
        validate_json_object([1, "x"], "List[int]")


def test_dict_schema():
    validate_json_object({"a": 1, "b": 2}, "Dict[str,int]")
    validate_json_object({"a": "anything", "b": 3}, "Dict")
    with pytest.raises(JsonValidationError):
        validate_json_object({"a": "x"}, "Dict[str,int]")
    with pytest.raises(JsonValidationError):
        validate_json_object([("a", 1)], "Dict[str,int]")


def test_optional_schema():
    validate_json_object("x", "Optional[str]")
    validate_json_object(None, "Optional[str]")
    with pytest.raises(JsonValidationError):
        validate_json_object(3, "Optional[str]")


def test_union_schema():
    validate_json_object(3, "Union[int,str]")
    validate_json_object("x", "Union[int,str]")
    with pytest.raises(JsonValidationError):
        validate_json_object(1.5, "Union[int,str]")


def test_nested_schema():
    schema = "Dict[str,List[Union[int,str]]]"
    validate_json_object({"a": [1, "x"], "b": []}, schema)
    with pytest.raises(JsonValidationError):
        validate_json_object({"a": [1.5]}, schema)


def test_typing_object_schemas():
    validate_json_object({"a": 1}, typing.Dict[str, int])
    validate_json_object([1, 2], typing.List[int])
    validate_json_object("x", typing.Optional[str])
    with pytest.raises(JsonValidationError):
        validate_json_object({"a": "x"}, typing.Dict[str, int])


def test_unsupported_composite_type_message():
    # typing objects skip _validate_schema, so the dispatch whitelist is the
    # last line of defense; it must raise a clear error, not KeyError.
    with pytest.raises(InvalidJsonSchemaException, match="Tuple"):
        validate_json_object((1,), typing.Tuple[int])


def test_validate_json_schema_still_rejects_unknown_types():
    with pytest.raises(InvalidJsonSchemaException):
        validate_json_schema("Tuple[int]")
    validate_json_schema("Dict[str,int]")

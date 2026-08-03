# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Bingyu Liu

"""Tests for muller.util.path helpers."""

import pytest

from muller.util.path import get_path_type, is_remote_path


@pytest.mark.parametrize(
    "path,expected",
    [
        ("http://example.com/ds", "http"),
        ("https://example.com/ds", "http"),
        ("huawei-obs://bucket/ds", "obs"),
        ("obs://bucket/ds", "huashan-obs"),
        ("huashan:ds", "huashan-file"),
        ("mep://ds", "mep"),
        ("roma://ds", "roma"),
        ("s3://bucket/ds", "s3"),
        ("mem://ds", "mem"),
        ("/tmp/ds", "local"),
        ("relative/ds", "local"),
        ("./ds", "local"),
    ],
)
def test_get_path_type(path, expected):
    assert get_path_type(path) == expected


@pytest.mark.parametrize(
    "path",
    [
        "mem://ds",
        "s3://bucket/ds",
        "huawei-obs://bucket/ds",
        "obs://bucket/ds",
        "roma://ds",
        "mep://ds",
        "http://example.com/ds",
    ],
)
def test_is_remote_path_true(path):
    # Storage backends that bypass the plain local filesystem (including the
    # in-memory MemoryProvider behind mem://) must be treated as remote so
    # that guards like InvertedIndexVectorized._ensure_local_for_cpp fire.
    assert is_remote_path(path) is True


@pytest.mark.parametrize("path", ["/tmp/ds", "relative/ds", "./ds"])
def test_is_remote_path_false_for_local(path):
    assert is_remote_path(path) is False

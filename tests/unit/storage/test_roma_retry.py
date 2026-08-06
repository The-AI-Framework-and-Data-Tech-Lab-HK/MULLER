# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

"""Tests for the RomaProvider retry paths.

These used to swallow every failure: a write that failed all retries returned
silently (losing data) and a failed read returned ``None``, which callers
misinterpreted as a missing key. Failures must now surface as RomaSetError /
RomaGetError, while missing keys (KeyError) are never retried.

The provider is built without ``__init__`` so no network access is needed;
only the retry logic is under test.
"""

import pytest

from muller.core.storage.roma import RomaProvider
from muller.util.exceptions import RomaGetError, RomaSetError


def _make_provider(retry_times: int = 2) -> RomaProvider:
    provider = object.__new__(RomaProvider)
    provider.root = ""
    provider.retry_times = retry_times
    provider.thread_num = 2
    return provider


def test_setitem_raises_after_exhausting_retries():
    provider = _make_provider(retry_times=2)
    calls = []

    def failing_set(key, content):
        calls.append(key)
        raise RomaSetError("boom")

    provider._set = failing_set
    with pytest.raises(RomaSetError):
        provider["some/key"] = b"payload"
    assert len(calls) == 3  # initial attempt + 2 retries


def test_setitem_succeeds_after_transient_failure():
    provider = _make_provider(retry_times=2)
    attempts = []

    def flaky_set(key, content):
        attempts.append(key)
        if len(attempts) < 3:
            raise RomaSetError("transient")

    provider._set = flaky_set
    provider["some/key"] = b"payload"  # must not raise
    assert len(attempts) == 3


def test_get_bytes_raises_after_exhausting_retries():
    provider = _make_provider(retry_times=2)
    calls = []

    def failing_get(path, start_byte=None, end_byte=None):
        calls.append(path)
        raise RomaGetError("boom")

    provider._get_bytes = failing_get
    with pytest.raises(RomaGetError):
        provider.get_bytes("some/key")
    assert len(calls) == 3


def test_get_bytes_missing_key_is_not_retried():
    provider = _make_provider(retry_times=3)
    calls = []

    def missing_get(path, start_byte=None, end_byte=None):
        calls.append(path)
        raise KeyError(path)

    provider._get_bytes = missing_get
    with pytest.raises(KeyError):
        provider.get_bytes("missing/key")
    assert len(calls) == 1  # KeyError must propagate immediately


def test_getitem_missing_key_raises_key_error():
    provider = _make_provider()

    def missing_get(path, start_byte=None, end_byte=None):
        raise KeyError(path)

    provider._get_bytes = missing_get
    with pytest.raises(KeyError):
        provider["missing/key"]


def test_get_items_returns_contents_and_strips_root():
    provider = _make_provider()
    provider.root = "base/"

    def fake_get(path):
        return (path, f"content-of-{path}".encode())

    provider._get_object_with_return_key = fake_get
    result = provider.get_items({"a", "b"})
    assert result == {"a": b"content-of-base/a", "b": b"content-of-base/b"}


def test_get_items_propagates_missing_key():
    provider = _make_provider()

    def missing_get(path):
        raise KeyError(path)

    provider._get_object_with_return_key = missing_get
    with pytest.raises(KeyError):
        provider.get_items({"a"})


def test_set_items_raises_after_exhausting_retries(monkeypatch):
    provider = _make_provider()
    calls = []

    def failing_set(path, content):
        calls.append(path)
        raise RomaSetError("boom")

    provider._set_single_obj_by_key = failing_set
    monkeypatch.setattr("muller.core.storage.roma.time.sleep", lambda seconds: None)
    with pytest.raises(RomaSetError):
        provider.set_items({"a": b"1", "b": b"2"}, multi_retry=2)
    assert len(calls) >= 2  # both batches were attempted


def test_set_items_succeeds_without_retry():
    provider = _make_provider()
    written = {}

    def working_set(path, content):
        written[path] = content

    provider._set_single_obj_by_key = working_set
    provider.set_items({"a": b"1", "b": b"2"})
    assert written == {"a": b"1", "b": b"2"}

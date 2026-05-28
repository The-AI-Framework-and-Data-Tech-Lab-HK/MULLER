# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for storage-abstraction correctness in InvertedIndexVectorized.

The C++ inverted-index engine reads and writes the local filesystem
directly (``muller::Reader`` via ``std::ifstream``, ``saveToFileNoCompression``
via ``fopen``), bypassing MULLER's ``StorageProvider`` abstraction. The
Python engine is meant to flow through ``self.storage`` exclusively.

These tests pin the two invariants that follow from that split:

  1. ``InvertedIndexVectorized.__init__`` must not crash on a
     remote-backed dataset (the legacy code unconditionally opened
     ``dataset.path + "/index_details.log"`` as a local file).
  2. Every C++ entry point must fail fast with ``UnsupportedMethod`` when
     ``dataset.path`` is remote, rather than silently writing to the wrong
     place or producing an unreachable index.
"""

from types import SimpleNamespace

import pytest

from muller.constants import FIRST_COMMIT_ID
from muller.core.query.inverted_index_vectorized import InvertedIndexVectorized
from muller.util.exceptions import UnsupportedMethod


REMOTE_PATHS = [
    "s3://bucket/dataset",
    "huawei-obs://bucket/dataset",
    "roma://bucket/dataset",
    "obs://bucket/dataset",
    "http://host/dataset",
    "https://host/dataset",
]


def _stub_dataset(path: str):
    """Minimal dataset stub: only ``.path``, ``.commit_id``, ``.version_state``
    are touched by the code paths under test."""
    return SimpleNamespace(
        path=path,
        commit_id=FIRST_COMMIT_ID,
        version_state={"branch": "main", "commit_id": FIRST_COMMIT_ID},
    )


@pytest.mark.parametrize("remote_path", REMOTE_PATHS)
def test_init_tolerates_remote_dataset_path(remote_path):
    """``__init__`` must not attempt to open a local FILTER_LOG file when
    ``dataset.path`` is remote. Previously this raised ``OSError`` and made
    even ``search()`` unreachable on S3/OBS-backed datasets."""
    inv = InvertedIndexVectorized(
        _stub_dataset(remote_path), storage=None, branch="main", column_name="text"
    )
    assert inv is not None


@pytest.mark.parametrize("remote_path", REMOTE_PATHS)
def test_ensure_local_for_cpp_raises_on_remote_paths(remote_path):
    inv = InvertedIndexVectorized(
        _stub_dataset(remote_path), storage=None, branch="main", column_name="text"
    )
    with pytest.raises(UnsupportedMethod):
        inv._ensure_local_for_cpp()


def test_ensure_local_for_cpp_allows_local_paths(tmp_path):
    inv = InvertedIndexVectorized(
        _stub_dataset(str(tmp_path)), storage=None, branch="main", column_name="text"
    )
    # Must not raise.
    inv._ensure_local_for_cpp()


@pytest.mark.parametrize("remote_path", REMOTE_PATHS)
def test_search_cpp_refuses_remote(remote_path):
    inv = InvertedIndexVectorized(
        _stub_dataset(remote_path), storage=None, branch="main", column_name="text"
    )
    with pytest.raises(UnsupportedMethod):
        inv.search_cpp("query")


@pytest.mark.parametrize("remote_path", REMOTE_PATHS)
def test_cpp_complex_search_refuses_remote(remote_path):
    inv = InvertedIndexVectorized(
        _stub_dataset(remote_path), storage=None, branch="main", column_name="text"
    )
    with pytest.raises(UnsupportedMethod):
        # Meta layout matches _obtain_meta(["num_of_shards","tokenizer","cut_all",
        # "stop_words_list","compulsory_words","case_sensitive"]); values are
        # irrelevant because the guard fires before meta is consumed.
        inv._cpp_complex_search("q", meta_data=(2, "jieba", False, [], "", False), max_workers=1)


@pytest.mark.parametrize("remote_path", REMOTE_PATHS)
def test_update_with_cpp_refuses_remote(remote_path):
    inv = InvertedIndexVectorized(
        _stub_dataset(remote_path), storage=None, branch="main", column_name="text"
    )
    with pytest.raises(UnsupportedMethod):
        inv._update_with_cpp(
            start_index=0, end_index=1, num_of_batches=1, num_of_shards=1,
            max_workers=1, stop_words=set(), compulsory_words="",
            case_sensitive=False, cut_all=False,
        )


@pytest.mark.parametrize("remote_path", REMOTE_PATHS)
def test_create_cpp_index_refuses_remote(remote_path):
    inv = InvertedIndexVectorized(
        _stub_dataset(remote_path), storage=None, branch="main", column_name="text"
    )
    with pytest.raises(UnsupportedMethod):
        inv._create_cpp_index(
            batch_params={"num_of_batches": 1, "num_of_shards": 1, "use_uuids": False},
            cut_all=False, stop_words=set(), compulsory_words=None,
            case_sensitive=False, max_workers=1,
        )

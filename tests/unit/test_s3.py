# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Bingyu Liu

import sys
import pytest

from muller.core.storage.s3 import S3Provider


def create_s3_provider():
    """create a s3 provider"""
    endpoint = "http://xxx"
    ak = "xxx"
    sk = "xxx"
    bucket_name = "xxx"
    root = "xxx"
    provider = S3Provider(endpoint=endpoint, ak=ak, sk=sk, bucket_name=bucket_name, root=root)
    return provider


@pytest.mark.skipif(("--storage" not in sys.argv and not sys.argv[-1].startswith("--storage=")) or
                    ("--storage" in sys.argv and "s3" not in sys.argv) or
                    (sys.argv[-1].startswith("--storage=") and sys.argv[-1] != "--storage=s3"),
                    reason="It should be skipped if not in s3")
def test_upload_and_download_object():
    """test upload and download s3 object"""
    provider = create_s3_provider()
    provider["test_1.txt"] = b"s3 test1"
    provider["test_2.txt"] = b"This is s3 test2"

    assert provider["test_1.txt"] == b"s3 test1"
    assert provider["test_2.txt"] == b"This is s3 test2"


@pytest.mark.skipif(("--storage" not in sys.argv and not sys.argv[-1].startswith("--storage=")) or
                    ("--storage" in sys.argv and "s3" not in sys.argv) or
                    (sys.argv[-1].startswith("--storage=") and sys.argv[-1] != "--storage=s3"),
                    reason="It should be skipped if not in s3")
def test_list_objects():
    """test list s3 objects"""
    provider = create_s3_provider()
    object_keys = provider._all_keys()
    assert object_keys == set(["test_1.txt", "test_2.txt"])


@pytest.mark.skipif(("--storage" not in sys.argv and not sys.argv[-1].startswith("--storage=")) or
                    ("--storage" in sys.argv and "s3" not in sys.argv) or
                    (sys.argv[-1].startswith("--storage=") and sys.argv[-1] != "--storage=s3"),
                    reason="It should be skipped if not in s3")
def test_delete_object():
    """test delete s3 object"""
    provider = create_s3_provider()
    del provider["test_1.txt"]

    object_keys = provider._all_keys()
    assert "test_1.txt" not in object_keys


@pytest.mark.skipif(("--storage" not in sys.argv and not sys.argv[-1].startswith("--storage=")) or
                    ("--storage" in sys.argv and "s3" not in sys.argv) or
                    (sys.argv[-1].startswith("--storage=") and sys.argv[-1] != "--storage=s3"),
                    reason="It should be skipped if not in s3")
def test_delete_objects():
    """test delete multiple s3 objects"""
    provider = create_s3_provider()
    provider.clear()

    object_keys = provider._all_keys()
    assert object_keys == set()

    provider["test_1.txt"] = b"in test_s3"
    provider["folder/test_1.txt"] = b"in folder, it's test_1"
    provider["folder/test_2.txt"] = b"in folder, it's test_2"

    provider["folder1/test_1.txt"] = b"in folder1, it's test_1"
    provider["folder1/test_2.txt"] = b"in folder1, it's test_2"

    provider.clear(prefix="folder")
    object_keys = provider._all_keys()

    has_folder1 = False
    has_test_1 = False
    for key in object_keys:
        assert not key.startswith("folder/")
        if key.startswith("folder1"):
            has_folder1 = True
        if key == "test_1.txt":
            has_test_1 = True

    assert has_folder1
    assert has_test_1

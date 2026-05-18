# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import os

os.environ.setdefault("MULLER_PYTEST_ENABLED", "true")

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--storage",
        action="store",
        default="local",
        help="The StorageProvider used for the current test.",
    )
    parser.addoption(
        "--reading_test",
        action="store",
        default="local",
        help="Whether we conducting the testing of muller.read() for objects stored in local/http/roma.",
    )
    parser.addoption(
        "--test_time_consuming",
        action="store_true",
        help="Whether performing time-consuming test cases.",
    )
    parser.addoption(
        "--vector_index_test",
        action="store",
        default="local",
        help="Whether performing the vector index test cases.",
    )


@pytest.fixture
def storage(request):
    """obtain the storage config."""
    return request.config.getoption("--storage")


@pytest.fixture
def reading_test(request):
    """obtain the reading_test config."""
    return request.config.getoption("--reading_test")


@pytest.fixture
def test_time_consuming(request):
    """obtain the test_time_consuming config."""
    return request.config.getoption("--test_time_consuming")


@pytest.fixture
def vector_index_test(request):
    """obtain the vector_index_test config."""
    return request.config.getoption("--vector_index_test")

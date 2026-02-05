# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/constants.py
#
# Modifications Copyright (c) 2026 Xueling Lin

import os

import numpy as np

BYTE_PADDING = b"\0"
DATASET_META_FILENAME = "dataset_meta.json"

# number of bytes per unit
B = 1
KB = 1024 * B
MB = 1024 * KB
GB = 1024 * MB

DEFAULT_TILING_THRESHOLD = -1  # Note: set as -1 to disable tiling
DEFAULT_MEMORY_CACHE_SIZE = 20000  # without MB multiplication, meant for the dataset API that takes cache size in MBs
DEFAULT_LOCAL_CACHE_SIZE = 0

DEFAULT_MAX_CHUNK_SIZE = 128 * MB
DELETE_SAFETY_SIZE = 8 * GB

LOCAL_CACHE_PREFIX = "~/muller/cache"

ENCODING_DTYPE = np.uint32

DATASET_LOCK_UPDATE_INTERVAL = 120  # seconds

FIRST_COMMIT_ID = "firstdbf9474d461a19e9333c2fd19b46115348f"
SAMPLE_INFO_TENSOR_MAX_CHUNK_SIZE = 4 * MB
TENSOR_COMMIT_CHUNK_MAP_FILENAME = "chunk_set"
PYTEST_ENABLED = os.environ.get("MULLER_PYTEST_ENABLED", "").lower().strip() == "true"
SPINNER_ENABLED = not PYTEST_ENABLED
SPINNER_START_DELAY = 2
CHUNKS_FOLDER = "chunks"
COMMIT_INFO_FILENAME = "commit_info.json"
VDS_INDEX = "VDS_INDEX"
DATASET_DIFF_FILENAME = "dataset_diff"
DATASET_LOCK_FILENAME = "dataset_lock.lock"
ENCODED_CREDS_FOLDER = "creds_index"
UNSHARDED_ENCODER_FILENAME = "unsharded"
ENCODED_CHUNK_NAMES_FOLDER = "chunks_index"
ENCODED_TILE_NAMES_FOLDER = "tiles_index"
TENSOR_META_FILENAME = "tensor_meta.json"
TENSOR_COMMIT_DIFF_FILENAME = "commit_diff"
VERSION_CONTROL_INFO_FILENAME = "version_control_info.json"
VERSION_CONTROL_INFO_LOCK_FILENAME = "version_control_info.lock"
QUERIES_FILENAME = "queries.json"
QUERIES_LOCK_FILENAME = "queries.lock"
LOCK_LOCAL_DATASETS = not PYTEST_ENABLED
DATASET_LOCK_VALIDITY = 300  # seconds
LOCK_VERIFY_INTERVAL = 0.5  # seconds

# Multi-user branch locking configuration
LOCK_BY_BRANCH = True  # Branch-level locking for HEAD nodes (enables multi-user collaboration)
AUTO_COMMIT_BEFORE_CHECKOUT = False  # Auto-commit uncommitted changes before checkout (opt-in feature)
REQUIRE_ADMIN_MODE = False  # Require admin mode for creator to modify others' branches (opt-in security feature)

SHOW_ITERATION_WARNING = True
WRITE_TILES_INDEX = False
TILE_ENCODER_ENABLED = False
VIEW_SUMMARY_SAFE_LIMIT = 10000
TO_DATAFRAME_SAFE_LIMIT = 100000
PARTIAL_NUM_SAMPLES = 0.5
FAST_EXTEND_BAIL = -1

# used to show maximum chunk size allowed to have during random update operation
RANDOM_MINIMAL_CHUNK_SIZE = 2 * MB
CHUNK_UPDATE_WARN_PORTION = 0.2
CONVERT_GRAYSCALE = True
CHUNK_ID_COLUMN = 0

UNSPECIFIED = "unspecified"
QUERY_PROGRESS_UPDATE_FREQUENCY = 5
TRANSFORM_PROGRESSBAR_UPDATE_INTERVAL = 5

DEFAULT_TRANSFORM_SAMPLE_CACHE_SIZE = 16

TRANSFORM_RECHUNK_AVG_SIZE_BOUND = 0.1
TRANSFORM_CHUNK_CACHE_SIZE = 64 * MB

ENABLE_RANDOM_ASSIGNMENT = True

DEFAULT_MAX_TASK_RETRY_TIMES = 1
DEFAULT_MAX_TASK_WAIT_TIME = 1

DEFAULT_MAX_SEARCH_LIMIT = 999999999
DEFAULT_MAX_NUMPY_BATCH_SIZE = 1000000

INVERTED_INDEX_BATCH_SIZE = 100000
FILTER_LOG = "index_details.log"

REGEX_BATCH_SIZE = 100000

CREATE_TENSOR_HIDDEN_UUID = False
DATASET_UUID_NAME = "_uuid"

FILTER_CACHE_SIZE = 100

MAX_WORKERS_FOR_CHUNK_ENGINE = 50

MAX_WORKERS_FOR_INVERTED_INDEX_SEARCH = 50

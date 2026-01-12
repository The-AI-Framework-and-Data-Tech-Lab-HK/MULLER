# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/util/hash.py
#
# Modifications Copyright (c) 2026 Bingyu Liu

import hashlib


def hash_inputs(*args) -> str:
    """Get hash inputs."""
    sha = hashlib.sha3_256()
    for item in args:
        sha.update(str(item).encode("utf-8"))
    return sha.hexdigest()

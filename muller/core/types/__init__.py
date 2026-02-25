# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Core type system for MULLER."""

from muller.core.types.casting import (
    get_dtype,
    get_empty_text_like_sample,
    get_htype,
    get_incompatible_dtype,
    intelligent_cast,
)
from muller.core.types.class_label import (
    convert_hash_to_idx,
    convert_to_hash,
    convert_to_idx,
    convert_to_text,
)

__all__ = [
    'convert_hash_to_idx',
    'convert_to_hash',
    'convert_to_idx',
    'convert_to_text',
    'get_dtype',
    'get_empty_text_like_sample',
    'get_htype',
    'get_incompatible_dtype',
    'intelligent_cast',
]

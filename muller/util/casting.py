# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
Deprecated: This module has been moved to muller.core.types.casting

This module is kept for backward compatibility and will be removed in a future version.
Please update your imports to use muller.core.types.casting instead.
"""

import warnings

# Re-export all functions from the new location
from muller.core.types.casting import (  # noqa: F401
    get_dtype,
    get_empty_text_like_sample,
    get_htype,
    get_incompatible_dtype,
    intelligent_cast,
)

warnings.warn(
    "muller.util.casting is deprecated and will be removed in a future version. "
    "Please use muller.core.types.casting instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    'get_dtype',
    'get_empty_text_like_sample',
    'get_htype',
    'get_incompatible_dtype',
    'intelligent_cast',
]

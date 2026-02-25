# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
Deprecated: This module has been moved to muller.core.types.class_label

This module is kept for backward compatibility and will be removed in a future version.
Please update your imports to use muller.core.types.class_label instead.
"""

import warnings

# Re-export all functions from the new location
from muller.core.types.class_label import (  # noqa: F401
    convert_hash_to_idx,
    convert_to_hash,
    convert_to_idx,
    convert_to_text,
)

warnings.warn(
    "muller.util.class_label is deprecated and will be removed in a future version. "
    "Please use muller.core.types.class_label instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    'convert_hash_to_idx',
    'convert_to_hash',
    'convert_to_idx',
    'convert_to_text',
]

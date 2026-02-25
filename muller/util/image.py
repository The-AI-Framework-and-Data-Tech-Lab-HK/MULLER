# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
Deprecated: This module has been moved to muller.core.image.processing

This module is kept for backward compatibility and will be removed in a future version.
Please update your imports to use muller.core.image.processing instead.
"""

import warnings

# Re-export all functions from the new location
from muller.core.image.processing import (  # noqa: F401
    convert_img_arr,
    convert_sample,
)

warnings.warn(
    "muller.util.image is deprecated and will be removed in a future version. "
    "Please use muller.core.image.processing instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    'convert_img_arr',
    'convert_sample',
]

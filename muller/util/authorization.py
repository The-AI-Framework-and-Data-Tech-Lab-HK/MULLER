# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin
from muller.util.sensitive_config import SensitiveConfig


def obtain_current_user():
    """Obtain the user info of the current user."""
    user_info = SensitiveConfig().uid
    return user_info

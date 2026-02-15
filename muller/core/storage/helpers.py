# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

from typing import Union, Any

from muller.core.storage.muller_memory_object import MULLERMemoryObject


def _get_nbytes(obj: Union[bytes, memoryview, MULLERMemoryObject]):
    if isinstance(obj, MULLERMemoryObject):
        return obj.nbytes
    return len(obj)


def identity(x: Any):
    """Identity function."""
    return x

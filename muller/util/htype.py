# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/util/htype.py
#
# Modifications Copyright (c) 2026 Bingyu Liu

from typing import Optional, Union
from muller.htype import HTYPE_CONFIGURATIONS
from muller.util.exceptions import TensorMetaInvalidHtype


def parse_complex_htype(htype: Optional[Union[str, None]]):
    is_sequence = False

    if not htype:
        return False, False, None

    elif htype.startswith("sequence"):
        is_sequence, _, htype = parse_sequence_start(htype)

    if htype and ("[" in htype or "]" in htype):
        raise TensorMetaInvalidHtype(htype, list(HTYPE_CONFIGURATIONS))

    return is_sequence, False, htype


def parse_sequence_start(htype):
    if htype == "sequence":
        return True, False, None
    if htype[len("sequence")] != "[" or htype[-1] != "]":
        raise TensorMetaInvalidHtype(htype, list(HTYPE_CONFIGURATIONS))
    htype = htype.split("[", 1)[1][:-1]
    if not htype:
        return True, False, None

    return True, False, htype

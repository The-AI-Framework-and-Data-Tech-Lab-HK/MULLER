# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/core/meta/encode/sequence.py
#
# Modifications Copyright (c) 2026 Bingyu Liu

from muller.core.meta.encode.byte_positions import BytePositionsEncoder
from muller.core.storage.muller_memory_object import MULLERMemoryObject
from muller.core.serialize import (
    serialize_sequence_or_creds_encoder,
    deserialize_sequence_or_creds_encoder,
)


class SequenceEncoder(BytePositionsEncoder, MULLERMemoryObject):
    @classmethod
    def frombuffer(cls, buffer: bytes, **kwargs):
        instance = cls()
        if not buffer:
            return instance

        version, ids = deserialize_sequence_or_creds_encoder(buffer, "seq")
        if ids.nbytes:
            instance._encoded = ids
        instance.version = version
        instance.is_dirty = False
        return instance

    def tobytes(self) -> memoryview:
        return memoryview(
            serialize_sequence_or_creds_encoder(self.version, self._encoded)
        )

    def pop(self, index):
        self.is_dirty = True
        super().pop(index)

# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/core/io.py
#
# Modifications Copyright (c) 2026 Xueling Lin

from random import shuffle
from typing import Dict, List, Optional

from muller.core.chunk.chunk_engine import ChunkEngine
from muller.core.storage import (
    LRUCache,
)

ChunkEngineMap = Dict[str, ChunkEngine]
CachesMap = Dict[str, LRUCache]


class IOBlock:
    """
    Represents ordered sequential read of samples from corresponding tensor chunks.

    """

    def __init__(self, chunks: List[List[Optional[str]]], indexes: List[int]) -> None:
        self._chunks: List[List[Optional[str]]] = chunks
        self._ind: List[int] = indexes

    def __len__(self) -> int:
        return len(self._ind)

    def shuffle(self):
        r"""
        Shuffle sequence in which indices would be read from the IOBlock
        """
        shuffle(self._ind)

    def chunk_names(self, tensor_index: int) -> List[Optional[str]]:
        """Obtain chunk names. """
        return self._chunks[tensor_index]

    def indices(self) -> List[int]:
        """Obtain indices. """
        return self._ind

    def chunks(self) -> List[List[Optional[str]]]:
        """Obtain chunks"""
        return self._chunks

    def split(self, n) -> List["IOBlock"]:
        """Split IOBlock into chunks. """
        k, m = divmod(len(self._ind), n)
        return [
            IOBlock(
                self._chunks, self._ind[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
            )
            for i in range(n)
        ]


class Schedule:
    def __init__(self, blocks: List[IOBlock]) -> None:
        self._blocks: List[IOBlock] = blocks

    def __iter__(self):
        return iter(self._blocks)

    def __len__(self):
        return sum(map(len, self._blocks))

    def shuffle(self) -> None:
        r"""
        Shuffle IOBlocks in the schedule as well as each IOBlock
        """
        shuffle(self._blocks)

        for block in self._blocks:
            block.shuffle()

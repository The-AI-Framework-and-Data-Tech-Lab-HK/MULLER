/*
 * Copyright (c) 2026 Bingyu Liu. All rights reserved.

 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#pragma once
#include <stdexcept>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <vector>

namespace muller {

struct Chunk
{
    /* ---------- 1. shape_info ---------- */
    std::size_t                       nShapeRows{0};
    std::size_t                       nShapeCols{0};
    std::vector<std::uint32_t>        shapeInfo;      // 行主序 (size = rows*cols)

    /* ---------- 2. byte_positions (N×3) ------ */
    // 三列：[length , offset , local_sample_index]
    using BPRow = std::array<std::uint32_t,3>;
    std::vector<BPRow> bytePositionsRaw;

    // 返回 (start,end) = (offset, offset+length)
    std::pair<std::size_t,std::size_t>
    startEnd(std::size_t localSampleIdx) const
    {
        if (bytePositionsRaw.empty())
            throw std::runtime_error("Chunk has no byte_positions rows");

        std::size_t rowIdx = 0;
        if (bytePositionsRaw.size() > 1) {           // 可改为二分查找（与python侧一致
            while (rowIdx + 1 < bytePositionsRaw.size() &&
                   bytePositionsRaw[rowIdx][2] < localSampleIdx)
            {
                ++rowIdx;
            }
        }

        const auto& row = bytePositionsRaw[rowIdx];

        std::uint32_t indexBias = 0;
        if (rowIdx >= 1)
            indexBias = bytePositionsRaw[rowIdx - 1][2] + 1;

        std::size_t numBytes   = row[0];
        std::size_t rowStart   = row[1];
        std::size_t startByte  = rowStart +                  // Python:
                                 (localSampleIdx - indexBias) * numBytes;
        return {startByte, startByte + numBytes};
    }

    /* ---------- 3. 数据区 -------------------- */
     const std::byte*           dataPtr{nullptr};   // 起始地址
     std::size_t                dataSize{0};        // 字节数
     std::vector<std::byte>     owned;              // 若 copyData=true 时持有

     bool empty() const noexcept { return dataSize == 0; }
     const std::byte* data() const noexcept { return dataPtr; }
     std::size_t      size() const noexcept { return dataSize; }

    /* ---------- 反序列化 ---------------------- */
    static Chunk deserialize(const std::byte* bytes,
                             std::size_t      nBytes,
                             bool             copyData = true,
                             bool             partial  = false);
};

} // namespace muller
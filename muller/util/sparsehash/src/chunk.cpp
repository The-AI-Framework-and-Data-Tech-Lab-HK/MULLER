/*
 * Copyright (c) 2026 Bingyu Liu. All rights reserved.

 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "muller/chunk.h"
#include <cstring>
#include <stdexcept>

namespace muller {

namespace {
/* —— 辅助：确保剩余字节足够 —— */
inline void require(std::size_t need, std::size_t have, bool partial)
{
    if (have < need) {
        if (partial)
            throw std::runtime_error("deserialize_chunk: header incomplete (partial=true)");
        throw std::runtime_error("deserialize_chunk: truncated byte stream");
    }
}
} // namespace

Chunk Chunk::deserialize(const std::byte *src,
                         std::size_t      n,
                         bool             copy,
                         bool             partial)
{
    const std::byte* cur = src;
    std::size_t      left = n;

    auto pull = [&](std::size_t k) -> const std::byte* {
        require(k, left, partial);
        const std::byte* p = cur;
        cur   += k;
        left  -= k;
        return p;
    };

    {
        require(1, left, partial);                 // 1 byte len
        std::uint8_t lenVer = static_cast<std::uint8_t>(*cur);
        ++cur;  --left;                            // consume len

        require(lenVer, left, partial);            // version bytes
        /* 可选：保存版本
           std::string ver(reinterpret_cast<const char*>(cur), lenVer);
           out.version = std::move(ver);
        */
        cur  += lenVer;
        left -= lenVer;
    }

    Chunk out;

    /* --- 1. shape_info 行数 / 列数 (2 × int32) --- */
    {
        const auto* p = pull(8);
        std::int32_t r{}, c{};
        std::memcpy(&r, p, 4);
        std::memcpy(&c, p + 4, 4);
        out.nShapeRows = static_cast<std::size_t>(r);
        out.nShapeCols = static_cast<std::size_t>(c);

        std::size_t total = out.nShapeRows * out.nShapeCols;
        if (total) {
            std::size_t bytes = total * sizeof(std::uint32_t);
            const auto* srcShape = pull(bytes);
            out.shapeInfo.resize(total);
            std::memcpy(out.shapeInfo.data(), srcShape, bytes);
        }
    }

    /* --- 2. byte_positions 行数 (uint32) --- */
    std::uint32_t bpRows{};
    {
        const auto* p = pull(4);
        std::memcpy(&bpRows, p, 4);
    }

    /* --- 3. byte_positions 数据 (rows × 3 × uint32) --- */
    if (bpRows) {
        std::size_t bytes = static_cast<std::size_t>(bpRows) * 3 * sizeof(std::uint32_t);
        const auto* srcBP = pull(bytes);
        out.bytePositionsRaw.resize(bpRows);
        std::memcpy(out.bytePositionsRaw.data(), srcBP, bytes);
    }

    /* --- 4. data 区 --- */
     if (copy) {
         out.owned.assign(cur, cur + left);
         out.dataPtr  = out.owned.data();
         out.dataSize = out.owned.size();
     } else {
         out.dataPtr  = cur;
         out.dataSize = left;
     }

    return out;
}

} // namespace muller
/*
 * Copyright (c) 2026 Bingyu Liu. All rights reserved.

 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <sstream>
#include <utility>
#include <vector>

namespace muller {

struct ChunkIdRow {
    std::uint64_t chunkId{};      // 第一列
    std::uint64_t lastSeenIndex{};/* 第二列；uint32 直接提升为 64 */
};

class ChunkIdEncoder {
public:
    /* —— 从字节反序列化 —— */
    static ChunkIdEncoder deserialize(const std::byte* bytes, std::size_t nBytes);

    /* —— 直接从磁盘文件读取 —— */
    static ChunkIdEncoder loadFromFile(const std::string&           root,
                                       const std::string&           tensorName,
                                       const std::string&           version = "",
                                       const std::string&           fileName = "unsharded");

    /* —— 查询接口 —— */
    std::size_t translateIndex(std::size_t globalSampleIdx);
    std::vector<std::pair<std::uint64_t,std::size_t>>
        get(std::size_t globalSampleIdx);
    std::size_t translateIndexRelativeToChunks(std::size_t globalSampleIdx);

    static std::string name_from_id(std::uint64_t id) {
    std::ostringstream oss;
    oss << std::hex << id;      // 转为小写十六进制
    return oss.str();
}

private:
    explicit ChunkIdEncoder(std::vector<ChunkIdRow> rows);

    std::vector<ChunkIdRow> rows_;
    std::size_t             lastRow_{0};   // 近邻缓存
};
} // namespace muller
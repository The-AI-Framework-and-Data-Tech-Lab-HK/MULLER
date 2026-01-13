/*
 * Copyright (c) 2026 Bingyu Liu. All rights reserved.

 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "muller/chunk_id_encoder.h"
#include <algorithm>
#include <fstream>
#include <sys/stat.h>
#include <stdexcept>
#include <string_view>
#include <sstream>
#include <cstring>

using namespace std::string_view_literals;

namespace muller {

/* ----------------------------------------------------------- */
/*                     反序列化实现                            */
/* ----------------------------------------------------------- */

ChunkIdEncoder
ChunkIdEncoder::deserialize(const std::byte* bytes, std::size_t nBytes)
{
    const auto need = [&](std::size_t k) {
        if (nBytes < k)
            throw std::runtime_error("chunk-id-encoder: truncated header");
    };

    /* --- 1. 读取 version 字符串长度 & 丢弃内容 --- */
    need(1);
    std::size_t lenVers = static_cast<std::uint8_t>(bytes[0]);
    need(1 + lenVers);
    const std::byte* cur  = bytes + 1 + lenVers;
    std::size_t      left = nBytes - (1 + lenVers);
    /* --- 2. 每 entry 字节数（4 或 8） --- */
    need(1 + lenVers + 1);
    std::size_t numBytesPerEntry = static_cast<std::uint8_t>(cur[0]);
    if (numBytesPerEntry != 4 && numBytesPerEntry != 8){
        throw std::runtime_error("chunk-id-encoder: num_bytes must be 4 or 8");
    }
    ++cur; --left;
    /* --- 3. 剩余部分就是 ids 数组，形状 (-1,2) --- */
    if (left % (numBytesPerEntry * 2) != 0){
        throw std::runtime_error("chunk-id-encoder: byte length not divisible by entry size");
    }
    std::size_t nRows = left / (numBytesPerEntry * 2);
    std::vector<ChunkIdRow> rows;
    rows.reserve(nRows);

    if (numBytesPerEntry == 4) {
        for (std::size_t i = 0; i < nRows; ++i) {
            std::uint32_t cid32{}, idx32{};
            std::memcpy(&cid32, cur, 4); cur += 4;
            std::memcpy(&idx32, cur, 4); cur += 4;
            rows.push_back({static_cast<std::uint64_t>(cid32),
                            static_cast<std::uint64_t>(idx32)});
        }
    } else { // 8-byte entries
        for (std::size_t i = 0; i < nRows; ++i) {
            std::uint64_t cid64{}, idx64{};
            std::memcpy(&cid64, cur, 8); cur += 8;
            std::memcpy(&idx64, cur, 8); cur += 8;
            rows.push_back({cid64, idx64});
        }
    }
    return ChunkIdEncoder(std::move(rows));
}

/* ----------------------------------------------------------- */
/*                     从文件读取                              */
/* ----------------------------------------------------------- */

ChunkIdEncoder
ChunkIdEncoder::loadFromFile(const std::string& root,
                             const std::string& version,
                             const std::string&           tensorName,
                             const std::string&           fileName)
{
    std::string full;
    if (version.empty())
        // root/tensor/chunks_index/file
        full = root + "/" + tensorName + "/chunks_index/" + fileName;
    else
        // root/version/tensor/chunks_index/file
        full = root + "versions" + "/" + version + "/" + tensorName + "/chunks_index/" + fileName;

    std::ifstream f(full, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + full);

    std::streamsize sz = f.tellg();
    f.seekg(0);
    std::vector<std::byte> buf(static_cast<std::size_t>(sz));
    f.read(reinterpret_cast<char*>(buf.data()), sz);
    return deserialize(buf.data(), buf.size());
}

/* ----------------------------------------------------------- */
/*                     查询算法                                */
/* ----------------------------------------------------------- */

ChunkIdEncoder::ChunkIdEncoder(std::vector<ChunkIdRow> rows)
    : rows_(std::move(rows)) {}

std::size_t ChunkIdEncoder::translateIndex(std::size_t globalSampleIdx) {
    auto inRow = [globalSampleIdx](const ChunkIdRow& r){
        return r.lastSeenIndex >= globalSampleIdx;
    };
    if (lastRow_ < rows_.size() && inRow(rows_[lastRow_]) &&
        (lastRow_ == 0 || rows_[lastRow_-1].lastSeenIndex < globalSampleIdx))
        return lastRow_;

    auto it = std::lower_bound(
        rows_.begin(), rows_.end(), globalSampleIdx,
        [](const ChunkIdRow& row, std::size_t value){
            return row.lastSeenIndex < value;
        });
    if (it == rows_.end())
        throw std::out_of_range("globalSampleIdx out of range");

    lastRow_ = static_cast<std::size_t>(std::distance(rows_.begin(), it));
    return lastRow_;
}

std::vector<std::pair<std::uint64_t,std::size_t>>
ChunkIdEncoder::get(std::size_t globalSampleIdx) {
    std::vector<std::pair<std::uint64_t,std::size_t>> out;
    std::size_t rowIdx = translateIndex(globalSampleIdx);
    out.emplace_back(rows_[rowIdx].chunkId, rowIdx);

    std::size_t r = rowIdx + 1;
    while (r < rows_.size() && rows_[r].lastSeenIndex == globalSampleIdx) {
        out.emplace_back(rows_[r].chunkId, r);
        ++r;
    }
    return out;
}

std::size_t
ChunkIdEncoder::translateIndexRelativeToChunks(std::size_t globalSampleIdx) {
    std::size_t rowIdx = translateIndex(globalSampleIdx);
    std::size_t prevLast = (rowIdx == 0 ? 0 : rows_[rowIdx-1].lastSeenIndex + 1);
    return globalSampleIdx - prevLast;
}

} // namespace muller
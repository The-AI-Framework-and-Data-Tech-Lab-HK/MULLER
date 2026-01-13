/*
 * Copyright (c) 2026 Bingyu Liu. All rights reserved.

 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "muller/reader.h"
#include <sstream>
#include <cstring>
#include <string>
#include <iostream>
#include <algorithm>
#include <unordered_set>

namespace muller {

void Reader::ensureEncoder()
{
    if (encoder_) return;
    encoder_.emplace(
        ChunkIdEncoder::loadFromFile(root_, version_, tensorName_));
}

const Chunk& Reader::getChunk(std::uint64_t chunkId)
{
    std::lock_guard<std::mutex> lock(cacheMutex_);

    // 检查缓存
    auto it = chunkCache_.find(chunkId);
    if (it != chunkCache_.end()) {
        // 更新访问计数和LRU
        it->second.accessCount++;
        auto lruIt = std::find(lruQueue_.begin(), lruQueue_.end(), chunkId);
        if (lruIt != lruQueue_.end()) {
            lruQueue_.erase(lruIt);
            lruQueue_.push_back(chunkId);
        }
        return it->second.chunk;
    }

    // 加载chunk
    std::string chunkName = ChunkIdEncoder::name_from_id(chunkId);
    std::string relPath = tensorName_ + "/chunks/" + chunkName;
    auto bytes = fileProvider_[relPath];

    // 反序列化（注意：这里总是copy=true以便缓存）
    Chunk chunk = Chunk::deserialize(bytes.data(), bytes.size(), true);

    // 计算大小
    std::size_t chunkSize = chunk.owned.size() +
                           chunk.shapeInfo.size() * sizeof(std::uint32_t) +
                           chunk.bytePositionsRaw.size() * sizeof(Chunk::BPRow);

    // 检查缓存大小，必要时清理
    while (currentCacheBytes_ + chunkSize > maxCacheBytes_ && !lruQueue_.empty()) {
        evictOldChunks();
    }

    // 添加到缓存
    CacheEntry entry{std::move(chunk), 1, chunkSize};
    auto result = chunkCache_.emplace(chunkId, std::move(entry));
    lruQueue_.push_back(chunkId);
    currentCacheBytes_ += chunkSize;

    return result.first->second.chunk;
}

void Reader::evictOldChunks()
{
    if (lruQueue_.empty()) return;

    // 移除最少使用的chunk
    std::uint64_t oldId = lruQueue_.front();
    lruQueue_.pop_front();

    auto it = chunkCache_.find(oldId);
    if (it != chunkCache_.end()) {
        currentCacheBytes_ -= it->second.byteSize;
        chunkCache_.erase(it);
    }
}

std::vector<std::byte> Reader::operator()(std::size_t globalSampleIdx, bool copyChunk)
{
    ensureEncoder();
    auto &enc = *encoder_;

    // 找chunkId
    auto rows = enc.get(globalSampleIdx);
    std::uint64_t chunkId = rows.front().first;

    // 从缓存获取chunk
    const Chunk& chunk = getChunk(chunkId);

    // 计算局部索引
    std::size_t local = enc.translateIndexRelativeToChunks(globalSampleIdx);
    auto [sb, eb] = chunk.startEnd(local);

    // 返回数据
    std::vector<std::byte> out(eb - sb);
    std::memcpy(out.data(), chunk.data() + sb, eb - sb);
    return out;
}

void Reader::preloadChunks(std::size_t startIdx, std::size_t endIdx)
{
    ensureEncoder();
    auto &enc = *encoder_;

    // 收集需要的所有chunk IDs
    std::unordered_set<std::uint64_t> neededChunks;
    for (std::size_t idx = startIdx; idx <= endIdx; ++idx) {
        auto rows = enc.get(idx);
        for (const auto& row : rows) {
            neededChunks.insert(row.first);
        }
    }

    // 预加载chunks
    for (std::uint64_t chunkId : neededChunks) {
        getChunk(chunkId);  // 这会自动缓存
    }
}

std::vector<std::vector<std::byte>> Reader::readBatch(
    std::size_t startIdx, std::size_t endIdx, bool copyChunk)
{
    // 先预加载
    preloadChunks(startIdx, endIdx);

    // 批量读取
    std::vector<std::vector<std::byte>> results;
    results.reserve(endIdx - startIdx + 1);

    for (std::size_t idx = startIdx; idx <= endIdx; ++idx) {
        results.push_back((*this)(idx, copyChunk));
    }

    return results;
}

void Reader::clearCache()
{
    std::lock_guard<std::mutex> lock(cacheMutex_);
    chunkCache_.clear();
    lruQueue_.clear();
    currentCacheBytes_ = 0;
}

} // namespace muller
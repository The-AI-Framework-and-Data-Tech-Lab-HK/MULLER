/*
 * Copyright (c) 2026 Bingyu Liu. All rights reserved.

 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#pragma once
#include "muller/chunk.h"
#include "muller/chunk_id_encoder.h"
#include "muller_reader/io/file_provider.h"
#include <optional>
#include <utility>
#include <unordered_map>
#include <deque>
#include <mutex>

namespace muller {

class Reader {
public:
    Reader(std::string root, std::string version, std::string tensorName)
        : root_(std::move(root)),
          version_(std::move(version)),
          fileProvider_(root_),
          tensorName_(std::move(tensorName)) {}

    /* 原有接口 */
    std::vector<std::byte> operator()(std::size_t globalSampleIdx,
                                      bool copyChunk = true);

    /* 新增批量读取接口 */
    std::vector<std::vector<std::byte>> readBatch(std::size_t startIdx,
                                                   std::size_t endIdx,
                                                   bool copyChunk = true);

    /* 预加载chunks到缓存 */
    void preloadChunks(std::size_t startIdx, std::size_t endIdx);

    /* 清理缓存 */
    void clearCache();

    /* 设置缓存大小限制（字节） */
    void setCacheLimit(std::size_t maxBytes) { maxCacheBytes_ = maxBytes; }

private:
    void ensureEncoder();

    /* 获取chunk，优先从缓存读取 */
    const Chunk& getChunk(std::uint64_t chunkId);

    /* 缓存管理 */
    void evictOldChunks();

    std::string                      root_;
    std::string                      version_;
    muller_reader::io::FileProvider     fileProvider_;
    std::string                      tensorName_;
    std::optional<ChunkIdEncoder>    encoder_;

    /* 缓存相关 */
    struct CacheEntry {
        Chunk chunk;
        std::size_t accessCount{0};
        std::size_t byteSize{0};
    };

    std::unordered_map<std::uint64_t, CacheEntry> chunkCache_;
    std::deque<std::uint64_t> lruQueue_;  // LRU队列
    std::size_t currentCacheBytes_{0};
    std::size_t maxCacheBytes_{10LL * 1024 * 1024 * 1024}; // 默认10GB
    mutable std::mutex cacheMutex_;  // 线程安全
};

} // namespace muller
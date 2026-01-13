/*
 * Copyright (c) 2026 Bingyu Liu. All rights reserved.

 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "muller/reader.h"
#include "index_processor.h"
#include "jieba_utils.h"
#include "gil_manager.h"
#include "logger.h"
#include "custom_hash_map.h"
#include "async_shard_writer.h"

#include <sys/stat.h>
#include <sys/types.h>
#include <cerrno>
#include <thread>
#include <mutex>
#include <memory>
#include <atomic>
#include <iostream>
#include <unordered_map>
#include <uuid/uuid.h>
#include <pybind11/stl.h>
#include <chrono>


namespace std {
    template <class T, class... Args>
    inline std::unique_ptr<T> my_make_unique(Args&&... args) {
        return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
    }
}

namespace py = pybind11;

namespace hashmap {

bool directory_exists(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) == 0) {
        return S_ISDIR(st.st_mode);
    }
    return false;
}

bool create_directory(const std::string& path) {
    if (directory_exists(path)) {
        return true;
    }

    if (mkdir(path.c_str(), 0755) != 0 && errno!=EEXIST) {
        return false;
    }

    return true;
}

void IndexProcessor::BlockingQueue::push(std::unique_ptr<DocBlock> blk) {
    {
        std::lock_guard<std::mutex> lk(mtx_);
        q_.push(std::move(blk));
    }
    cv_.notify_one();
}
std::unique_ptr<IndexProcessor::DocBlock> IndexProcessor::BlockingQueue::pop() {
    std::unique_lock<std::mutex> lk(mtx_);
    cv_.wait(lk, [&]{ return !q_.empty(); });
    auto blk = std::move(q_.front());
    q_.pop();
    return blk;
}

bool dump_single_shard(int                       shard_id,
                       const std::shared_ptr<CustomHashMap>& shard_ptr,
                       const std::string&        index_folder,
                       int               start)
{
    if (!shard_ptr || shard_ptr->empty()) return true;

    /* 1. 确保子目录存在 */
    std::string shard_dir = index_folder + "/" + std::to_string(shard_id);
    if (!create_directory(shard_dir)) {
        LOG_ERROR("无法创建 shard 目录: " + shard_dir);
        return false;
    }

    /* 2. 生成与 Python 一致的文件名：<shard_id>/<start> */
    std::string full_path = shard_dir + "/" + std::to_string(start);
    /* 3. 落盘 */
    try {
        shard_ptr->saveToFileNoCompression(full_path);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("saveToFile error: " + std::string(e.what()));
        return false;
    }
}


bool IndexProcessor::process_index_parallel(
        const std::string& root_path,
        const std::string& tensor_name,
        const std::string& index_folder,
        const std::string& col_log_folder,
        const std::vector<std::size_t>& starts,
        const std::vector<std::size_t>& ends,
        int  num_threads,
        int  num_of_shards,
        bool cut_all,
        bool case_sensitive,
        const std::unordered_set<std::string>& full_stop_words,
        const std::string& version,
        const std::string& compulsory_dict_path)
{
    std::string full_index_path = root_path + index_folder;
    std::string full_col_log_path = full_index_path + "/" + col_log_folder;
    if (!create_directory(full_index_path) || !create_directory(full_col_log_path))
    {
        LOG_ERROR("创建目录失败: " + full_index_path);
        return false;
    }
    if (starts.size() != ends.size()) {
        LOG_ERROR("starts / ends size mismatch"); return false;
    }
    const std::size_t task_cnt = starts.size();
    if (task_cnt == 0) return true;
    if (num_threads <= 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 1;
    }
    hashmap::JiebaUtils::getJieba(compulsory_dict_path);
    LOG_INFO("将使用 " + std::to_string(num_threads) + " 个线程" + "将使用 " + std::to_string(num_of_shards) + " 个shards");
    /* ------------- 并发区域：先释放 GIL ------------- */
    py::gil_scoped_release unlock;   // 整个函数体都无 GIL
    std::atomic<std::size_t> next_task(0);
    std::vector<std::thread> workers;
    std::atomic<bool> all_ok(true);      // 聚合返回值
    auto worker_fn = [&](){
        while (true) {
            std::size_t idx = next_task.fetch_add(1);
            if (idx >= task_cnt) break;

            bool ok = process_index_single(
                        root_path, tensor_name,
                        full_index_path, col_log_folder,
                        static_cast<int>(idx),        // batch_count
                        starts[idx], ends[idx],
                        num_of_shards,
                        cut_all, case_sensitive,
                        compulsory_dict_path,
                        version,
                        full_stop_words);
            if (!ok) all_ok = false;
        }
    };

    /* 启动线程 */
    for (int t = 0; t < num_threads; ++t)
        workers.emplace_back(worker_fn);
    for (auto& th : workers) th.join();

    return all_ok.load();
}


bool IndexProcessor::process_index_single(
                                           const std::string&        root_path,
                                           const std::string&        tensor_name,
                                           const std::string&        index_folder,
                                           const std::string&        col_log_folder,
                                           int                       batch_count,
                                           std::size_t               start,
                                           std::size_t               end,
                                           int                       num_of_shards,
                                           bool                      cut_all,
                                           bool                      case_sensitive,
                                           const std::string& compulsory_dict_path,
                                           const std::string&        version,
                                           const std::unordered_set<std::string>& full_stop_words)
{
    auto& jieba = hashmap::JiebaUtils::getJieba();
    muller::Reader reader(root_path, version, tensor_name);

    reader.setCacheLimit(150LL * 1024 * 1024 * 1024);

    // 预加载这个批次需要的chunks
    reader.preloadChunks(start, end);

    std::unordered_map<int, std::shared_ptr<CustomHashMap>> shards;

    std::vector<std::byte> buffer;
    buffer.reserve(1 << 20);
    for (std::size_t gidx = start; gidx <= end; ++gidx) {
        buffer = reader(gidx, /*copyChunk=*/false);
        std::string_view sv(reinterpret_cast<const char*>(buffer.data()),
                            buffer.size());
        if (sv.empty()) {
            continue;
        };
        /* 3.2 文本处理 */
        std::string text(sv);
        if (!case_sensitive) {
            std::transform(text.begin(), text.end(), text.begin(),
                           [](unsigned char c){ return std::tolower(c); });
        }
        /* 3.3 分词 */
        std::vector<std::string> words;
        if (cut_all)
            jieba.CutAll(text, words);
        else
            jieba.Cut(text, words);
        /* 3.4 写入倒排索引 */
        for (const std::string& w : words) {
            if (full_stop_words.find(w) != full_stop_words.end()) continue;

            auto hash_shard = hashmap::JiebaUtils::wordToInt64(w, num_of_shards);
            int64_t hash_val = hash_shard.first;
            int     shard_id = hash_shard.second;

            auto& shard_ptr = shards[shard_id];
            if (!shard_ptr) shard_ptr = std::make_shared<CustomHashMap>();
            shard_ptr->add(hash_val, static_cast<uint32_t>(gidx));
        }
    }
    reader.clearCache();
    /* ---------- 4. 最后一次 flush ---------- */
    for (auto& kv : shards)
        dump_single_shard(kv.first, kv.second, index_folder, start);
    /* ---------- 5. 生成空日志文件 ---------- */
    {
        std::string log_path = index_folder + "/" + col_log_folder + "/" +
                               std::to_string(start);
        std::ofstream ofs(log_path, std::ios::binary);  // 空文件
        if (!ofs) {
            LOG_ERROR("无法创建日志文件: " + log_path);
        }
    }
    /* ---------- 6. 完成 ---------- */
    LOG_INFO("batch " + std::to_string(batch_count) +
             " (start=" + std::to_string(start) + ") finished");
    return true;
}

}
/*
 * Copyright (c) 2026 Bingyu Liu. All rights reserved.

 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "index_utils.h"
#include "custom_hash_map.h"
#include "gil_manager.h"
#include "logger.h"
#include <experimental/filesystem>
#include <thread>
#include <mutex>
#include <atomic>
#include <iostream>

namespace fs = std::experimental::filesystem;

namespace hashmap {
namespace index_utils {

bool merge_index_files(
    const std::string& tmp_index_path,
    const std::string& optimized_index_path,
    const std::string& current_index_folder,
    const std::string& optimize_mode,
    int num_shards,
    int num_threads) {

    LOG_INFO("共有 " + std::to_string(num_shards) + " 个分片需要合并");

    if (num_threads <= 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 1;
    }

    num_threads = std::min(num_threads, num_shards);
    LOG_INFO("将使用 " + std::to_string(num_threads) + " 个线程进行合并");

    GILManager::ScopedRelease release;

    std::atomic<int> processed_shards(0);
    std::mutex console_mutex;

    // 内存池保持不变
    struct MemoryPool {
        std::mutex mutex;
        std::vector<std::shared_ptr<CustomHashMap>> available;
        std::atomic<size_t> total_memory_used{0};
        const size_t max_memory = 120ULL * 1024 * 1024 * 1024; // 120GB

        std::shared_ptr<CustomHashMap> acquire() {
            std::lock_guard<std::mutex> lock(mutex);
            if (!available.empty()) {
                auto map = available.back();
                available.pop_back();
                map->clear();
                return map;
            }
            return std::make_shared<CustomHashMap>();
        }

        void release(std::shared_ptr<CustomHashMap> map) {
            std::lock_guard<std::mutex> lock(mutex);
            size_t map_size = map->estimateMemoryUsage();
            if (total_memory_used + map_size < max_memory) {
                available.push_back(map);
                total_memory_used += map_size;
            }
        }
    };

    auto memory_pool = std::make_shared<MemoryPool>();
    auto start_time = std::chrono::steady_clock::now();
    std::atomic<size_t> total_files_processed{0};

    auto worker = [&](int thread_id, int start_shard, int end_shard) {
        for (int shard_id = start_shard; shard_id < end_shard; ++shard_id) {
            try {
                fs::path target_file = fs::path(optimized_index_path) / std::to_string(shard_id);
                if (fs::exists(target_file)) continue;

                fs::path shard_folder = fs::path(tmp_index_path) / std::to_string(shard_id);
                std::error_code ec;
                if (!fs::exists(shard_folder, ec)) continue;

                std::vector<std::string> file_list;
                for (const auto& entry : fs::directory_iterator(shard_folder, ec)) {
                    if (!ec && fs::is_regular_file(entry)) {
                        file_list.push_back(entry.path().string());
                    }
                }

                if (file_list.empty()) continue;

                if (file_list.size() == 1 && optimize_mode != "update") {
                    fs::copy_file(file_list[0], target_file, fs::copy_options::overwrite_existing, ec);
                } else {
                    auto merged_map = memory_pool->acquire();

                    // 优化逻辑：如果是 update 模式，先加载现有索引作为基础
                    if (optimize_mode == "update" && fs::exists(current_index_folder, ec) && !ec) {
                        fs::path existing_shard_file = fs::path(current_index_folder) / std::to_string(shard_id);
                        if (fs::exists(existing_shard_file, ec) && !ec) {
                            // 【恢复改动】使用您期望的大容量固定缓存
                            if (!merged_map->mmapReadOnly(existing_shard_file.string(), 10ULL * 1024 * 1024 * 1024)) { // 10GB Cache
                                merged_map->loadFromFile(existing_shard_file.string());
                            }
                        }
                    }

                    // 【核心改动】分批处理临时文件，避免同时mmap过多文件
                    const size_t batch_size = 100; // 可以根据平均文件大小和内存情况适当调整，100是一个安全且高效的选择
                    for (size_t i = 0; i < file_list.size(); i += batch_size) {
                        size_t end = std::min(i + batch_size, file_list.size());

                        std::vector<std::shared_ptr<CustomHashMap>> batch_maps;
                        batch_maps.reserve(end - i);

                        for (size_t j = i; j < end; ++j) {
                            auto temp_map = std::make_shared<CustomHashMap>();

                            // 【恢复改动】恢复使用大的固定mmap缓存大小，因为您确认内存充足
                            bool mmap_success = temp_map->mmapReadOnly(file_list[j], 20ULL * 1024 * 1024 * 1024); // 20GB Cache

                            if (!mmap_success) {
                                try {
                                    temp_map->loadFromFile(file_list[j]);
                                } catch (const std::exception& e) {
                                    std::lock_guard<std::mutex> lock(console_mutex);
                                    LOG_ERROR("Failed to load file " + file_list[j] + ": " + e.what());
                                    continue; // 跳过这个损坏或无法读取的文件
                                }
                            }
                            batch_maps.push_back(temp_map);
                        }

                        // 合并这一批文件
                        if (!batch_maps.empty()) {
                            merged_map->mergeBatch(batch_maps);
                        }
                        // `batch_maps` 在此作用域结束时自动销毁，其包含的shared_ptr引用计数减一，从而释放mmap句柄
                    }

                    merged_map->saveToFileNoCompression(target_file.string());
                    memory_pool->release(merged_map);
                    total_files_processed += file_list.size();
                }

                int completed = ++processed_shards;

                // 进度报告逻辑保持不变
                if (completed % 100 == 0 || std::chrono::steady_clock::now() - start_time > std::chrono::seconds(10)) {
                    auto elapsed = std::chrono::steady_clock::now() - start_time;
                    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
                    std::lock_guard<std::mutex> lock(console_mutex);
                    LOG_INFO("进度: " + std::to_string(completed) + "/" + std::to_string(num_shards) +
                            " (" + std::to_string(completed * 100 / num_shards) + "%) - " +
                            "速度: " + std::to_string(seconds > 0 ? completed / seconds : 0) + " shards/s, " +
                            std::to_string(seconds > 0 ? total_files_processed / seconds : 0) + " files/s");
                    start_time = std::chrono::steady_clock::now();
                }

            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(console_mutex);
                LOG_ERROR("Shard " + std::to_string(shard_id) + " 合并失败: " + std::string(e.what()));
            }
        }
    };

    // 线程创建和管理逻辑保持不变
    std::vector<std::thread> threads;
    int shards_per_thread = (num_shards + num_threads - 1) / num_threads;

    for (int i = 0; i < num_threads; ++i) {
        int start_shard = i * shards_per_thread;
        int end_shard = std::min((i + 1) * shards_per_thread, num_shards);

        if (start_shard < num_shards) {
            threads.emplace_back(worker, i, start_shard, end_shard);
        }
    }

    for (auto& thread : threads) {
        thread.join();
    }

    LOG_INFO("所有分片合并完成，共处理 " + std::to_string(total_files_processed.load()) + " 个文件");
    return true;
}

}
}
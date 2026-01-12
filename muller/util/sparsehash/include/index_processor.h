/*
 * Copyright (c) 2026 Bingyu Liu. All rights reserved.

 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#pragma once

#include "common.h"
#include "custom_hash_map.h"
#include "async_shard_writer.h"
#include <string>
#include <vector>
#include <memory>
#include <condition_variable>
#include <queue>
#include <thread>

namespace hashmap {

class IndexProcessor {
public:
    static bool MULLER_F_API process_index_parallel(
                       const std::string&        root_path,
                       const std::string&        tensor_name,
                       const std::string&        index_folder,
                       const std::string&        col_log_folder,
                       const std::vector<std::size_t>& starts,
                       const std::vector<std::size_t>& ends,
                       int                       num_threads,
                       int                       num_of_shards,
                       bool                      cut_all,
                       bool                      case_sensitive,
                       const std::unordered_set<std::string>& full_stop_words,
                       const std::string&        version = "",
                       const std::string&        compulsory_dict_path = "");

    static bool MULLER_F_API process_index_single(
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
                   const std::unordered_set<std::string>& full_stop_words);

private:
    struct DocBlock {
    std::vector<std::string> docs;   // 真正的数据
    int64_t                  offset; // 该块首行的绝对行号
    };

    class BlockingQueue {
    public:
        void push(std::unique_ptr<DocBlock> blk);
        std::unique_ptr<DocBlock> pop();
    private:
        std::queue<std::unique_ptr<DocBlock>> q_;
        std::mutex                            mtx_;
        std::condition_variable               cv_;
    };
};
}
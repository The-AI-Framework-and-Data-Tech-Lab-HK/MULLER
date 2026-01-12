/*
 * Copyright (c) 2026 Bingyu Liu. All rights reserved.

 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#pragma once
#include "custom_hash_map.h"

#include <string>
#include <thread>
#include <vector>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <memory>

namespace hashmap {

class AsyncShardWriter {
public:
    typedef std::shared_ptr<CustomHashMap> ShardPtr;

    AsyncShardWriter(const std::string& folder,
                     int   io_threads = 1,
                     size_t max_queue = 1024);

    ~AsyncShardWriter();

    void submit(int shard_id, ShardPtr shard);

    void stop();

private:
    struct Task {
        int shard_id;
        ShardPtr shard;
    };

    void worker_loop(int tid);
    void flush_file(const Task& t);

    const std::string folder_;
    const size_t      max_queue_;

    std::vector<std::thread> workers_;
    std::queue<Task>         queue_;
    std::mutex               mtx_;
    std::condition_variable  cv_full_;
    std::condition_variable  cv_empty_;
    std::atomic<bool>        stop_flag_;
};

}
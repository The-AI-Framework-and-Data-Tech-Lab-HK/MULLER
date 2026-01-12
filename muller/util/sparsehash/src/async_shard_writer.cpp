/*
 * Copyright (c) 2026 Bingyu Liu. All rights reserved.

 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "async_shard_writer.h"
#include "logger.h"

#include <uuid/uuid.h>
#include <cstdio>

namespace hashmap {

static std::string make_uuid_file(int shard_id, const std::string& folder)
{
    uuid_t uuid;
    uuid_generate(uuid);
    char buf[37];
    uuid_unparse(uuid, buf);
    return folder + "/" + std::to_string(shard_id) + "_" + std::string(buf) + ".dat";
}

AsyncShardWriter::AsyncShardWriter(const std::string& folder,
                                   int   io_threads,
                                   size_t max_queue)
    : folder_(folder),
      max_queue_(max_queue),
      stop_flag_(false)
{
    if (io_threads <= 0) {
        unsigned hw = std::thread::hardware_concurrency();
        io_threads  = hw ? static_cast<int>(hw / 2 + 1) : 1;
    }

    workers_.reserve(static_cast<size_t>(io_threads));
    for (int i = 0; i < io_threads; ++i) {
        workers_.push_back(std::thread(&AsyncShardWriter::worker_loop, this, i));
    }
}

AsyncShardWriter::~AsyncShardWriter()
{
    stop();
}

void AsyncShardWriter::stop()
{
    bool expected = false;
    if (!stop_flag_.compare_exchange_strong(expected, true)) {
        return;
    }

    cv_full_.notify_all();
    cv_empty_.notify_all();

    for (size_t i = 0; i < workers_.size(); ++i) {
        if (workers_[i].joinable())
            workers_[i].join();
    }
}

void AsyncShardWriter::submit(int shard_id, ShardPtr shard)
{
    if (!shard || shard->empty())
        return;

    std::unique_lock<std::mutex> lk(mtx_);
    cv_full_.wait(lk, [this] { return queue_.size() < max_queue_ || stop_flag_.load(); });

    if (stop_flag_)
        return;

    Task task;
    task.shard_id = shard_id;
    task.shard    = shard;
    queue_.push(task);

    lk.unlock();
    cv_empty_.notify_one();
}

void AsyncShardWriter::worker_loop(int tid)
{
    while (true) {
        Task task;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_empty_.wait(lk, [this] { return stop_flag_.load() || !queue_.empty(); });

            if (stop_flag_ && queue_.empty())
                break;

            task = queue_.front();
            queue_.pop();
            lk.unlock();
            cv_full_.notify_one();
        }

        try {
            flush_file(task);
        } catch (const std::exception& e) {
            LOG_ERROR("I/O thread " + std::to_string(tid) +
                      " write shard failed: " + e.what());
        }
    }
}

void AsyncShardWriter::flush_file(const Task& t)
{
    const std::string path = make_uuid_file(t.shard_id, folder_);
    t.shard->saveToFile(path);
}

} // namespace hashmap
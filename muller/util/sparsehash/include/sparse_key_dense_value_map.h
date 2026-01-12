/*
 * Copyright (c) 2026 Bingyu Liu. All rights reserved.

 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <array>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>
#include <cstring>

template<typename KeyType, typename ValueType>
class SparseKeyDenseValueMap {
private:
    // 内存块，避免频繁分配
    struct MemoryBlock {
        static constexpr size_t BLOCK_SIZE = 1 << 20;  // 1M个元素
        std::unique_ptr<ValueType[]> data;
        size_t used = 0;
        std::unique_ptr<MemoryBlock> next;

        MemoryBlock() : data(std::make_unique<ValueType[]>(BLOCK_SIZE)) {}

        bool has_space() const { return used < BLOCK_SIZE; }

        ValueType* add(ValueType value) {
            data[used] = value;
            return &data[used++];
        }
    };

    // 每个key对应的数据
    struct KeyData {
        int shard_id = -1;
        std::unique_ptr<MemoryBlock> first_block;
        MemoryBlock* current_block = nullptr;
        size_t total_count = 0;

        void add(ValueType value) {
            if (!first_block || !current_block->has_space()) {
                auto new_block = std::make_unique<MemoryBlock>();
                if (current_block) {
                    current_block->next = std::move(new_block);
                    current_block = current_block->next.get();
                } else {
                    first_block = std::move(new_block);
                    current_block = first_block.get();
                }
            }
            current_block->add(value);
            total_count++;
        }

        // 高效转换为vector - 一次性分配正确大小
        std::vector<ValueType> to_vector() const {
            std::vector<ValueType> result(total_count);  // 一次分配

            size_t idx = 0;
            for (auto* block = first_block.get(); block; block = block->next.get()) {
                std::memcpy(&result[idx], block->data.get(),
                           block->used * sizeof(ValueType));
                idx += block->used;
            }

            return result;
        }

        // 移动语义版本，避免拷贝
        void move_to_vector(std::vector<ValueType>& result) {
            result.clear();
            result.resize(total_count);

            size_t idx = 0;
            for (auto* block = first_block.get(); block; block = block->next.get()) {
                std::memcpy(&result[idx], block->data.get(),
                           block->used * sizeof(ValueType));
                idx += block->used;
            }
        }
    };

    // 使用数组存储，因为key很少
    static constexpr size_t MAX_KEYS = 300;
    std::array<std::optional<KeyData>, MAX_KEYS> data_array;
    std::unordered_map<KeyType, size_t> key_to_index;
    size_t next_index = 0;

public:
    // 添加值 - O(1) 操作
    void add(KeyType key, ValueType value, int shard_id = 0) {
        auto it = key_to_index.find(key);
        size_t idx;

        if (it == key_to_index.end()) {
            idx = next_index++;
            key_to_index[key] = idx;
            data_array[idx] = KeyData();
            data_array[idx]->shard_id = shard_id;
        } else {
            idx = it->second;
        }

        data_array[idx]->add(value);
    }

    // 获取某个key的所有值
    std::vector<ValueType> get_values(KeyType key) const {
        auto it = key_to_index.find(key);
        if (it == key_to_index.end()) {
            return {};
        }
        return data_array[it->second]->to_vector();
    }

    // 遍历所有key
    template<typename Func>
    void for_each_key(Func func) const {
        for (const auto& [key, idx] : key_to_index) {
            const auto& data = data_array[idx];
            func(key, data->shard_id, data->to_vector());
        }
    }

    // 获取统计信息
    size_t key_count() const { return key_to_index.size(); }
    size_t total_values() const {
        size_t total = 0;
        for (size_t i = 0; i < next_index; ++i) {
            total += data_array[i]->total_count;
        }
        return total;
    }
};
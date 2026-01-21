/*
 * Copyright (c) 2026 Bingyu Liu. All rights reserved.

 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#pragma once
#include <unordered_map>
#include <vector>
#include <stdint.h>
#include <memory>
#include <string>
#include <list>
#include <sys/mman.h>  // Add mmap related headers

namespace hashmap {

class CustomHashMap {
public:
    typedef int64_t                       KeyType;
    typedef int64_t                       ValType;
    typedef std::vector<ValType>          TValueSet;

    CustomHashMap();
    ~CustomHashMap();

    void add(KeyType key, ValType value);
    void add_batch(KeyType key, const TValueSet& values);
    void set_values(KeyType key, const TValueSet& values) {
    map_[key] = std::make_shared<TValueSet>(values);
}
    void merge(const CustomHashMap& other);
    void saveToFile(const std::string& filename);

    void saveToFileNoCompression(const std::string& filename);

    // 清空内容 - 声明而不是直接定义
    void clear();

    // 批量合并优化
    void mergeBatch(const std::vector<std::shared_ptr<CustomHashMap>>& others);

    // 预分配空间
    void reserve(size_t expectedKeys) { map_.reserve(expectedKeys); }

    // 获取当前大小（用于内存估算）
    size_t estimateMemoryUsage() const {
        size_t total = sizeof(*this);
        total += map_.size() * (sizeof(KeyType) + sizeof(void*));
        for (const auto& kv : map_) {
            total += kv.second->capacity() * sizeof(ValType);
        }
        return total;
    }

    void mergeLazy(const CustomHashMap& other);

    void loadFromFile(const std::string& filename);
    bool loadFromPickleFile(const std::string& filename);

    bool mmapReadOnly(const std::string& filename, size_t max_cache_bytes = 4*1024*1024);

    bool find(KeyType key) const;
    std::shared_ptr<const TValueSet> getValuesByKey(KeyType key) const;

    inline size_t size()  const { return map_.size(); }
    inline bool   empty() const { return map_.empty(); }

    typedef std::unordered_map<KeyType, std::shared_ptr<TValueSet>>::const_iterator
            iterator;
    iterator begin() const;
    iterator end()   const;

private:
    static void     writeVarInt(FILE* fp, int64_t value);
    static int64_t  readVarInt (FILE* fp);
    static uint8_t  getIntCategory(int64_t value);

    struct ListMeta {
        uint32_t offset;
        uint32_t bytes;
        uint32_t size;
    };
    bool lazy_mode_ = false;
    std::unordered_map<KeyType, ListMeta> lazy_index_;
    mutable std::unordered_map<KeyType, std::shared_ptr<TValueSet>> lazy_cache_;
    void*  file_data_ = nullptr;
    size_t file_len_  = 0;
    size_t lazy_cache_cap_;
    mutable std::list<KeyType> lru_list_;
    void   touch(KeyType k) const;
    void   evict_if_needed() const;
    void loadFromFileV1(FILE* fp);
    void loadFromFileNoCompression(FILE* fp);

    typedef std::unordered_map<KeyType, std::shared_ptr<TValueSet> > MapType;
    MapType map_;
};

} // namespace
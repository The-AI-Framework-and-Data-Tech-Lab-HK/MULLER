/*
 * Copyright (c) 2026 Bingyu Liu. All rights reserved.

 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "custom_hash_map.h"
#include "logger.h"
#include <Python.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <list>
#include <sys/stat.h>
#include <fstream>
#include <unordered_set>
#include <sstream>



namespace hashmap {

CustomHashMap::CustomHashMap() {
}

CustomHashMap::~CustomHashMap() {
    if (file_data_) ::munmap(file_data_, file_len_);
}

void CustomHashMap::add(KeyType key, ValType value) {
    auto it = map_.find(key);
    if (it == map_.end()) {
        map_[key] = std::make_shared<TValueSet>();
        map_[key]->push_back(value);
    } else {
        auto& vec = *(it->second);
        if (vec.empty() || vec.back() != value) {
            vec.push_back(value);
        }
    }
}

void CustomHashMap::add_batch(KeyType key, const TValueSet& values) {
    if (values.empty()) {
        return;
    }

    auto it = map_.find(key);
    if (it == map_.end()) {
        map_[key] = std::make_shared<TValueSet>(values);
    } else {
        TValueSet& target_values = *(it->second);

        target_values.reserve(target_values.size() + values.size());

        for (const ValType& val : values) {
            if (std::find(target_values.begin(), target_values.end(), val) == target_values.end()) {
                target_values.push_back(val);
            }
        }
    }
}

void CustomHashMap::clear() {
    map_.clear();
    if (file_data_) {
        ::munmap(file_data_, file_len_);
        file_data_ = nullptr;
        file_len_ = 0;
    }
    lazy_index_.clear();
    lazy_cache_.clear();
    lru_list_.clear();
    // 注意：currentCacheBytes_ 在原代码中没有定义，应该使用现有的成员
    lazy_cache_cap_ = 0;
    lazy_mode_ = false;
}

// 实现mergeBatch方法
void CustomHashMap::mergeBatch(const std::vector<std::shared_ptr<CustomHashMap>>& others) {
    size_t totalKeys = map_.size();
    for (const auto& other : others) {
        totalKeys += other->size();
    }
    map_.reserve(totalKeys);

    for (const auto& other : others) {
        // 统一处理 lazy 和非 lazy 模式
        auto process_kv = [&](KeyType key, const std::shared_ptr<const TValueSet>& other_values_ptr) {
            if (!other_values_ptr || other_values_ptr->empty()) return;

            auto it = map_.find(key);
            if (it == map_.end()) {
                // Key 不存在，直接创建一个副本并插入
                map_[key] = std::make_shared<TValueSet>(*other_values_ptr);
            } else {
                // Key 已存在，执行高效的有序归并去重
                TValueSet& target_values = *(it->second);
                const TValueSet& other_values = *other_values_ptr;
                target_values.insert(target_values.end(), other_values.begin(), other_values.end());
            }
        };

        if (other->lazy_mode_) {
            for (const auto& kv : other->lazy_index_) {
                process_kv(kv.first, other->getValuesByKey(kv.first));
            }
        } else {
            for (const auto& kv : other->map_) {
                process_kv(kv.first, kv.second);
            }
        }
    }
}

void CustomHashMap::merge(const CustomHashMap& other) {
    for (auto it = other.begin(); it != other.end(); ++it) {
        KeyType key = it->first;
        const TValueSet& values = *(it->second);

        add_batch(key, values);
    }
}

bool CustomHashMap::find(KeyType key) const
{
    if (lazy_mode_) return lazy_index_.find(key) != lazy_index_.end();
    return map_.find(key) != map_.end();
}

std::shared_ptr<const CustomHashMap::TValueSet>
CustomHashMap::getValuesByKey(KeyType key) const
{
    if (!lazy_mode_) {
        auto it = map_.find(key);
        return (it == map_.end()) ? nullptr : it->second;
    }

    auto itc = lazy_cache_.find(key);
    if (itc != lazy_cache_.end()) {
        touch(key);
        return itc->second;
    }

    auto it = lazy_index_.find(key);
    if (it == lazy_index_.end()) return nullptr;

    const ListMeta& meta = it->second;
    const uint8_t* ptr = (uint8_t*)file_data_ + meta.offset;
    const uint8_t* cur = ptr;

    TValueSet* vec = new TValueSet;
    vec->reserve(meta.size);

    auto readVar = [&](const uint8_t*& p)->int64_t{
        uint64_t res=0; int shift=0; uint8_t byte;
        do{ byte=*p++; res |= uint64_t(byte & 0x7f)<<shift; shift+=7; }while(byte&0x80);
        return (res>>1) ^ (-(res&1));
    };

    if (meta.size){
        ValType prev = readVar(cur);
        vec->push_back(prev);
        for(size_t i=1;i<meta.size;++i){
            ValType d = readVar(cur);
            ValType v = prev + d;
            vec->push_back(v);
            prev = v;
        }
    }

    std::shared_ptr<TValueSet> sp(vec);
    lazy_cache_[key] = sp;
    touch(key);
    evict_if_needed();
    return sp;
}

void CustomHashMap::touch(KeyType k) const {
    lru_list_.remove(k);
    lru_list_.push_front(k);
}
void CustomHashMap::evict_if_needed() const {
    size_t bytes = 0;
    for(auto& kv:lazy_cache_) bytes += kv.second->capacity()*sizeof(ValType);
    while(bytes > lazy_cache_cap_ && !lru_list_.empty()){
        KeyType victim = lru_list_.back(); lru_list_.pop_back();
        auto it = lazy_cache_.find(victim);
        if(it!=lazy_cache_.end()){
            bytes -= it->second->capacity()*sizeof(ValType);
            lazy_cache_.erase(it);
        }
    }
}

void CustomHashMap::writeVarInt(FILE* fp, int64_t value) {
    uint64_t zigzag = (value << 1) ^ (value >> 63);

    do {
        uint8_t byte = zigzag & 0x7F;
        zigzag >>= 7;
        if (zigzag) byte |= 0x80;
        fwrite(&byte, 1, 1, fp);
    } while (zigzag);
}

int64_t CustomHashMap::readVarInt(FILE* fp) {
    uint64_t result = 0;
    uint8_t byte;
    int shift = 0;

    do {
        if (fread(&byte, 1, 1, fp) != 1) {
            throw std::runtime_error("Failed to read VarInt");
        }

        result |= (uint64_t)(byte & 0x7F) << shift;
        shift += 7;
    } while (byte & 0x80);

    return (result >> 1) ^ (-(result & 1));
}

uint8_t CustomHashMap::getIntCategory(int64_t value) {
    if (value >= -128 && value <= 127) return 1;
    if (value >= -32768 && value <= 32767) return 2;
    if (value >= -8388608 && value <= 8388607) return 3;
    if (value >= -2147483648LL && value <= 2147483647LL) return 4;
    return 8;
}

void CustomHashMap::saveToFile(const std::string& filename) {
    FILE* fp = fopen(filename.c_str(), "wb");
    if (!fp) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }
    const uint8_t FORMAT_VERSION = 1;
    fwrite(&FORMAT_VERSION, 1, 1, fp);

    writeVarInt(fp, map_.size());

    // 仍然对键进行排序（为了压缩效果）
    std::vector<KeyType> keys;
    keys.reserve(map_.size());
    for (const auto& pair : map_) {
        keys.push_back(pair.first);
    }
    std::sort(keys.begin(), keys.end());

    KeyType prev_key = 0;
    for (const KeyType& key : keys) {
        writeVarInt(fp, key - prev_key);
        prev_key = key;

        const TValueSet& valueSet = *map_.find(key)->second;
        writeVarInt(fp, valueSet.size());

        if (!valueSet.empty()) {
            // 直接使用原始数据，不再排序
            writeVarInt(fp, valueSet[0]);
            for (size_t i = 1; i < valueSet.size(); ++i) {
                int64_t delta = valueSet[i] - valueSet[i-1];
                writeVarInt(fp, delta);
            }
        }
    }

    fclose(fp);
}

void CustomHashMap::saveToFileNoCompression(const std::string& filename) {
    FILE* fp = fopen(filename.c_str(), "wb");
    if (!fp) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    // 定义一个新的格式版本号，例如 2，用于区分旧的压缩格式(v1)
    const uint8_t FORMAT_VERSION = 2;
    fwrite(&FORMAT_VERSION, sizeof(uint8_t), 1, fp);

    // 直接写入 map 的大小
    int64_t mapSize = map_.size();
    fwrite(&mapSize, sizeof(int64_t), 1, fp);

    // 遍历 map，不进行任何排序
    for (const auto& pair : map_) {
        const KeyType key = pair.first;
        const TValueSet& valueSet = *pair.second;

        // 1. 直接写入原始 key
        fwrite(&key, sizeof(KeyType), 1, fp);

        // 2. 写入 value 列表的大小
        int64_t valueSetSize = valueSet.size();
        fwrite(&valueSetSize, sizeof(int64_t), 1, fp);

        // 3. 直接写入整个 value 列表的二进制数据
        if (valueSetSize > 0) {
            fwrite(valueSet.data(), sizeof(ValType), valueSetSize, fp);
        }
    }

    fclose(fp);
}

void CustomHashMap::loadFromFile(const std::string& filename) {
    // 1. 状态清理：这部分完全来自您提供的原始函数，确保 mmap 被正确处理
    lazy_mode_ = false;
    if (file_data_) {
        ::munmap(file_data_, file_len_);
        file_data_ = nullptr;
        file_len_ = 0; // 好习惯：重置长度
    }
    lazy_index_.clear();
    lazy_cache_.clear();
    map_.clear();

    // 2. 打开文件
    FILE* fp = fopen(filename.c_str(), "rb");
    if (!fp) {
        std::cerr << "open fail " << filename << std::endl;
        return;
    }

    try {
        // 3. 读取版本号进行调度
        uint8_t version;
        size_t bytes_read = fread(&version, sizeof(uint8_t), 1, fp);

        if (bytes_read != 1) {
            if (feof(fp)) { // 文件为空，是有效的空 map
                fclose(fp);
                return;
            }
            throw std::runtime_error("Failed to read format version.");
        }

        if (version == 1) {
            // 调用 V1 加载器
            loadFromFileV1(fp);
        } else if (version == 2) {
            // 调用 V2 加载器
            loadFromFileNoCompression(fp);
        } else {
            // 未知版本
            char error_msg[100];
            snprintf(error_msg, sizeof(error_msg), "Unsupported file format version: %u", version);
            throw std::runtime_error(error_msg);
        }

    } catch (const std::exception& e) {
        std::cerr << "Load error for file '" << filename << "': " << e.what() << std::endl;
        fclose(fp); // 确保异常时关闭文件
        throw;      // 重新抛出异常，让调用者知道失败了
    }

    fclose(fp);
}

void CustomHashMap::loadFromFileNoCompression(FILE* fp) {
    int64_t mapSize;
    if (fread(&mapSize, sizeof(int64_t), 1, fp) != 1) {
        throw std::runtime_error("V2: Failed to read mapSize");
    }

    map_.reserve(mapSize);

    for (int64_t i = 0; i < mapSize; ++i) {
        KeyType key;
        if (fread(&key, sizeof(KeyType), 1, fp) != 1) {
            throw std::runtime_error("V2: Failed to read key");
        }

        int64_t valueSetSize;
        if (fread(&valueSetSize, sizeof(int64_t), 1, fp) != 1) {
            throw std::runtime_error("V2: Failed to read valueSetSize");
        }

        auto vs = std::make_shared<TValueSet>(valueSetSize);
        if (valueSetSize > 0) {
            if (fread(vs->data(), sizeof(ValType), valueSetSize, fp) != (size_t)valueSetSize) {
                throw std::runtime_error("V2: Failed to read values data");
            }
        }
        map_[key] = vs;
    }
}

void CustomHashMap::loadFromFileV1(FILE* fp) {
    // 读取 map 大小
    size_t mapSize = readVarInt(fp);
    map_.reserve(mapSize);

    KeyType prev_key = 0;
    for (size_t i = 0; i < mapSize; ++i) {
        int64_t key_delta = readVarInt(fp);
        KeyType key = prev_key + key_delta;
        prev_key = key;

        size_t setSize = readVarInt(fp);
        auto vs = std::make_shared<TValueSet>();
        vs->reserve(setSize);

        if (setSize > 0) {
            ValType prev_val = readVarInt(fp);
            vs->push_back(prev_val);
            for (size_t j = 1; j < setSize; ++j) {
                int64_t d = readVarInt(fp);
                ValType v = prev_val + d;
                vs->push_back(v);
                prev_val = v;
            }
        }
        map_[key] = vs;
    }
}

bool hashmap::CustomHashMap::loadFromPickleFile(const std::string& filename)
{
    lazy_mode_ = false;
    if (file_data_) { ::munmap(file_data_, file_len_); file_data_ = nullptr; }
    lazy_index_.clear(); lazy_cache_.clear();
    map_.clear();

    std::ifstream ifs(filename, std::ios::binary | std::ios::ate);
    if (!ifs) {
        std::cerr << "open fail " << filename << std::endl;
        return false;
    }
    std::streamsize len = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::vector<char> buffer(len);
    if (!ifs.read(buffer.data(), len)) {
        std::cerr << "read fail " << filename << std::endl;
        return false;
    }

    if (!Py_IsInitialized()) {
    Py_Initialize();
//    PyEval_InitThreads();
}

    PyGILState_STATE gstate = PyGILState_Ensure();

    bool ok = false;
    PyObject *bytesObj = nullptr, *pickleMod = nullptr,
             *loadsFunc = nullptr, *pyDict = nullptr, *items = nullptr;

    try {
        if (buffer.empty()) throw std::runtime_error("Empty buffer");

        bytesObj = PyBytes_FromStringAndSize(buffer.data(), len);
        if (!bytesObj) throw std::runtime_error("PyBytes_FromStringAndSize");

        pickleMod = PyImport_ImportModule("pickle");
        if (!pickleMod) throw std::runtime_error("import pickle");

        loadsFunc = PyObject_GetAttrString(pickleMod, "loads");
        if (!loadsFunc || !PyCallable_Check(loadsFunc))
            throw std::runtime_error("pickle.loads missing");

        pyDict = PyObject_CallFunctionObjArgs(loadsFunc, bytesObj, nullptr);
        if (!pyDict || !PyDict_Check(pyDict))
            throw std::runtime_error("unpickled object is not dict");

        items = PyDict_Items(pyDict);
        Py_ssize_t n = PyList_Size(items);

        map_.reserve(static_cast<size_t>(n));

        for (Py_ssize_t i = 0; i < n; ++i) {
            PyObject* tup   = PyList_GetItem(items, i);
            PyObject* pyKey = PyTuple_GetItem(tup, 0);
            PyObject* pyVal = PyTuple_GetItem(tup, 1);

            if (!PyLong_Check(pyKey) || !PyList_Check(pyVal))
                continue;

            KeyType key = static_cast<KeyType>(PyLong_AsLongLong(pyKey));

            std::shared_ptr<TValueSet> vec(new TValueSet);
            Py_ssize_t m = PyList_Size(pyVal);
            vec->reserve(static_cast<size_t>(m));

            for (Py_ssize_t j = 0; j < m; ++j) {
                PyObject* pyItem = PyList_GetItem(pyVal, j);
                if (!PyLong_Check(pyItem)) continue;
                ValType v = static_cast<ValType>(PyLong_AsLongLong(pyItem));
                vec->push_back(v);
            }
            map_[key] = std::move(vec);
        }
        ok = true;
    }
    catch (const std::exception& e) {
        std::cerr << "loadFromPickleFile error: " << e.what() << std::endl;
    }

    Py_XDECREF(items);
    Py_XDECREF(pyDict);
    Py_XDECREF(loadsFunc);
    Py_XDECREF(pickleMod);
    Py_XDECREF(bytesObj);

    PyGILState_Release(gstate);

    return ok;
}

bool CustomHashMap::mmapReadOnly(const std::string& filename, size_t max_cache_bytes)
{
    int fd = ::open(filename.c_str(), O_RDONLY);
    if (fd < 0) {
        perror("open");
        return false;
    }

    struct stat st;
    if (::fstat(fd, &st) != 0) {
        perror("stat");
        ::close(fd);
        return false;
    }

    file_len_ = st.st_size;
    // 处理空文件的情况
    if (file_len_ == 0) {
        ::close(fd);
        lazy_mode_ = true; // 视为空的懒加载模式
        lazy_cache_cap_ = max_cache_bytes;
        return true;
    }

    file_data_ = ::mmap(nullptr, file_len_, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (file_data_ == MAP_FAILED) {
        perror("mmap");
        file_data_ = nullptr;
        return false;
    }

    const uint8_t* p = static_cast<const uint8_t*>(file_data_);
    uint8_t version = *p; // 读取第一个字节作为版本号

    if (version != 1) {
        ::munmap(file_data_, file_len_); // 清理刚刚 mmap 的资源
        file_data_ = nullptr;
        file_len_ = 0;
        return false; // <--- 通知调用者 mmap 失败，请使用其他加载方式
    }

    ++p;

    const uint8_t* end = static_cast<const uint8_t*>(file_data_) + file_len_;

    // 2. 后续所有针对 V1 格式的解析逻辑完全保持不变
    auto readVar = [&](const uint8_t*& cur)->int64_t{
        uint64_t res=0; int shift=0; uint8_t byte;
        do{
            if(cur>=end){throw std::runtime_error("eof");}
            byte=*cur++; res |= uint64_t(byte & 0x7f)<<shift; shift+=7;
        }while(byte&0x80);
        return (res>>1) ^ (-(res&1));
    };

    try { // 将解析逻辑包裹在 try-catch 中，以防 readVar 抛出异常
        size_t mapSize = readVar(p);
        KeyType prev_key = 0;

        lazy_index_.reserve(mapSize);
        for(size_t i=0;i<mapSize;++i){
            KeyType key = prev_key + readVar(p); prev_key = key;

            size_t vecSize = readVar(p);
            const uint8_t* list_begin = p;
            if (vecSize) {
                readVar(p); // 跳过第一个值
                for(size_t j=1;j<vecSize;++j) readVar(p); // 跳过后续的差值
            }
            size_t bytes = p - list_begin;

            ListMeta meta;
            meta.offset = uint32_t(list_begin - static_cast<const uint8_t*>(file_data_));
            meta.bytes  = uint32_t(bytes);
            meta.size   = uint32_t(vecSize);
            lazy_index_[key] = meta;
        }
    } catch (const std::exception& e) {
        // 如果解析过程中发生错误（如文件损坏导致读取到末尾）
        std::cerr << "Error parsing V1 mmap file '" << filename << "': " << e.what() << std::endl;
        ::munmap(file_data_, file_len_);
        file_data_ = nullptr;
        file_len_ = 0;
        return false;
    }

    lazy_mode_ = true;
    lazy_cache_cap_ = max_cache_bytes;
    return true;
}

CustomHashMap::iterator CustomHashMap::begin() const { return map_.begin(); }
CustomHashMap::iterator CustomHashMap::end()   const { return map_.end(); }

}
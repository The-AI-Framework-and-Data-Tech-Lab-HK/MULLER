/*
 * Copyright (c) 2026 Bingyu Liu. All rights reserved.

 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "jieba_utils.h"
#include "murmurhash/MurmurHash3.h"
#include "logger.h"
#include <iostream>
#include <fstream>
#include <mutex>
#include <string>
#include <sstream>
#include <iomanip>
#include <cstddef>
#include <cstdint>

namespace hashmap {

namespace JiebaPath {
    static const char* const DICT = JIEBA_DICT_DIR "jieba.dict.utf8";
    static const char* const HMM = JIEBA_DICT_DIR "hmm_model.utf8";
    static const char* const USER = JIEBA_DICT_DIR "user.dict.utf8";
    static const char* const IDF = JIEBA_DICT_DIR "idf.utf8";
    static const char* const STOP = JIEBA_DICT_DIR "stop_words.utf8";
}

std::string JiebaUtils::sanitizeUserDict(const std::string& orig_path) {
    if (orig_path.empty()) return orig_path;

    std::ifstream fin(orig_path, std::ios::binary);
    if (!fin) {
        LOG_ERROR("cannot open user dict: " + orig_path);
        return orig_path;
    }

    std::string raw((std::istreambuf_iterator<char>(fin)),
                     std::istreambuf_iterator<char>());
    fin.close();

    bool need_fix = false;
    size_t i = 0;
    // 1. 跳过 BOM
    if (raw.size() >= 3 &&
        (unsigned char)raw[0] == 0xEF &&
        (unsigned char)raw[1] == 0xBB &&
        (unsigned char)raw[2] == 0xBF) {
        need_fix = true;
        i = 3;
    }

    std::string cleaned;
    cleaned.reserve(raw.size());
    for (; i < raw.size(); ++i) {
        char c = raw[i];
        if (c == '\r') {                 // 去掉 CR
            need_fix = true;
            continue;
        }
        cleaned.push_back(c);
    }
    if (!need_fix) {                    // 文件本来就是干净的
        return orig_path;
    }

    // 写入临时文件：orig_path + ".unix"
    std::string tmp_path = orig_path + ".unix";
    std::ofstream fout(tmp_path, std::ios::binary | std::ios::trunc);
    fout.write(cleaned.data(), cleaned.size());
    fout.close();

    LOG_INFO("user dict sanitized to " + tmp_path);
    return tmp_path;
}

cppjieba::Jieba& JiebaUtils::getJieba(const std::string& path) {
    static std::once_flag param_flag;
    static std::string    dict_path;

    // 把主线程传进来的 path 保存一次
    std::call_once(param_flag, [&]{
        if (!path.empty())
            dict_path = sanitizeUserDict(path);
        else
            dict_path = "";
    });
    using namespace JiebaPath;
    static thread_local cppjieba::Jieba jieba(DICT,   // 主词典
                                              HMM,    // HMM
                                              dict_path,   // 内置用户词典
                                              IDF,    // IDF
                                              STOP);

    return jieba;
}

std::unordered_set<std::string>& JiebaUtils::getStopWords() {
    static std::unordered_set<std::string> stopWords;
    static bool loaded = false;

    if (!loaded) {
        static const char* const STOP_WORD_PATH = JIEBA_DICT_DIR "stop_words.utf8";
        std::ifstream stopWordFile(STOP_WORD_PATH);
        if (stopWordFile.is_open()) {
            std::string line;
            while (std::getline(stopWordFile, line)) {
                if (!line.empty()) {
                    stopWords.insert(line);
                }
            }
            loaded = true;
        } else {
            std::cerr << "Failed to load stop words file: " << STOP_WORD_PATH << std::endl;
        }
    }
    std::cerr << "Successfully load stop words file: " << std::endl;
    return stopWords;
}

std::pair<int64_t, int> JiebaUtils::wordToInt64(const std::string& word, int num_of_shards) {
    const char* data = word.data();
    size_t len = word.size();

    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    const auto* p = reinterpret_cast<const std::uint8_t*>(data);
    for (std::size_t i = 0; i < len; ++i) {
        oss << std::setw(2) << static_cast<unsigned>(p[i]);
        if (i + 1 < len) oss << ' ';                              // 用空格分隔
    }
    int64_t hash_signed;
    MurmurHash3_x64_128(data, len, 0, &hash_signed);

    int shard_id = std::abs(hash_signed % num_of_shards);

    return {hash_signed, shard_id};
}

}
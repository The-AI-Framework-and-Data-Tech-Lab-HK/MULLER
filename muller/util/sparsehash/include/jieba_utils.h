/*
 * Copyright (c) 2026 Bingyu Liu. All rights reserved.

 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#pragma once
#include <string>
#include <unordered_set>
#include <utility>
#include "cppjieba/Jieba.hpp"

namespace hashmap {

class JiebaUtils {
public:
    // 新接口：第一次调用时为本线程加载 compulsory_dict_path
    static std::string sanitizeUserDict(const std::string& path);
    static cppjieba::Jieba& getJieba(const std::string& path = "");

    static std::unordered_set<std::string>& getStopWords();
    static std::pair<int64_t, int> wordToInt64(const std::string& word,
                                               int num_of_shards);
};

} // namespace hashmap
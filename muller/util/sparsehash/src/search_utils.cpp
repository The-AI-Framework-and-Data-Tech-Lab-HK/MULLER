/*
 * Copyright (c) 2026 Bingyu Liu. All rights reserved.

 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "search_utils.h"
#include "custom_hash_map.h"
#include "jieba_utils.h"
#include "thread_pool.h"
#include "logger.h"

#include <unordered_map>
#include <unordered_set>
#include <set>
#include <vector>
#include <string>
#include <mutex>
#include <future>
#include <thread>
#include <memory>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <regex>

namespace hashmap { namespace search_utils {


static bool file_exists(const std::string& path) {
#ifdef _WIN32
    return ::_access(path.c_str(), 0) == 0;
#else
    struct stat sb;
    return ::stat(path.c_str(), &sb) == 0;
#endif
}


struct ShardCache {
    std::unordered_map<int, std::shared_ptr<CustomHashMap> > maps;
    std::mutex                                               mtx;

    static ShardCache& instance() {
        static ShardCache cache;
        return cache;
    }
};


static ThreadPool& get_pool(int requested_threads) {
    static ThreadPool pool(
        requested_threads > 0
            ? requested_threads
            : (std::thread::hardware_concurrency()
                   ? std::thread::hardware_concurrency()
                   : 1));
    return pool;
}

static std::vector<std::string> split_or_queries(const std::string& query) {
    std::vector<std::string> result;
    std::regex or_regex(R"(\|\|)");
    std::sregex_token_iterator it(query.begin(), query.end(), or_regex, -1);
    std::sregex_token_iterator end;

    for (; it != end; ++it) {
        std::string sub = it->str();
        // 去除首尾空格
        size_t start = sub.find_first_not_of(" \t\n\r");
        size_t end = sub.find_last_not_of(" \t\n\r");
        if (start != std::string::npos && end != std::string::npos) {
            result.push_back(sub.substr(start, end - start + 1));
        }
    }
    return result;
}

static std::unordered_map<int64_t, std::unordered_set<int64_t>>
search_single_shard_for_complex(
    const std::string& index_folder,
    int shard_id,
    const std::vector<int64_t>& hash_values)
{
    const std::string shard_path = index_folder + "/" + std::to_string(shard_id);

    std::shared_ptr<CustomHashMap> shard_map = std::make_shared<CustomHashMap>();
    if (file_exists(shard_path)) {
        if (!shard_map->mmapReadOnly(shard_path))
            shard_map->loadFromFile(shard_path);
    } else {
        return {};
    }

    std::unordered_map<int64_t, std::unordered_set<int64_t>> result;

    for (int64_t hv : hash_values) {
        std::shared_ptr<const std::vector<int64_t>> values_ptr =
            shard_map->getValuesByKey(hv);

        if (values_ptr && !values_ptr->empty()) {
            const std::vector<int64_t>& values = *values_ptr;
            result[hv].insert(values.begin(), values.end());
        }
    }

    return result;
}

static std::unordered_set<int64_t> search_single_shard_fast(
        const std::string&           index_folder,
        int                          shard_id,
        const std::vector<int64_t>&  hash_values)
{
    const std::string shard_path = index_folder + "/" + std::to_string(shard_id);

    std::shared_ptr<CustomHashMap> shard_map = std::make_shared<CustomHashMap>();
    if (file_exists(shard_path)) {
        if (!shard_map->mmapReadOnly(shard_path))
            shard_map->loadFromFile(shard_path);
    } else {
        return {};
    }

    std::unordered_set<int64_t> result;
    bool first_hash = true;

    for (size_t i = 0; i < hash_values.size(); ++i) {
        int64_t hv = hash_values[i];
        std::shared_ptr<const std::vector<int64_t> > values_ptr =
            shard_map->getValuesByKey(hv);

        if (!values_ptr || values_ptr->empty()){
            return std::unordered_set<int64_t>();
        }

        const std::vector<int64_t>& values = *values_ptr;

        if (first_hash) {
            result.reserve(values.size() * 2);
            result.insert(values.begin(), values.end());
            first_hash = false;
            continue;
        }

        if (values.size() < result.size()) {
            std::unordered_set<int64_t> tmp;
            tmp.reserve(values.size() * 2);
            for (size_t j = 0; j < values.size(); ++j)
                if (result.count(values[j])) tmp.insert(values[j]);
            result.swap(tmp);
        } else {
            for (auto it = result.begin(); it != result.end(); ) {
                if (std::find(values.begin(), values.end(), *it) == values.end())
                    it = result.erase(it);
                else
                    ++it;
            }
        }
        if (result.empty()) return std::unordered_set<int64_t>();
    }
    return result;
}

std::set<int64_t> search_idx(
        const std::string& query,
        const std::string& index_folder,
        const std::string& search_type,
        int num_of_shards,
        bool cut_all,
        const std::unordered_set<std::string>& full_stop_words,
        bool case_sensitive,
        int  max_workers,
        const std::string& compulsory_dict_path)
{
    std::unordered_map<int, std::vector<int64_t> > shard_hashes;

    if (search_type == "exact_match") {
        // 精确匹配模式：直接对整个query进行哈希
        std::pair<int64_t, int> hp = JiebaUtils::wordToInt64(query, num_of_shards);
        shard_hashes[hp.second].push_back(hp.first);
    }
    else if (search_type == "complex_fuzzy_match") {
        // 复杂模糊匹配模式：支持OR查询
        std::vector<std::string> sub_queries = split_or_queries(query);
        if (sub_queries.empty()) {
            return std::set<int64_t>();
        }

        // 存储每个子查询的分词结果
        std::unordered_map<std::string, std::vector<int64_t>> query_tok_dict;

        cppjieba::Jieba& jieba = JiebaUtils::getJieba(compulsory_dict_path);

        // 处理每个子查询
        for (const std::string& sub_query : sub_queries) {
            std::string processed = sub_query;
            if (!case_sensitive) {
                std::transform(processed.begin(), processed.end(),
                             processed.begin(), ::tolower);
            }

            std::vector<std::string> words;
            if (cut_all)
                jieba.CutAll(processed, words);
            else
                jieba.Cut(processed, words);

            std::vector<int64_t> hash_list;
            for (const std::string& w : words) {
                if (full_stop_words.find(w) != full_stop_words.end()) continue;

                std::pair<int64_t, int> hp = JiebaUtils::wordToInt64(w, num_of_shards);
                shard_hashes[hp.second].push_back(hp.first);
                hash_list.push_back(hp.first);
            }

            if (!hash_list.empty()) {
                query_tok_dict[sub_query] = hash_list;
            }
        }

        if (query_tok_dict.empty()) {
            return std::set<int64_t>();
        }

        // 使用线程池并行搜索各个shard
        ThreadPool& pool = get_pool(max_workers);
        std::vector<std::future<std::unordered_map<int64_t, std::unordered_set<int64_t>>>> futures;

        for (const auto& kv : shard_hashes) {
            futures.emplace_back(
                pool.enqueue(search_single_shard_for_complex,
                           index_folder,
                           kv.first,
                           kv.second));
        }

        // 收集所有shard的结果
        std::unordered_map<int64_t, std::unordered_set<int64_t>> final_word_docs;
        for (auto& fut : futures) {
            auto shard_result = fut.get();
            for (const auto& kv : shard_result) {
                final_word_docs[kv.first] = kv.second;
            }
        }

        // 计算最终结果
        std::unordered_set<int64_t> final_ids;

        // 对每个子查询，计算内部的交集（AND），然后合并结果（OR）
        for (const auto& kv : query_tok_dict) {
            const std::vector<int64_t>& words = kv.second;
            if (words.empty()) continue;

            // 获取第一个词的文档集作为初始集合
            auto it = final_word_docs.find(words[0]);
            if (it == final_word_docs.end() || it->second.empty()) continue;

            std::unordered_set<int64_t> sub_result = it->second;

            // 与其他词的文档集求交集
            for (size_t i = 1; i < words.size(); ++i) {
                auto word_it = final_word_docs.find(words[i]);
                if (word_it == final_word_docs.end() || word_it->second.empty()) {
                    sub_result.clear();
                    break;
                }

                // 求交集
                std::unordered_set<int64_t> tmp;
                for (int64_t id : sub_result) {
                    if (word_it->second.count(id)) {
                        tmp.insert(id);
                    }
                }
                sub_result = std::move(tmp);

                if (sub_result.empty()) break;
            }

            // 将子查询结果并入最终结果（OR）
            final_ids.insert(sub_result.begin(), sub_result.end());
        }

        return std::set<int64_t>(final_ids.begin(), final_ids.end());
    }
    else { // fuzzy_match
        // 模糊匹配模式：分词处理
        std::string processed_query = query;
        if (!case_sensitive) {
            std::transform(processed_query.begin(), processed_query.end(),
                         processed_query.begin(), ::tolower);
        }
        cppjieba::Jieba& jieba = JiebaUtils::getJieba(compulsory_dict_path);

        std::vector<std::string> words;
        if (cut_all)
            jieba.CutAll(processed_query, words);
        else
            jieba.Cut(processed_query, words);

        for (const std::string& w : words) {
            if (full_stop_words.find(w) != full_stop_words.end()) continue;
            std::pair<int64_t, int> hp = JiebaUtils::wordToInt64(w, num_of_shards);
            shard_hashes[hp.second].push_back(hp.first);
        }
    }

    if (shard_hashes.empty()) {
        return std::set<int64_t>();
    }

    if (search_type != "complex_fuzzy_match") {
        ThreadPool& pool = get_pool(max_workers);

        std::vector<std::future<std::unordered_set<int64_t> > > futures;
        futures.reserve(shard_hashes.size());

        for (auto& kv : shard_hashes) {
            futures.emplace_back(
                pool.enqueue(search_single_shard_fast,
                             index_folder,
                             kv.first,
                             kv.second) );
        }

        std::unordered_set<int64_t> final_res;
        bool first = true;

        for (size_t i = 0; i < futures.size(); ++i) {
            std::unordered_set<int64_t> shard_res = futures[i].get();

            if (first) {
                final_res.swap(shard_res);
                first = false;
                continue;
            }

            if (shard_res.empty()) return std::set<int64_t>();

            if (shard_res.size() < final_res.size()) {
                std::unordered_set<int64_t> tmp;
                tmp.reserve(shard_res.size() * 2);
                for (auto id : shard_res)
                    if (final_res.count(id)) tmp.insert(id);
                final_res.swap(tmp);
            } else {
                for (auto it = final_res.begin(); it != final_res.end(); ) {
                    if (!shard_res.count(*it))
                        it = final_res.erase(it);
                    else
                        ++it;
                }
            }

            if (final_res.empty()) return std::set<int64_t>();
        }

        return std::set<int64_t>(final_res.begin(), final_res.end());
    }
    return std::set<int64_t>();
}

}}
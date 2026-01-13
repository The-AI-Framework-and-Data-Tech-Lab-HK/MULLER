/*
 * Copyright (c) 2026 Bingyu Liu. All rights reserved.

 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "value_set.h"
#include "custom_hash_map.h"
#include "index_processor.h"
#include "index_utils.h"
#include "search_utils.h"
#include "logger.h"

namespace py = pybind11;
using namespace hashmap;

PYBIND11_MODULE(custom_hash_map, m) {
    using namespace hashmap;

    py::class_<TValueSet>(m, "TValueSet")
        .def(py::init<>())
        .def("insert", [](TValueSet& self, ValType value) { self.insert(value); })
        .def("contains", [](const TValueSet& self, ValType value) { return self.contains(value); })
        .def("__iter__", [](const TValueSet& self) {
            return py::make_iterator(self.begin(), self.end());
        }, py::keep_alive<0, 1>());

    py::class_<CustomHashMap>(m, "CustomHashMap")
        .def(py::init<>())
        .def("add", &CustomHashMap::add)
        .def("find", &CustomHashMap::find)
        .def("get_values_by_key", [](const CustomHashMap& self, KeyType key) -> py::object {
            auto values = self.getValuesByKey(key);
            if (values) {
                return py::cast(*values);
            }
            return py::none();
        })
        .def("save_to_file", &CustomHashMap::saveToFile)
        .def("load_from_file", &CustomHashMap::loadFromFile)
        .def("__iter__", [](const CustomHashMap& self) {
            return py::make_key_iterator(self.begin(), self.end());
        }, py::keep_alive<0, 1>())
        .def("items", [](const CustomHashMap& self) {
            auto items = py::list();
            for (const auto& pair : self) {
                auto py_values = py::cast(*pair.second);
                items.append(py::make_tuple(pair.first, py_values));
            }
            return items;
        });

    py::class_<IndexProcessor>(m, "IndexProcessor")
        .def(py::init<>())
        .def_static("process_index_parallel", &IndexProcessor::process_index_parallel,
                   py::arg("root_path"),          // muller data path
                   py::arg("tensor_name"),        // tensor name
                   py::arg("index_folder"),       // shard 落盘的地址
                   py::arg("col_log_folder"),     // 记录落盘成功的空日志文件的地址
                   py::arg("starts"),              // 读取数据gid的起始
                   py::arg("ends"),                // 读取数据gid的结束
                   py::arg("num_threads"),                // 读取数据gid的结束
                   py::arg("num_of_shards"),      // 分片数量
                   py::arg("cut_all"),            // jieba参数
                   py::arg("case_sensitive"),     // jieba参数,大小写敏感
                   py::arg("full_stop_words"),    // 传入的停止词
                   py::arg("version"),    // 传入的版本号
                   py::arg("compulsory_dict_path"),
                   "Process documents and build inverted index shards")
        .def_static("process_index_single", &IndexProcessor::process_index_single,
                   py::arg("root_path"),          // muller data path
                   py::arg("tensor_name"),        // tensor name
                   py::arg("index_folder"),       // shard 落盘的地址
                   py::arg("col_log_folder"),     // 记录落盘成功的空日志文件的地址
                   py::arg("batch_count"),        // batch的数量记录
                   py::arg("start"),              // 读取数据gid的起始
                   py::arg("end"),                // 读取数据gid的结束
                   py::arg("num_of_shards"),      // 分片数量
                   py::arg("cut_all"),            // jieba参数
                   py::arg("case_sensitive"),     // jieba参数,大小写敏感
                   py::arg("full_stop_words"),    // 传入的停止词
                   py::arg("version"),    // 传入的版本号
                   py::arg("compulsory_dict_path"),
                   "Process documents and build inverted index shards");

    m.def("merge_index_files", &index_utils::merge_index_files,
          py::arg("tmp_index_path"),          // 需要读取的文件目录
          py::arg("optimized_index_path"),        // 需要写入的optimize之后的目录
          py::arg("current_index_folder"),        // 当前的索引索林目录
          py::arg("optimize_mode"),        // 优化模式
          py::arg("num_shards") = 0,      // 需要合并的shards的数量
          py::arg("num_threads") = 0,      // 线程数量（默认自动选择）
          "Merge index files with the same prefix into consolidated files");

    m.def("search_idx", &search_utils::search_idx,
          py::arg("query"),                // 查询字符串
          py::arg("index_folder"),         // 索引文件夹路径
          py::arg("search_type"),         // 搜索类型
          py::arg("num_of_shards"),        // 分片数量
          py::arg("cut_all"),               // jieba参数
          py::arg("full_stop_words"),        // 传入的停止词
          py::arg("compulsory_words_path"),        // 分片数量
          py::arg("case_sensitive"),        // 大小写是否敏感
          py::arg("max_workers") = 0,      // 最大线程数（默认自动选择）
          "Search for documents matching all terms in the query");

    m.def("init_logger", [](const std::string& logFilePath, int logLevel) {
        std::cout << "尝试初始化日志系统: " << logFilePath << std::endl;

        // 检查目录是否存在
        std::string dir = logFilePath.substr(0, logFilePath.find_last_of("/\\"));
        struct stat info;
        if (stat(dir.c_str(), &info) != 0 || !(info.st_mode & S_IFDIR)) {
            std::cout << "目录不存在: " << dir << std::endl;
            return false;
        }

        bool result = hashmap::Logger::getInstance().init(logFilePath,
                                                static_cast<hashmap::Logger::LogLevel>(logLevel));
        std::cout << "日志初始化结果: " << (result ? "成功" : "失败") << std::endl;
        return result;
    }, py::arg("log_file_path"), py::arg("log_level") = 1);

    py::enum_<hashmap::Logger::LogLevel>(m, "LogLevel")
        .value("DEBUG", hashmap::Logger::DEBUG)
        .value("INFO", hashmap::Logger::INFO)
        .value("WARNING", hashmap::Logger::WARNING)
        .value("ERROR", hashmap::Logger::ERROR)
        .export_values();
}

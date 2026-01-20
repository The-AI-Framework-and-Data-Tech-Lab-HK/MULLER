/*
 * Copyright (c) 2026 Bingyu Liu. All rights reserved.

 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#pragma once
#include "exceptions.h"
#include <fstream>
#include <sys/stat.h>
#include <string>
#include <vector>

namespace muller_reader::io {

class FileProvider {
public:
    explicit FileProvider(std::string root_) : root(std::move(root_)) {}

    // 相当于 python __getitem__
    std::vector<std::byte> operator[](const std::string& rel) const {
        auto full = checkIsFile(rel);
        return readAll(full);
    }

    std::string checkIsFile(const std::string& rel) const {
        std::string full = root + "/" + rel;
        struct stat st {};
        if (::stat(full.c_str(), &st) == 0 && S_ISDIR(st.st_mode))
            throw DirectoryAtPath("directory at " + full);
        return full;
    }

private:
    std::string root;

    static std::vector<std::byte> readAll(const std::string& p) {
        std::ifstream f(p, std::ios::binary | std::ios::ate);
        if (!f) throw std::runtime_error("open failed: " + p);
        std::streamsize sz = f.tellg();
        f.seekg(0);
        std::vector<std::byte> buf(static_cast<std::size_t>(sz));
        f.read(reinterpret_cast<char*>(buf.data()), sz);
        return buf;
    }
};

} // namespace muller_reader::io
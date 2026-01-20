/*
 * Copyright (c) 2026 Bingyu Liu. All rights reserved.

 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#pragma once
#include <stdexcept>
#include <string>

namespace muller_reader::io {

struct DirectoryAtPath : std::runtime_error {
    using std::runtime_error::runtime_error;
};
struct GetChunkError : std::runtime_error {
    explicit GetChunkError(const std::string& k)
        : std::runtime_error("GetChunkError for key: " + k) {}
};

} // namespace muller_reader::io
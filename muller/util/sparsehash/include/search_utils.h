/*
 * Copyright (c) 2026 Bingyu Liu. All rights reserved.

 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#pragma once

#include "common.h"
#include <string>
#include <vector>
#include <set>

namespace hashmap {
namespace search_utils {

std::set<int64_t> search_idx(
        const std::string& query,
        const std::string& index_folder,
        const std::string& search_type,
        int num_of_shards,
        bool cut_all,
        const std::unordered_set<std::string>& full_stop_words,
        bool case_sensitive,
        int  max_workers,
        const std::string& compulsory_dict_path = "");

}
}
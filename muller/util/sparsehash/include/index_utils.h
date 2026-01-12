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

namespace hashmap {
namespace index_utils {

bool merge_index_files(
    const std::string& tmp_index_path,
    const std::string& optimized_index_path,
    const std::string& current_index_folder,
    const std::string& optimize_mode,
    int num_shards,
    int num_threads);

}
}
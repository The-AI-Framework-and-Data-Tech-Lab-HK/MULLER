/*
 * Copyright (c) 2026 Bingyu Liu. All rights reserved.

 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <memory>
#include <iostream>
#include <algorithm>

#include <sparsehash/sparse_hash_map>

#ifndef MULLER_F_EXPORT_H
#define MULLER_F_EXPORT_H

#define MULLER_F_API __attribute__((visibility("default")))

#endif

namespace hashmap {

using KeyType = int64_t;
using ValType = int64_t;

constexpr int DEFAULT_NUM_THREADS = 0;

}
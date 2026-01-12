/*
 * Copyright (c) 2026 Bingyu Liu. All rights reserved.

 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#pragma once

#include "common.h"
#include <vector>
#include <algorithm>

namespace hashmap {

class TValueSet : public std::vector<ValType> {
public:
    TValueSet();

    void insert(ValType value);
    bool contains(ValType value) const;
    typename std::vector<ValType>::iterator find(ValType value);
    typename std::vector<ValType>::const_iterator find(ValType value) const;
};

}
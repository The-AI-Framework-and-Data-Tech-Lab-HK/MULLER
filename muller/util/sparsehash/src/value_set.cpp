/*
 * Copyright (c) 2026 Bingyu Liu. All rights reserved.

 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "value_set.h"

namespace hashmap {

TValueSet::TValueSet() {}

void TValueSet::insert(ValType value) {
    if (find(value) == end()) {
        push_back(value);
    }
}

bool TValueSet::contains(ValType value) const {
    return find(value) != end();
}

typename std::vector<ValType>::iterator TValueSet::find(ValType value) {
    return std::find(begin(), end(), value);
}

typename std::vector<ValType>::const_iterator TValueSet::find(ValType value) const {
    return std::find(begin(), end(), value);
}
}
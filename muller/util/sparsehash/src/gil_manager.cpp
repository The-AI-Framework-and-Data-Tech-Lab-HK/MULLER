/*
 * Copyright (c) 2026 Bingyu Liu. All rights reserved.

 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "gil_manager.h"
#include <pybind11/pybind11.h>
#include <pybind11/gil.h>

namespace hashmap {

// 实现 GIL 管理
struct GILManager::ScopedRelease::Impl {
    pybind11::gil_scoped_release* release;

    Impl() : release(new pybind11::gil_scoped_release()) {}
    ~Impl() { delete release; }
};

GILManager::ScopedRelease::ScopedRelease() : pImpl(new Impl()) {}

GILManager::ScopedRelease::~ScopedRelease() {
    delete pImpl;
}

} // namespace hashmap
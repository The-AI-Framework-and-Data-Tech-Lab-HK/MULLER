/*
 * Copyright (c) 2026 Bingyu Liu. All rights reserved.

 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#pragma once

namespace hashmap {

// GIL 管理类
class GILManager {
public:
    // 获取一个 GIL 释放对象（RAII 风格）
    class ScopedRelease {
    public:
        ScopedRelease();
        ~ScopedRelease();
    private:
        struct Impl;
        Impl* pImpl;
    };

    // 静态辅助方法
    static ScopedRelease release_gil() {
        return ScopedRelease();
    }
};

} // namespace hashmap
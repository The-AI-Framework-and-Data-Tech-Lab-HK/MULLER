# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

from muller.util.cache import DataCache


class FlaskCache(DataCache):
    def get_paginated_data(self, key: str, offset: int, limit: int):
        """Get value from data LRU cache by offset and limit."""
        with self.lock:
            if key not in self.cache:
                return []
            return self.cache[key][offset: offset + limit]

    def total_items(self, key: str):
        """Get the number of items stored in the cache for the given key."""
        with self.lock:
            try:
                return len(self.cache[key])
            except KeyError:
                return 0

    def is_cached(self, key: str):
        """Check if a specific key exists in the cache."""
        with self.lock:
            return key in self.cache

    def clear_cache(self, key: str):
        """clear certain key from cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]

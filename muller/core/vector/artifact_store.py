# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import json
import os
import posixpath
from pathlib import Path
from typing import Dict, List

from muller.core.storage import StorageProvider


class IndexArtifactStore:
    """Small storage-backed helper for vector index artifacts."""

    def __init__(self, storage: StorageProvider, root: str):
        self.storage = storage
        self.root = root.strip("/")

    def key(self, *parts: str) -> str:
        clean_parts = [str(part).strip("/") for part in parts if str(part).strip("/")]
        if self.root:
            return posixpath.join(self.root, *clean_parts)
        return posixpath.join(*clean_parts)

    def exists(self, *parts: str) -> bool:
        try:
            self.storage[self.key(*parts)]
            return True
        except KeyError:
            return False

    def read_bytes(self, *parts: str) -> bytes:
        return self.storage[self.key(*parts)]

    def write_bytes(self, data: bytes, *parts: str) -> None:
        self.storage[self.key(*parts)] = data

    def load_json(self, *parts: str) -> Dict:
        return json.loads(self.read_bytes(*parts).decode("utf-8"))

    def save_json(self, data: Dict, *parts: str) -> None:
        self.write_bytes(
            json.dumps(data, sort_keys=True).encode("utf-8"),
            *parts,
        )

    def list_prefix(self, *parts: str) -> List[str]:
        prefix = self.key(*parts).rstrip("/")
        prefix_with_slash = f"{prefix}/"
        return sorted(
            key for key in self.storage._all_keys()
            if key == prefix or key.startswith(prefix_with_slash)
        )

    def delete_prefix(self, *parts: str) -> None:
        keys = set(self.list_prefix(*parts))
        if not keys:
            return
        for key in keys:
            del self.storage[key]

    def publish_dir(self, local_dir: Path, *parts: str) -> List[Dict[str, object]]:
        manifest = []
        for root, _, files in os.walk(local_dir):
            for file_name in files:
                local_path = Path(root, file_name)
                relative_path = local_path.relative_to(local_dir).as_posix()
                artifact_key = self.key(*parts, relative_path)
                content = local_path.read_bytes()
                self.storage[artifact_key] = content
                manifest.append({
                    "path": relative_path,
                    "key": artifact_key,
                    "size": len(content),
                })
        manifest.sort(key=lambda item: item["path"])
        return manifest

    def materialize_dir(self, local_dir: Path, *parts: str) -> None:
        prefix = self.key(*parts).rstrip("/")
        prefix_with_slash = f"{prefix}/"
        for key in self.list_prefix(*parts):
            relative_path = key[len(prefix_with_slash):] if key.startswith(prefix_with_slash) else Path(key).name
            local_path = local_dir / relative_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(self.storage[key])

# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/core/dataset/view_entry.py
#
# Modifications Copyright (c) 2026 Bingyu Liu

import logging
from typing import Dict, Optional, Any


class ViewEntry:
    """Represents a view saved inside a dataset."""

    def __init__(
        self, info: Dict, dataset, source_dataset=None, external: bool = False
    ):
        self.info = info
        self._ds = dataset
        self._src_ds = source_dataset if external else dataset
        self._external = external

    def __getitem__(self, key: str):
        return self.info[key]

    def __str__(self):
        return (f"View(id='{self.id}', message='{self.message}', virtual={self.virtual}, commit_id={self.commit_id}, "
                f"query='{self.query}, tql_query='{self.tql_query}')")

    __repr__ = __str__

    @property
    def id(self) -> str:
        """Returns id of the view."""
        return self.info["id"].split("]")[-1]

    @property
    def query(self) -> Optional[str]:
        """Returns query of the view."""
        return self.info.get("query")

    @property
    def tql_query(self) -> Optional[str]:
        """Returns query of the view."""
        return self.info.get("tql_query")

    @property
    def message(self) -> str:
        """Returns the message with which the view was saved."""
        return self.info.get("message", "")

    @property
    def commit_id(self) -> str:
        """Returns the commit id of the view."""
        return self.info["source-dataset-version"]

    @property
    def virtual(self) -> bool:
        """Returns whether the view is virtual."""
        return self.info["virtual-datasource"]

    @property
    def source_dataset_path(self) -> str:
        """Returns the source dataset path."""
        return self.info["source-dataset"]

    @property
    def src_ds(self):
        """Returns the source dataset."""
        return self._src_ds

    def get(self, key: str, default: Optional[Any] = None):
        """Get a value from the view entry."""
        return self.info.get(key, default)

    def load(self, verbose=True):
        """Loads the view and returns the :class:`~muller.core.dataset.Dataset`.

        Args:
            verbose (bool): If ``True``, logs will be printed. Defaults to ``True``.

        Returns:
            Dataset: Loaded dataset view.
        """
        if self.commit_id != self._ds.commit_id:
            logging.info(f"Loading view from commit id {self.commit_id}.")

        ds = self._ds.sub_ds(
            ".queries/" + (self.info.get("path") or self.info["id"]),
            verbose=False,
            read_only=True,
        )

        if self.virtual:
            ds = ds.get_view_for_vds(inherit_creds=not self._external)

        if self.tql_query is not None:
            query_str = self.tql_query
            ds = ds.query(query_str)

        ds.view_entry = self

        return ds

    def delete(self):
        """Deletes the view."""
        self._ds.delete_view(id=self.info["id"])

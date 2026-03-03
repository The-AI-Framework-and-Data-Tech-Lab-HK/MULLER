# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

"""Statistics mixin for Dataset class."""

from muller.constants import VIEW_SUMMARY_SAFE_LIMIT
from muller.util.exceptions import SummaryLimit


class StatisticsMixin:
    """Mixin providing statistics operations for Dataset."""

    def summary(self, force: bool = False):
        """Print out a summarization of the schema and statistic information of the dataset."""
        from muller.core.dataset.statistics.summary import summary_dataset
        if (
                not self.index.is_trivial()
                and self.max_len > VIEW_SUMMARY_SAFE_LIMIT
                and not force
        ):
            raise SummaryLimit(self.max_len, VIEW_SUMMARY_SAFE_LIMIT)
        pretty_print = summary_dataset(self)
        print(pretty_print)

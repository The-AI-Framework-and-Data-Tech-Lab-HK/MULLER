# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

"""Unit tests for CommitDiff.pop / translate_index.

Coordinate conventions asserted here (and relied on by diff/merge display):

- ``data_added`` and ``data_updated`` are maintained in *live* coordinates,
  i.e. the current global indices of the samples after all pops so far.
- ``data_deleted`` records indices in the *commit-start* coordinate space
  (the tensor layout at the moment this commit's diff was created), so that
  the diff can report which parent-commit samples were removed.
- ``data_deleted_ids`` stays index-aligned in length with ``data_deleted``
  when sample ids are provided (serialization assumes equal lengths).
"""

import random

from muller.core.version_control.commit_diff import CommitDiff


class TestPopAddedSamples:
    """Popping samples that were appended within the same commit."""

    def test_pop_added_sample_fresh_dataset(self):
        diff = CommitDiff(first_index=0)
        diff.add_data(5)

        diff.pop(2, 102)

        assert diff.data_added == [0, 4]
        assert list(diff.data_deleted) == []
        assert diff.data_deleted_ids == []

    def test_pop_added_sample_on_top_of_base(self):
        diff = CommitDiff(first_index=5)
        diff.add_data(2)  # live layout: base 0-4, added 5-6

        diff.pop(6, 106)

        assert diff.data_added == [5, 6]
        assert list(diff.data_deleted) == []
        assert diff.data_deleted_ids == []


class TestPopBaseSamples:
    """Popping samples that already existed when the commit started."""

    def test_pop_single_base_sample(self):
        diff = CommitDiff(first_index=5)

        diff.pop(2, 102)

        assert list(diff.data_deleted) == [2]
        assert diff.data_deleted_ids == [102]
        assert diff.data_added == [4, 4]

    def test_pop_same_live_index_twice(self):
        # After deleting commit-start sample 2, live index 2 refers to
        # commit-start sample 3.
        diff = CommitDiff(first_index=5)

        diff.pop(2, 102)
        diff.pop(2, 103)

        assert list(diff.data_deleted) == [2, 3]
        assert diff.data_deleted_ids == [102, 103]
        assert diff.data_added == [3, 3]

    def test_pop_head_repeatedly(self):
        diff = CommitDiff(first_index=5)

        for sample_id in (100, 101, 102):
            diff.pop(0, sample_id)

        assert list(diff.data_deleted) == [0, 1, 2]
        assert diff.data_deleted_ids == [100, 101, 102]
        assert diff.data_added == [2, 2]

    def test_pop_descending_batch(self):
        # Dataset.pop sorts indices in descending order before popping.
        diff = CommitDiff(first_index=5)

        for idx, sample_id in ((4, 104), (2, 102), (0, 100)):
            diff.pop(idx, sample_id)

        assert list(diff.data_deleted) == [0, 2, 4]
        assert diff.data_deleted_ids == [104, 102, 100]
        assert diff.data_added == [2, 2]

    def test_translation_cascades_over_deleted_run(self):
        diff = CommitDiff(first_index=6)

        diff.pop(2, 102)  # commit-start 2
        diff.pop(2, 103)  # commit-start 3
        diff.pop(1, 101)  # commit-start 1
        # Live layout is now [0, 4, 5]; live 1 is commit-start 4.
        diff.pop(1, 104)

        assert list(diff.data_deleted) == [1, 2, 3, 4]
        assert diff.data_deleted_ids == [102, 103, 101, 104]

    def test_deleted_ids_length_matches_deleted_indices(self):
        diff = CommitDiff(first_index=5)

        diff.pop(2, 102)
        diff.pop(2, 103)

        assert len(diff.data_deleted) == len(diff.data_deleted_ids)


class TestPopMixedAddedAndBase:
    def test_pop_added_sample_after_base_deletion(self):
        # The added-range membership check must use live coordinates.
        diff = CommitDiff(first_index=2)
        diff.add_data(2)  # live layout: base 0-1, added 2-3

        diff.pop(0, 100)  # delete base sample 0 -> live [base1, add0, add1]
        assert diff.data_added == [1, 3]

        diff.pop(2, 900)  # live 2 is a sample added in this commit
        assert diff.data_added == [1, 2]
        assert list(diff.data_deleted) == [0]
        assert diff.data_deleted_ids == [100]

    def test_pop_base_then_all_added(self):
        diff = CommitDiff(first_index=1)
        diff.add_data(2)  # live: base 0, added 1-2

        diff.pop(0, 100)  # delete the only base sample
        diff.pop(1, 901)  # live 1 == second added sample
        diff.pop(0, 900)  # live 0 == first added sample

        assert diff.data_added == [0, 0]
        assert list(diff.data_deleted) == [0]
        assert diff.data_deleted_ids == [100]


class TestPopDataUpdatedMaintenance:
    def test_pop_shifts_updated_live_indices(self):
        diff = CommitDiff(first_index=5)
        diff.update_data(3)

        diff.pop(1, 101)

        assert diff.data_updated == {2}
        assert list(diff.data_deleted) == [1]

    def test_pop_keeps_updated_entry_after_prior_deletion(self):
        diff = CommitDiff(first_index=5)

        diff.pop(0, 100)     # delete commit-start 0
        diff.update_data(2)  # live 2 == commit-start 3
        diff.pop(1, 102)     # live 1 == commit-start 2

        # The updated sample shifts from live 2 to live 1.
        assert diff.data_updated == {1}
        assert list(diff.data_deleted) == [0, 2]

    def test_pop_updated_sample_removes_entry(self):
        diff = CommitDiff(first_index=5)
        diff.update_data(3)

        diff.pop(3, 103)

        assert diff.data_updated == set()
        assert list(diff.data_deleted) == [3]


class TestSerializationAfterPops:
    def test_roundtrip_preserves_pop_state(self):
        diff = CommitDiff(first_index=5)
        diff.add_data(2)
        diff.update_data(1)
        diff.pop(2, 102)
        diff.pop(2, 103)

        restored = CommitDiff.frombuffer(diff.tobytes())

        assert restored.data_added == diff.data_added
        assert restored.data_updated == diff.data_updated
        assert list(restored.data_deleted) == list(diff.data_deleted)
        assert restored.data_deleted_ids == diff.data_deleted_ids


class TestPopAgainstReferenceModel:
    def test_randomized_operations_match_reference_model(self):
        rng = random.Random(20260805)

        for round_no in range(100):
            base_count = rng.randint(0, 8)
            diff = CommitDiff(first_index=base_count)

            # Live tensor layout: ("base", commit_start_idx) entries always
            # precede ("added", n) entries because appends go to the tail.
            live = [("base", i) for i in range(base_count)]
            added_serial = 0
            expected_deleted = []
            expected_deleted_ids = []
            updated_base = set()  # commit-start indices of live updated samples

            for _ in range(rng.randint(1, 20)):
                op = rng.choice(["append", "update", "pop"])
                if op == "append":
                    count = rng.randint(1, 3)
                    diff.add_data(count)
                    for _ in range(count):
                        live.append(("added", added_serial))
                        added_serial += 1
                elif op == "update" and live:
                    idx = rng.randrange(len(live))
                    diff.update_data(idx)
                    kind, orig = live[idx]
                    if kind == "base":
                        updated_base.add(orig)
                elif op == "pop" and live:
                    idx = rng.randrange(len(live))
                    kind, orig = live.pop(idx)
                    # The id tensor supplies an id for every sample, but only
                    # deletions of pre-existing samples should record one.
                    diff.pop(idx, (1000 if kind == "base" else 5000) + orig)
                    if kind == "base":
                        expected_deleted.append(orig)
                        expected_deleted_ids.append(1000 + orig)
                        updated_base.discard(orig)

            remaining_base = sum(1 for kind, _ in live if kind == "base")
            expected_updated = {
                live_idx
                for live_idx, (kind, orig) in enumerate(live)
                if kind == "base" and orig in updated_base
            }

            context = f"round {round_no}"
            assert diff.data_added == [remaining_base, len(live)], context
            assert list(diff.data_deleted) == sorted(expected_deleted), context
            assert diff.data_deleted_ids == expected_deleted_ids, context
            assert diff.data_updated == expected_updated, context

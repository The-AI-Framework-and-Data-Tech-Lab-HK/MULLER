# SPDX-License-Identifier: MPL-2.0
"""Regression tests for NumPy scalar integer support in IndexEntry / Index.

Background:
    ``filter_vectorized`` returns sub-views whose positional indices are
    stored as ``np.int64`` tuples. Several internal predicates in
    ``IndexEntry`` (``subscriptable``, ``indices``, ``validate``) gated on
    ``isinstance(value, int)``, which is **False** for NumPy scalar
    integers. As a result, picking a single sample out of a filter view
    (``result_ds.bbox[0]``) silently returned an empty list ``[]`` from
    ``.numpy(aslist=True)`` instead of the expected sample data.

    The fix normalizes ``np.integer`` -> Python ``int`` at the
    IndexEntry / Index / Tensor / Dataset entry points. These tests pin
    that behaviour down so it cannot silently regress.
"""
from __future__ import annotations

import numpy as np
import pytest

import muller
from muller.core.index.index import IndexEntry, Index


# ---------------------------------------------------------------------------
# IndexEntry-level normalization
# ---------------------------------------------------------------------------


def test_index_entry_normalizes_numpy_scalar_int_to_python_int():
    e = IndexEntry(np.int64(7))
    assert type(e.value) is int
    assert e.value == 7


def test_index_entry_normalizes_numpy_int_in_tuple():
    e = IndexEntry((np.int64(1), np.int32(2), 3))
    assert all(type(v) is int for v in e.value)
    assert e.value == (1, 2, 3)


def test_index_entry_normalizes_numpy_int_in_list():
    e = IndexEntry([np.int64(4), np.int32(5)])
    assert all(type(v) is int for v in e.value)
    assert e.value == [4, 5]


def test_index_entry_subscriptable_after_numpy_scalar():
    """np.int64 used to slip through ``isinstance(value, int)``, leaving the
    entry erroneously marked subscriptable=True."""
    assert IndexEntry(np.int64(0)).subscriptable() is False
    assert IndexEntry(0).subscriptable() is False
    # Tuple / slice are still subscriptable (multiple samples).
    assert IndexEntry((np.int64(0), np.int64(1))).subscriptable() is True
    assert IndexEntry(slice(0, 5)).subscriptable() is True


def test_index_entry_indices_yields_for_numpy_scalar():
    """Pre-fix: ``indices()`` yielded **nothing** for np.int64 values."""
    assert list(IndexEntry(np.int64(7)).indices(100)) == [7]
    assert list(IndexEntry(0).indices(100)) == [0]


def test_index_entry_indices_handles_negative_numpy_scalar():
    """parse_int folds negatives via length+i; np.int64 must work too."""
    assert list(IndexEntry(np.int64(-1)).indices(10)) == [9]


def test_index_entry_validate_accepts_numpy_scalar():
    """``validate`` used ``isinstance(value, int)`` to decide whether to wrap
    in a tuple before bounds-checking; np.int64 used to skip the wrapping."""
    IndexEntry(np.int64(5)).validate(10)  # in-bounds -> no raise
    with pytest.raises(IndexError):
        IndexEntry(np.int64(20)).validate(10)


def test_index_entry_getitem_accepts_numpy_scalar():
    e = IndexEntry(slice(10, 20))
    sub = e[np.int64(3)]  # used to TypeError "unrecognized type"
    assert type(sub.value) is int
    assert sub.value == 13


def test_index_getitem_accepts_numpy_scalar():
    idx = Index([IndexEntry(slice(0, 100))])
    sub = idx[np.int64(5)]
    assert sub.values[0].value == 5


# ---------------------------------------------------------------------------
# End-to-end: filter view + single sample read returns real data, not []
# ---------------------------------------------------------------------------


def _build_coco_like_dataset(tmp_path):
    """Tiny COCO-shaped dataset with multi-annotation arrays per sample.

    Mirrors the real-world dataset shape that triggered the original bug:
    a per-sample (N, 4) bbox tensor and a per-sample (N,) class_label
    tensor, both with N varying across samples.
    """
    ds = muller.dataset(path=str(tmp_path / "coco_like"), overwrite=True)
    ds.create_tensor("image_id", htype="generic", dtype="int64")
    ds.create_tensor("bbox", htype="bbox", dtype="float32")
    ds.create_tensor("category_id", htype="class_label")
    n = 30
    for i in range(n):
        n_ann = (i % 5) + 2  # 2..6 annotations per image
        with ds:
            ds.image_id.append([100 + i])
            ds.bbox.append(np.arange(n_ann * 4, dtype="float32").reshape(n_ann, 4) + i)
            ds.category_id.append(np.arange(n_ann, dtype="uint32") + 1)
    ds.commit("init")
    return ds


def test_filter_view_single_sample_read_returns_data(tmp_path):
    """The exact bug reported in the SIGMOD demo:

    on a sub-view returned by ``filter_vectorized``, accessing a single
    non-image tensor sample (``result_ds.bbox[k]``) used to silently
    return ``[]``. With the fix it should return the real per-sample array
    and match what the bulk-fetch path returns.
    """
    ds = _build_coco_like_dataset(tmp_path)
    fv = ds.filter_vectorized([("image_id", ">=", 110)])
    assert len(fv) > 0

    bulk_bb = list(fv.bbox.numpy(aslist=True))
    bulk_cid = list(fv.category_id.numpy(aslist=True))

    # Single-element read on the filter view itself — was [] pre-fix.
    for k in range(min(3, len(fv))):
        single_bb = np.asarray(fv.bbox[k].numpy(aslist=True))
        single_cid = np.asarray(fv.category_id[k].numpy(aslist=True))
        assert single_bb.shape == np.asarray(bulk_bb[k]).shape
        assert np.array_equal(single_bb, bulk_bb[k])
        assert np.array_equal(single_cid, bulk_cid[k])

    # And the `slice-then-index` pattern the Streamlit UI uses.
    chunk = fv[0:3]
    for k in range(len(chunk)):
        single_bb = np.asarray(chunk.bbox[k].numpy(aslist=True))
        assert np.array_equal(single_bb, bulk_bb[k])


def test_filter_view_dataset_int64_row_access(tmp_path):
    """``result_ds[np.int64(0)]`` on the Dataset itself used to raise
    InvalidKeyTypeError. With the fix it transparently behaves like
    ``result_ds[0]``.
    """
    ds = _build_coco_like_dataset(tmp_path)
    fv = ds.filter_vectorized([("image_id", ">=", 110)])

    via_int = np.asarray(fv[0].bbox.numpy(aslist=True))
    via_npint = np.asarray(fv[np.int64(0)].bbox.numpy(aslist=True))
    assert np.array_equal(via_int, via_npint)
    assert via_int.shape[0] > 0


def test_tensor_int64_key_on_root_dataset(tmp_path):
    """Even on the root dataset, ``ds.bbox[np.int64(k)]`` should work — it
    was raising ``InvalidKeyTypeError`` before."""
    ds = _build_coco_like_dataset(tmp_path)
    a = np.asarray(ds.bbox[3].numpy(aslist=True))
    b = np.asarray(ds.bbox[np.int64(3)].numpy(aslist=True))
    assert np.array_equal(a, b)

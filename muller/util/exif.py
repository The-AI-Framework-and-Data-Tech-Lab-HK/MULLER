# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/util/exif.py
#
# Modifications Copyright (c) 2026 Bingyu Liu

from fractions import Fraction

from PIL import Image  # type: ignore
import PIL.ExifTags  # type: ignore

_LOOKUPS = {
    "MeteringMode": (
        "Undefined",
        "Average",
        "Center-weighted average",
        "Spot",
        "Multi-spot",
        "Multi-segment",
        "Partial",
    ),
    "ExposureProgram": (
        "Undefined",
        "Manual",
        "Program AE",
        "Aperture-priority AE",
        "Shutter speed priority AE",
        "Creative (Slow speed)",
        "Action (High speed)",
        "Portrait ",
        "Landscape",
        "Bulb",
    ),
    "ResolutionUnit": ("", "Undefined", "Inches", "Centimetres"),
    "Orientation": (
        "",
        "Horizontal",
        "Mirror horizontal",
        "Rotate 180",
        "Mirror vertical",
        "Mirror horizontal and rotate 270 CW",
        "Rotate 90 CW",
        "Mirror horizontal and rotate 90 CW",
        "Rotate 270 CW",
    ),
}

def getexif(image: Image):
    raw_exif = image._getexif()
    if not raw_exif:
        return {}
    exif = {}
    for k, v in raw_exif.items():
        tag = k
        k = PIL.ExifTags.TAGS.get(k, k)
        v = _process_exif_value(k, v)
        if v is None:
            continue
        exif[k] = v
    return exif

def _process_exif_value(k: str, v):
    if hasattr(v, "numerator"):  # Rational
        v = v.numerator / v.denominator if v.denominator else 0
        if k == "ExposureTime":
            v = float(Fraction(v).limit_denominator(8000))
        elif k in ("XResolution", "YResolution"):
            try:
                v = int(v)
            except ValueError:  # nan
                pass
        return v
    elif k in _LOOKUPS:
        return _LOOKUPS[k][v]
    elif isinstance(v, bytes):
        return str(v) if len(v) < 16 else "<bytes>"
    elif isinstance(v, (list, tuple)):
        return type(v)(_process_exif_value("_", x) for x in v)
    elif isinstance(v, dict):
        return {k: _process_exif_value("_", x) for k, x in v.items()}
    return str(v)
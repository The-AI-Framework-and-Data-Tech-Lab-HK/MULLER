# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/htype.py
#
# Modifications Copyright (c) 2026 Xueling Lin

from typing import Callable, Dict

import numpy as np

from .compression import IMAGE_COMPRESSIONS, BYTE_COMPRESSIONS, COMPRESSION_ALIASES
from .util.exceptions import IncompatibleHtypeError


class constraints:
    """Constraints for converting a tensor to a htype"""

    @staticmethod
    def ndim_error(temp_htype, ndim):
        """number of dimension error. """
        return f"Incompatible number of dimensions for htype {temp_htype}: {ndim}"

    @staticmethod
    def shape_error(temp_htype, shape):
        """shape error. """
        return f"Incompatible shape of tensor for htype {temp_htype}: {shape}"

    @staticmethod
    def dtype_error(temp_htype, dtype):
        """dtype error. """
        return f"Incompatible dtype of tensor for htype {temp_htype}: {dtype}"

    @staticmethod
    def instance_label(shape, dtype):
        """Return True for any shape and dtype (default behavior)."""
        return True

    INSTANCE_LABEL = instance_label

    @staticmethod
    def IMAGE(shape, dtype):
        """the image type"""
        if len(shape) not in (3, 4):
            raise IncompatibleHtypeError(constraints.ndim_error("image", len(shape)))
        if len(shape) == 4 and shape[-1] not in (1, 3, 4):
            raise IncompatibleHtypeError(constraints.shape_error("image", shape))

    @staticmethod
    def CLASS_LABEL(shape, dtype):
        """The class label type"""
        if len(shape) != 2:
            raise IncompatibleHtypeError(
                constraints.ndim_error("class_label", len(shape))
            )

    @staticmethod
    def BBOX(shape, dtype):
        """The bounding box type"""
        if len(shape) not in (2, 3):
            raise IncompatibleHtypeError(constraints.ndim_error("bbox", len(shape)))
        if shape[-1] != 4:
            raise IncompatibleHtypeError(constraints.shape_error("bbox", shape))

    @staticmethod
    def BBOX_3D(shape, dtype):
        """The bounding box 3D type"""
        if len(shape) not in (2, 3):
            raise IncompatibleHtypeError(constraints.ndim_error("bbox.3d", len(shape)))
        if shape[-1] != 8:
            raise IncompatibleHtypeError(constraints.shape_error("bbox.3d", shape))

    @staticmethod
    def EMBEDDING(shape, dtype):
        """The embedding type"""
        if dtype != np.float32:
            raise IncompatibleHtypeError(constraints.dtype_error("embedding", dtype))

    @staticmethod
    def BINARY_MASK(shape, dtype):
        """The binary mask type"""
        if len(shape) not in (3, 4):
            raise IncompatibleHtypeError(
                constraints.ndim_error("binary_mask", len(shape))
            )

    SEGMENT_MASK = BINARY_MASK

    @staticmethod
    def KEYPOINTS_COCO(shape, dtype):
        """The keypoint coco type"""
        if len(shape) != 3:
            raise IncompatibleHtypeError(
                constraints.ndim_error("keypoints_coco", len(shape))
            )
        if shape[1] % 3 != 0:
            raise IncompatibleHtypeError(
                constraints.shape_error("keypoints_coco", shape)
            )

    @staticmethod
    def POINT(shape, dtype):
        """The point type"""
        if len(shape) != 3:
            raise IncompatibleHtypeError(constraints.ndim_error("point", len(shape)))
        if shape[-1] not in (2, 3):
            raise IncompatibleHtypeError(constraints.shape_error("point", shape))


class htype:
    DEFAULT = "generic"
    IMAGE = "image"
    IMAGE_RGB = "image.rgb"
    IMAGE_GRAY = "image.gray"
    CLASS_LABEL = "class_label"
    BBOX = "bbox"
    BBOX_3D = "bbox.3d"
    VIDEO = "video"
    BINARY_MASK = "binary_mask"
    INSTANCE_LABEL = "instance_label"
    SEGMENT_MASK = "segment_mask"
    KEYPOINTS_COCO = "keypoints_coco"
    POINT = "point"
    AUDIO = "audio"
    TEXT = "text"
    JSON = "json"
    LIST = "list"
    DICOM = "dicom"
    NIFTI = "nifti"
    POINT_CLOUD = "point_cloud"
    INTRINSICS = "intrinsics"
    POLYGON = "polygon"
    MESH = "mesh"
    EMBEDDING = "embedding"
    VECTOR = "vector"


UNSPECIFIED = 'unspecified'

HTYPE_CONFIGURATIONS: Dict[str, Dict] = {
    htype.DEFAULT: {"dtype": None},
    htype.IMAGE: {
        "dtype": "uint8",
        "intrinsics": None,
        "_info": ["intrinsics"],
        "info": {},
    },
    htype.IMAGE_RGB: {
        "dtype": "uint8",
    },
    htype.IMAGE_GRAY: {
        "dtype": "uint8",
    },
    htype.CLASS_LABEL: {
        "dtype": "uint32",
        "class_names": [],
        "_info": ["class_names"],  # class_names should be stored in info, not meta
        "disable_temp_transform": False,
        "info": {},
    },
    htype.BBOX: {"dtype": "float32", "coords": {}, "_info": ["coords"], "info": {}},
    htype.BBOX_3D: {"dtype": "float32", "coords": {}, "_info": ["coords"], "info": {}},
    htype.AUDIO: {"dtype": "float64"},
    htype.EMBEDDING: {"dtype": "float32"},
    htype.VIDEO: {"dtype": "uint8"},
    htype.BINARY_MASK: {
        "dtype": "bool"
    },  # Sherry: pack numpy arrays to store bools as 1 bit instead of 1 byte
    htype.INSTANCE_LABEL: {"dtype": "uint32"},
    htype.SEGMENT_MASK: {
        "dtype": "uint32",
        "class_names": [],
        "_info": ["class_names"],
        "info": {},
    },
    htype.KEYPOINTS_COCO: {
        "dtype": "int32",
        "keypoints": [],
        "connections": [],
        "_info": [
            "keypoints",
            "connections",
        ],  # keypoints and connections should be stored in info, not meta
        "info": {},
    },
    htype.POINT: {"dtype": "int32"},
    htype.JSON: {
        "dtype": "Any",
    },
    htype.LIST: {"dtype": "List"},
    htype.TEXT: {"dtype": "str"},
    htype.DICOM: {"sample_compression": "dcm"},
    htype.NIFTI: {},
    htype.POINT_CLOUD: {"dtype": "float32"},
    htype.INTRINSICS: {"dtype": "float32"},
    htype.POLYGON: {"dtype": "float32"},
    htype.MESH: {"sample_compression": "ply"},
    htype.VECTOR: {"dtype": "float32",
                   "_info": ["dimension"],
                   "info": {}},
}

_image_compressions = (
        IMAGE_COMPRESSIONS[:] + BYTE_COMPRESSIONS + list(COMPRESSION_ALIASES)
)
_image_compressions.remove("dcm")

HTYPE_SUPPORTED_COMPRESSIONS = {
    htype.IMAGE: _image_compressions,
    htype.IMAGE_RGB: _image_compressions,
    htype.IMAGE_GRAY: _image_compressions,
}

HTYPE_VERIFICATIONS: Dict[str, Dict] = {
    htype.BBOX: {"coords": {"type": dict, "keys": ["type", "mode"]}},
    htype.BBOX_3D: {"coords": {"type": dict, "keys": ["mode"]}},
}

COMMON_CONFIGS = {
    "sample_compression": None,
    "chunk_compression": None,
    "dtype": None,
    "typestr": None,
    "max_chunk_size": None,
    "tiling_threshold": None,
    "hidden": False,
    "verify": False,
}

for config in HTYPE_CONFIGURATIONS.values():
    for key, v in COMMON_CONFIGS.items():
        if key not in config:
            config[key] = v


def verify_htype_key_value(_htype, _key, value):
    """Verify the h-type key and value."""
    htype_verifications = HTYPE_VERIFICATIONS.get(_htype, {})
    if _key in htype_verifications:
        expected_type = htype_verifications[_key].get("type")
        if expected_type and not isinstance(value, expected_type):
            raise TypeError(f"{_key} must be of type {expected_type}, not {type(value)}")
        if expected_type == dict:
            expected_keys = set(htype_verifications[_key].get("keys"))
            present_keys = set(value)
            if expected_keys and not present_keys.issubset(expected_keys):
                raise KeyError(f"{_key} must have keys belong to {expected_keys}")


HTYPE_CONVERSION_LHS = {htype.DEFAULT, htype.IMAGE}

HTYPE_CONSTRAINTS: Dict[str, Callable] = {
    htype.IMAGE: constraints.IMAGE,
    htype.CLASS_LABEL: constraints.CLASS_LABEL,
    htype.BBOX: constraints.BBOX,
    htype.BBOX_3D: constraints.BBOX_3D,
    htype.EMBEDDING: constraints.EMBEDDING,
    htype.BINARY_MASK: constraints.BINARY_MASK,
    htype.SEGMENT_MASK: constraints.SEGMENT_MASK,
    htype.INSTANCE_LABEL: constraints.INSTANCE_LABEL,
    htype.KEYPOINTS_COCO: constraints.KEYPOINTS_COCO,
    htype.POINT: constraints.POINT,
}

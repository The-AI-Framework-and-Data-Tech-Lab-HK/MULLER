# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/core/sample.py
#
# Modifications Copyright (c) 2026 Xueling Lin

import warnings
from io import BytesIO
from typing import Optional, Tuple, Union, Dict

import numpy as np
import requests
from PIL import Image  # type: ignore

from muller.core.compression import (
    compress_array,
    decompress_array,
    verify_compressed_file,
    read_meta_from_compressed_file,
    get_compression,
    get_compression_type
)
from muller.util.exceptions import SampleReadError, UnableToReadFromUrlError
from muller.util.exif import getexif
from muller.util.path import get_path_type, is_remote_path
from muller.core.storage.provider import StorageProvider
from muller.core.storage.provider import storage_factory
from muller.core.storage.roma import RomaProvider


class Sample:
    path: Optional[str]

    def __init__(
        self,
        path: Optional[str] = None,
        array: Optional[np.ndarray] = None,
        buffer: Optional[Union[bytes, memoryview]] = None,
        compression: Optional[str] = None,
        verify: bool = False,
        shape: Optional[Tuple[int]] = None,
        dtype: Optional[str] = None,
        creds: Optional[Dict] = None,
        storage: Optional[StorageProvider] = None,
    ):
        """Represents a *single sample* in a tensor column. Provides all important meta information in one place.

        Note:
            If ``self.is_lazy`` is ``True``, this :class:`Sample` doesn't actually have any data loaded.
            To read this data, simply try to read it into a numpy array (`sample.array`)

        Args:
            path (str): Path to a sample stored on the local file system that represents a single sample.
                        If ``path`` is provided, ``array`` should not be.
                        Implicitly makes ``self.is_lazy == True``.
            array (np.ndarray): Array that represents a single sample. If ``array`` is provided, ``path`` should not be.
                                Implicitly makes ``self.is_lazy == False``.
            buffer: (bytes): Byte buffer that represents a single sample.
                             If compressed, ``compression`` argument should be provided.
            compression (str): Specify in case of byte buffer.
            verify (bool): If a path is provided, verifies the sample if ``True``.
            shape (Tuple[int]): Shape of the sample.
            dtype (optional, str): Data type of the sample.
            creds (optional, Dict): Credentials for s3, gcp and http urls.
            storage (optional, StorageProvider): Storage provider.
        """
        self._compressed_bytes = {}
        self._uncompressed_bytes = None

        self._array = None
        self._pil = None
        self._typestr = None
        self._shape = shape or None
        self._dtype = dtype or None

        # Read from path
        self.path = None
        self.storage = storage
        self._buffer = None
        self._creds = creds or {}
        self._verify = verify

        if path is not None:
            self.path = path
            self._compression = compression
            if self._verify:
                if self._compression is None:
                    self._compression = get_compression(path=self.path)
                compressed_bytes = self._read_from_path()
                if self._compression is None:
                    self._compression = get_compression(header=compressed_bytes[:32])
                self._shape, self._typestr = verify_compressed_file(compressed_bytes, self._compression)  # type: ignore

        if array is not None:
            self._array = array
            self._shape = array.shape  # type: ignore
            self._typestr = array.__array_interface__["typestr"]
            self._dtype = np.dtype(self._typestr).name
            self._compression = None

        if buffer is not None:
            self._compression = compression
            self._buffer = buffer
            if compression is None:
                self._uncompressed_bytes = buffer
            else:
                self._compressed_bytes[compression] = buffer
                if self._verify:
                    self._shape, self._typestr = verify_compressed_file(buffer, self._compression)  # type: ignore

        self.htype = None

    def __str__(self):
        if self.is_lazy:
            return f"Sample(is_lazy=True, path={self.path})"

        return f"Sample(is_lazy=False, shape={self.shape}, compression='{self.compression}', " \
               f"dtype='{self.dtype}' path={self.path})"

    def __repr__(self):
        return str(self)

    def __array__(self, dtype=None):
        arr = self.array
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    def __eq__(self, other):
        if self.path is not None and other.path is not None:
            return self.path == other.path
        return self.buffer == other.buffer

    @property
    def is_lazy(self) -> bool:
        return self._array is None

    @property
    def array(self) -> np.ndarray:  # type: ignore
        """Return numpy array corresponding to the sample. Decompresses the sample if necessary.
        This is to facilitate the lazy decompression.

        Example:

            >>> sample = muller.read("./images/dog.jpg")
            >>> arr = sample.array
            >>> arr.shape
            (323, 480, 3)
        """
        arr = self._array
        if arr is not None:
            return arr
        self._decompress()
        return self._array  # type: ignore

    @property
    def buffer(self):
        if self._buffer is None and self.path is not None:
            self._read_from_path()
        if self._buffer is not None:
            return self._buffer
        return self.compressed_bytes(self.compression)

    @property
    def dtype(self):
        if self._dtype is None:
            self._read_meta()
            self._dtype = np.dtype(self._typestr).name
        return self._dtype

    @property
    def shape(self):
        self._read_meta()
        return self._shape

    @property
    def compression(self):
        if self._compression is None and self.path:
            self._read_meta()
        return self._compression

    @property
    def is_empty(self) -> bool:
        return 0 in self.shape

    @property
    def is_text_like(self):
        return self.htype in {"text", "list", "json", "tag"}

    @property
    def pil(self) -> Image.Image:  # type: ignore
        """Return PIL image corresponding to the sample. Decompresses the sample if necessary.

        Example:

            >>> sample = muller.read("./images/dog.jpg")
            >>> pil = sample.pil
            >>> pil.size
            (480, 323)
        """
        pil = self._pil
        if pil is not None:
            return pil
        self._decompress(to_pil=True)
        return self._pil

    @property
    def meta(self) -> dict:
        meta: Dict[str, Union[Dict, str]] = {}
        compression = self.compression
        compression_type = get_compression_type(compression)
        if compression == "dcm":
            meta.update(self._get_dicom_meta())
        elif compression_type == "image":
            meta["exif"] = self._getexif()
        meta["shape"] = self.shape
        meta["format"] = self.compression
        if self.path:
            meta["filename"] = str(self.path)
        return meta

    def copy(self):
        sample = Sample()
        sample._array = self._array
        sample._pil = self._pil
        sample._typestr = self._typestr
        sample._shape = self._shape
        sample._dtype = self._dtype
        sample.path = self.path
        sample.storage = self.storage
        sample._buffer = self._buffer
        sample._creds = self._creds
        sample._verify = self._verify
        sample._compression = self._compression
        return sample

    def compressed_bytes(self, compression: Optional[str]) -> bytes:
        """Returns this sample as compressed bytes.

        Note:
            If this sample is pointing to a path and the requested ``compression`` is the same as it's stored in,
            the data is returned without re-compressing.

        Args:
            compression (Optional[str]): ``self.array`` will be compressed into this format.
            If ``compression`` is ``None``, return :meth:`uncompressed_bytes`.

        Returns:
            bytes: Bytes for the compressed sample. Contains all metadata required to decompress within these bytes.

        Raises:
            ValueError: On recompression of unsupported formats.
        """

        if compression is None:
            return self.uncompressed_bytes()  # type: ignore

        compressed_bytes = self._compressed_bytes.get(compression)
        if compressed_bytes is None:
            if self.path is not None:
                if self._compression is None:
                    self._compression = get_compression(path=self.path)
                compressed_bytes = self._read_from_path()
                if self._compression is None:
                    self._compression = get_compression(header=compressed_bytes[:32])
                if self._compression == compression:
                    if self._shape is None:
                        _, self._shape, self._typestr = read_meta_from_compressed_file(
                            compressed_bytes, compression=self._compression
                        )
                else:
                    compressed_bytes = self._recompress(compressed_bytes, compression)
            elif self._buffer is not None:
                if self._compression is None:
                    self._compression = get_compression(header=self._buffer[:32])
                if self._compression == compression:
                    compressed_bytes = self._buffer
                else:
                    compressed_bytes = self._recompress(self._buffer, compression)
            else:
                compressed_bytes = compress_array(self.array, compression)
            self._compressed_bytes[compression] = compressed_bytes
        return compressed_bytes

    def uncompressed_bytes(self) -> Optional[bytes]:
        """Returns uncompressed bytes."""
        self._decompress()
        return self._uncompressed_bytes

    def _load_dicom(self):
        if self._array is not None:
            return
        try:
            from pydicom import dcmread
        except ImportError:
            raise ModuleNotFoundError(
                "Pydicom not found. Install using `pip install pydicom`"
            )
        if self.path and get_path_type(self.path) == "local":
            dcm = dcmread(self.path)
        else:
            dcm = dcmread(BytesIO(self.buffer))
        self._array = dcm.pixel_array
        self._shape = self._array.shape
        self._typestr = self._array.__array_interface__["typestr"]

    def _read_meta(self, f=None):
        """
        Get the self.dtype/self.shape/self.compression information
        """
        if self._shape is not None:
            return
        store = False
        if self._compression is None and self.path:
            self._compression = get_compression(path=self.path)
        if f is None:
            if self.path:
                if is_remote_path(self.path):
                    f = self._read_from_path()
                    self._buffer = f
                    store = True
                else:
                    f = self.path
            else:
                f = self._buffer
        self._compression, self._shape, self._typestr = read_meta_from_compressed_file(
            f, compression=self._compression
        )
        if store:
            self._compressed_bytes[self._compression] = f

    def _get_dicom_meta(self) -> dict:
        try:
            from pydicom import dcmread
            from pydicom.dataelem import RawDataElement
        except ImportError:
            raise ModuleNotFoundError(
                "Pydicom not found. Install using `pip install pydicom`"
            )
        if self.path and get_path_type(self.path) == "local":
            dcm = dcmread(self.path)
        else:
            dcm = dcmread(BytesIO(self.buffer))

        meta = {
            x.keyword: {
                "name": x.name,
                "tag": str(x.tag),
                "value": x.value
                if isinstance(x.value, (str, int, float))
                else x.to_json_dict(None, None).get("Value", ""),  # type: ignore
                "vr": x.VR,
            }
            for x in dcm
            if not isinstance(x.value, bytes)
        }
        return meta

    def _recompress(self, buffer: bytes, compression: str) -> bytes:
        # if get_compression_type(self._compression) != "image":
        #     raise ValueError(
        #         "Recompression with different format is only supported for images."
        #     )
        # img = Image.open(BytesIO(buffer))
        # if img.mode == "1":
        #     self._uncompressed_bytes = img.tobytes("raw", "L")
        # else:
        #     self._uncompressed_bytes = img.tobytes()
        return compress_array(self.array, compression)

    def _decompress(self, to_pil: bool = False):
        if not to_pil and self._array is not None:
            if self._uncompressed_bytes is None:
                self._uncompressed_bytes = self._array.tobytes()
            return
        compression = self.compression
        if compression is None and self._buffer is not None:
            if self.htype in ["list", "text", "json"]:
                from muller.core.serialize import bytes_to_text
                buffer = bytes(self._buffer)
                self._array = bytes_to_text(buffer, self.htype)
            else:
                self._array = np.frombuffer(self._buffer, dtype=self.dtype).reshape(self.shape)
        else:
            if self.path and get_path_type(self.path) == "local":
                compressed = self.path
            else:
                compressed = self.buffer

            if to_pil:
                self._pil = decompress_array(
                    compressed,
                    compression=compression,
                    shape=self.shape,
                    dtype=self.dtype,
                    to_pil=True,
                )  # type: ignore
            else:
                self._array = decompress_array(
                    compressed,
                    compression=compression,
                    shape=self.shape,
                    dtype=self.dtype,
                )
                self._uncompressed_bytes = self._array.tobytes()
                self._typestr = self._array.__array_interface__["typestr"]
                self._dtype = np.dtype(self._typestr).name

    def _read_from_path(self) -> bytes:  # type: ignore
        if self._buffer is None:
            path_type = get_path_type(self.path)
            try:
                if path_type == "local":
                    self._buffer = self._read_from_local()
                elif path_type == "roma":
                    self._buffer = self._read_from_roma()
                elif path_type == "http":
                    self._buffer = self._read_from_http()
                # TODO: do not support other path_type currently
            except Exception as e:
                raise SampleReadError(self.path) from e  # type: ignore
        return self._buffer  # type: ignore

    def _read_from_local(self) -> bytes:
        with open(self.path, "rb") as f:  # type: ignore
            return f.read()

    def _get_root_and_key(self, path):
        split_path = path.split("/", 2)
        if len(split_path) > 2:
            root, key = "/".join(split_path[:2]), split_path[2]
        else:
            root, key = split_path
        return root, key

    def _read_from_roma(self):
        if self.path is None:
            raise UnableToReadFromUrlError

        path = self.path.replace("roma://", "")  # type: ignore
        # If we have already specify a storage instance to retrieve a file from the path.
        if self.storage is not None:
            assert isinstance(self.storage, RomaProvider)
            return self.storage.get_object_from_full_url(path)
        # Otherwise, we use storage_factory to retrieve file from roma storage.
        # TODO: Check the correctness
        #root, key = self._get_root_and_key(path)
        #print(f"The root is {root}, the key is {key}.")
        #roma = storage_factory(RomaProvider, root="", creds=self._creds)
        roma = storage_factory(RomaProvider,
                               bucket_name=self._creds.get("bucket_name"),
                               region=self._creds.get("region"),
                               app_token=self._creds.get("app_token"),
                               vendor=self._creds.get("vendor"),
                               root="",
                               )
        return roma[path]

    def _read_from_http(self) -> bytes:
        if self.path is None:
            raise UnableToReadFromUrlError
        # assert self.path is not None

        if "Authorization" in self._creds:
            headers = {"Authorization": self._creds["Authorization"]}
        else:
            headers = {}
        proxies = self._creds["proxies"]
        # headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0",
        #            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        #            "Accept-Language": "en-US,en;q=0.9",
        #            "Connection": "close"
        #            }
        result = requests.get(self.path, headers=headers, proxies=proxies, verify=False)
        if result.status_code != 200:
            raise UnableToReadFromUrlError(self.path, result.status_code)
        # print(result.content)
        return result.content

    def _getexif(self) -> dict:
        if self.path and get_path_type(self.path) == "local":
            img = Image.open(self.path)
        else:
            img = Image.open(BytesIO(self.buffer))
        try:
            exif = getexif(img)
            img.close()
            return exif
        except Exception as e:
            warnings.warn(
                f"Error while reading exif data, possibly due to corrupt exif: {e}"
            )
            return {}

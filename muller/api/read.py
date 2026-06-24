# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/api/read.py
#
# Modifications Copyright (c) 2026 Xueling Lin

import pathlib
from typing import Optional, Dict, Union

from muller.core.sample import Sample
from muller.core.storage.provider import StorageProvider
from muller.util.path import convert_pathlib_to_string_if_needed


def read(
    path: Union[str, pathlib.Path],
    verify: bool = False,
    creds: Optional[Dict] = None,
    compression: Optional[str] = None,
    storage: Optional[StorageProvider] = None,
) -> Sample:
    """Utility that reads raw data from supported files into MULLER format.

        - Recompresses data into format required by the tensor if permitted by the tensor htype.
        - Copies data in the file if file format matches sample_compression of the tensor, to maximize upload speeds.
    Args:
        path (str): Path to a supported file.
        verify (bool):  If True, contents of the file are verified.
        creds (optional, Dict): Credentials or connection options for remote paths.
            HTTP reads may use values such as ``Authorization`` and ``proxies``;
            ROMA reads may use values such as ``bucket_name``, ``region``,
            ``app_token``, and ``vendor``.
        compression (optional, str): Format of the file. Only required if path does not have an extension.
        storage (optional, StorageProvider): Storage provider to use to retrieve remote files.
                                             Useful if multiple files are being read from same storage
                                             to minimize overhead of creating a new provider.

    Returns:
        Sample: Sample object. Call ``sample.array`` to get the ``np.ndarray``.
    """
    path = convert_pathlib_to_string_if_needed(path)

    return Sample(path, verify=verify, compression=compression, creds=creds, storage=storage)

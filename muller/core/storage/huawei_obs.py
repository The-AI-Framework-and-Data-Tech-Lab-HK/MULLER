# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

from typing import Set, Dict, Any

from obs import GetObjectHeader
from obs import ObsClient

from muller.core.storage.provider import StorageProvider
from muller.client.log import logger

class OBSProvider(StorageProvider):
    """Provide class for using Huawei Cloud Storage."""

    def __init__(self,
                 endpoint: str = None,
                 ak: str = None,
                 sk: str = None,
                 bucket_name: str = None,
                 root: str = None,

                 retry_times: int = 3
                 ):

        # Configurations for basic OBS auth setting
        self.endpoint = endpoint
        self.ak = ak
        self.sk = sk
        self.bucket_name = bucket_name
        self.root = root

        # Configurations for the file transfer setting
        self.retry_times = retry_times

        # Initialize the obs client
        self.obs_client = ObsClient(access_key_id=self.ak, secret_access_key=self.sk, server=self.endpoint)
        resp = self.obs_client.listBuckets(True)
        logger.info('List Buckets Succeeded')
        logger.info('requestId:', resp.requestId)
        logger.info('name:', resp.body.owner.owner_id)
        logger.info('create_date:', resp.body.owner.owner_name)
        logger.info('bucket_name', self.bucket_name)

    def __iter__(self):
        """Generator function that iterates over the keys of the S3Provider.

        Yields:
            str: the name of the object that it is iterating over.
        """
        yield from self._all_keys()

    def __len__(self):
        """Returns the number of files present at the root of the S3Provider."""
        return self.file_count

    def __setstate__(self, state_tuple):
        self.__init__(endpoint=state_tuple[0], ak=state_tuple[1], sk=state_tuple[2],
                      bucket_name=state_tuple[3], root=state_tuple[4])

    def __getstate__(self):
        return (self.endpoint, self.ak, self.sk, self.bucket_name, self.root)

    def __getitem__(self, key: str):
        """Gets the object present at the path.
        Args:
            key (str): the path relative to the root of the RomaProvider.

        Returns:
            bytes: The bytes of the object present at the path.

        Raises:
            KeyError: If an object is not found at the path.
            S3GetError: Any other error other than KeyError while retrieving the object.
        """
        if self.root:
            key = "".join((self.root, key))
        final_content = self._get(key)
        if final_content:
            return final_content
        raise KeyError(key)

    def __setitem__(self, key, content):
        """Sets the object present at the path with the value

        Args:
            key (str): the path relative to the root of the RomaProvider.
            content (bytes): the value to be assigned at the path.

        Raises:
            S3SetError: Any S3 error encountered while setting the value at the path.
            ReadOnlyError: If the provider is in read-only mode.
        """
        self.check_readonly()
        if self.root:
            key = "".join((self.root, key))
        content = bytes(memoryview(content))
        self._set(key, content)

    def __delitem__(self, key):
        pass

    def clear(self, prefix=""):
        pass

    def get_object(self, path: str):
        return self._get(path)

    def get_items(self, keys: Set[str], ignore_key_error: bool = False):
        # To improve the remote loading of tensor_meta json files - temporal version
        if ignore_key_error:
            raise NotImplementedError(f"ignore_key_error=True is not implemented for {self.__class__.__name__}")
        content_dict = {}
        for file in keys:
            content_dict.update({file: self._get(file)})
        return content_dict

    def set_items(self, contents: Dict[str, Any]):
        raise NotImplementedError(f"set_items is not implemented for {self.__class__.__name__}")

    def del_items(self, keys: Set[str]):
        raise NotImplementedError(f"del_items is not implemented for {self.__class__.__name__}")

    def _all_keys(self):
        """Helper function that lists all the objects present at the root of the S3Provider.

        Returns:
            set: set of all the objects found at the root of the S3Provider.

        Raises:
            S3ListError: Any S3 error encountered while listing the objects.
        """
        resp = self.obs_client.listObjects(self.bucket_name, self.root)
        key_list = []
        for content in resp.body.contents:
            if content.key != self.root:
                key_list.append(content.key)
        return key_list

    def _get(self, path: str):
        """
        Only return the content, no need write in the disk.
        """

        headers = GetObjectHeader()
        resp = self.obs_client.getObject(bucketName=self.bucket_name,
                                         objectKey=path,
                                         headers=headers,
                                         loadStreamInMemory=True
                                         )
        if resp.status < 300:
            return resp.body.buffer
        return b''

    def _set(self, path: str, content):
        self.obs_client.putContent(self.bucket_name, path, content)

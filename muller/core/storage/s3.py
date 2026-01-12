# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Set, Dict, Optional

import boto3

from muller.client.log import logger
from muller.core import StorageProvider


class S3Provider(StorageProvider):
    """Provider class for using the s3 platform"""

    def __init__(self,
                 endpoint: str = None,
                 ak: str = None,
                 sk: str = None,
                 bucket_name: str = None,
                 root: str = None,
                 ):

        self.endpoint = endpoint
        self.ak = ak
        self.sk = sk
        self.bucket_name = bucket_name
        self.root = root if root[-1] == "/" else root+"/"

        self._s3 = (boto3.Session(aws_access_key_id=ak, aws_secret_access_key=sk).
                    client("s3", endpoint_url=endpoint, verify=False))

    def __setstate__(self, state_tuple):
        self.__init__(endpoint=state_tuple[0], ak=state_tuple[1], sk=state_tuple[2],
                      bucket_name=state_tuple[3], root=state_tuple[4])

    def __getstate__(self):
        return (self.endpoint, self.ak, self.sk, self.bucket_name, self.root)

    def __iter__(self):
        """Generator function that iterates over the keys of the S3Provider.

        Yields:
            str: the name of the object that it is iterating over.
        """
        yield from self._all_keys()

    def __len__(self):
        return len(self._all_keys())

    def __getitem__(self, path: str):
        path = os.path.join(self.root, path)
        try:
            res = self._s3.get_object(Bucket=self.bucket_name, Key=path)
            return res["Body"].read()
        except Exception as e:
            raise KeyError(path) from e

    def __setitem__(self, path, content):
        """set s3 item"""
        self.check_readonly()
        path = os.path.join(self.root, path)
        content = bytes(memoryview(content))
        self._s3.put_object(Bucket=self.bucket_name, Key=path, Body=content)

    def __delitem__(self, path):
        self.check_readonly()
        path = os.path.join(self.root, path)
        self._s3.delete_object(Bucket=self.bucket_name, Key=path)

    def clear(self, prefix=""):
        self.check_readonly()
        self._clear(prefix)

    def set_items(self, contents: Dict[str, bytes], max_workers: Optional[int] = None):
        """flush item into disk with ThreadPool"""
        self.check_readonly()

        def set_file(path_bytes_dict: dict):
            self.__setitem__(*path_bytes_dict)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for _ in executor.map(set_file, contents.items()):
                pass

    def get_items(self, keys, ignore_key_error=False):
        # Sherry: To improve the remote loading of tensor_meta.json files
        def get_file(file):
            result = {}
            try:
                content = self.__getitem__(file)
                result = {file: content}
            except KeyError as e:
                if ignore_key_error:
                    pass
                else:
                    raise KeyError(str(file)) from e
            return result

        with ThreadPoolExecutor() as executor:
            future = executor.map(get_file, keys)

        result_dict = {
            k: v
            for d in list(future)
            for k, v in d.items()
        }
        return result_dict

    def get_object_size(self, key: str) -> int:
        key = os.path.join(self.root, key)
        try:
            resp = self._s3.head_object(Bucket=self.bucket_name, Key=key)
            object_size = resp["ContentLength"]
            return object_size
        except Exception as e:
            raise KeyError(key) from e

    def del_items(self, keys: Set[str]):
        key_list = []
        for key in keys:
            key_list.append({"Key": key})
            if len(key_list) < 1000:
                continue
            body = {"Objects": key_list}
            try:
                self._s3.delete_objects(
                    Delete=body,
                    Bucket=self.bucket_name
                )
            except Exception:
                logger.exception("Couldn't delete any objects from bucket %s.", self.bucket_name)
                raise

        if key_list:
            body = {"Objects": key_list}
            try:
                self._s3.delete_objects(
                    Delete=body,
                    Bucket=self.bucket_name
                )
            except Exception:
                logger.exception("Couldn't delete any objects from bucket %s.", self.bucket_name)
                raise

    def subdir(self, path: str, read_only: bool = False):
        """return a new s3Provider with the subdir as root"""
        sd = self.__class__(self.endpoint, self.ak, self.sk, self.bucket_name, os.path.join(self.root, path))
        sd.read_only = read_only
        return sd

    def files_partial_count(self) -> int:
        """Fetch the number of files with only the first call of _all_keys()"""
        return len(self._all_keys(single_fetch=True))

    def _all_keys(self, single_fetch: bool = False) -> Set[str]:
        key_set = set()
        prefix = self.root if self.root[-1] == "/" else self.root+"/"
        continuation_token = None
        while True:
            if continuation_token:
                resp = self._s3.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=prefix,
                    MaxKeys=1000,
                    ContinuationToken=continuation_token
                )
            else:
                resp = self._s3.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=prefix,
                    MaxKeys=1000,
                )
            contents = resp.get("Contents", None)
            if contents is None:
                break
            for content in contents:
                if content["Size"] == 0:
                    continue
                cur_key = content["Key"][len(self.root):]
                if cur_key != "" and cur_key[-1] != "/":
                    key_set.add(cur_key)

            is_truncated = resp["IsTruncated"]
            if single_fetch:
                break
            if "NextContinuationToken" not in resp:
                break
            if not is_truncated:
                break
            continuation_token = resp["NextContinuationToken"]
        return key_set

    def _clear(self, prefix=""):
        prefix = os.path.join(self.root, prefix)
        prefix = prefix if prefix[-1] == "/" else prefix+"/"
        continuation_token = None
        while True:
            key_set = set()
            if continuation_token:
                resp = self._s3.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=prefix,
                    MaxKeys=1000,
                    ContinuationToken=continuation_token
                )
            else:
                resp = self._s3.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=prefix,
                    MaxKeys=1000,
                )
            contents = resp.get("Contents", None)
            if contents is None:
                break
            for content in contents:
                cur_key = content["Key"]
                key_set.add(cur_key)

            is_truncated = resp["IsTruncated"]
            if "NextContinuationToken" not in resp:
                break
            continuation_token = resp["NextContinuationToken"]
            if not key_set and not is_truncated:
                break
            try:
                body = {"Objects": [{"Key": key} for key in key_set]}
                response = self._s3.delete_objects(
                    Delete=body,
                    Bucket=self.bucket_name
                )
                if "Errors" in response:
                    logger.warning(
                        "Could not delete objects '%s' from bucket '%s'.",
                        [
                            f"{del_obj['Key']}: {del_obj['Code']}"
                            for del_obj in response["Errors"]
                        ],
                        self.bucket_name,
                    )
            except Exception:
                logger.exception("Couldn't delete any objects from bucket %s.", self.bucket_name)
                raise

            if not is_truncated:
                break

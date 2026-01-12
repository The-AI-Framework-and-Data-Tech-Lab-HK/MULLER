# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin
# Acknowledgement to Tao Yuheng, You Tianming, Gao Han and Kong Xiangzhou who provided help to use Roma.

import base64
import json
import queue
import ssl
import sys
import threading
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Set, Dict, Any
from urllib import request

import psutil
import requests

from muller.client.log import logger
from muller.core.storage.provider import StorageProvider


class RomaProvider(StorageProvider):
    """Provide class for using Huawei Cloud Storage."""

    def __init__(self,
                 endpoint: str = "http://roma.huawei.com",
                 api: str = "/csb/rest/s3/bucket/endpoint",
                 bucket_name: Optional[str] = None,
                 region: Optional[str] = None,
                 app_token: Optional[str] = None,
                 vendor: Optional[str] = None,
                 root: Optional[str] = "",  # The root path

                 buffer_size: int = 65536,  # The default value of buffer_size is 65536.
                 retry_times: int = 3,
                 big_file: int = 100 * 1024 * 1024,  # The default value of big file is 100M.
                 thread_num: int = 12,
                 package_size: int = 50 * 1024 * 1024,  # The default value of package is 50M.
                 debug: bool = False,
                 show_speed: bool = True,  # To display the current upload speed, <psutil> is required.
                 fail_json_storage_path: str = "./fail.json",
                 single_file: bool = False,  # If we only download single file, then we set this as True.
                 time_wait: int = 10,  # The default waiting time for download thread (in seconds).
                 queue_size: int = 5000,  # The max size of download queue.
                 ):
        # Configurations for basic OBS auth setting
        self.endpoint = endpoint
        self.api = api
        self.bucket_name = bucket_name
        self.region = region
        self.app_token = app_token
        self.vendor = vendor
        self.root = root

        # Configurations for the file transfer setting
        self.buffer_size = buffer_size
        self.retry_times = retry_times
        self.big_file = big_file
        self.thread_num = thread_num
        self.package_size = package_size
        self.debug = debug
        self.show_speed = show_speed
        self.fail_json_storage_path = fail_json_storage_path

        # Other configurations for upload
        self.cond = threading.Condition()
        self.context = ssl._create_unverified_context()
        self.print_lock = threading.Lock()
        self.file_lock = threading.Lock()
        self.upload_queue = queue.Queue()
        self.big_file_upload_queue = queue.Queue()
        self.range_queue = queue.Queue()
        self.big_file_ls = []
        self.file_count = 0
        self.uploaded_count = 0
        self.big_file_process_count = 0
        self.big_file_process_total = 0
        self.error_count = 0
        self.current_speed = ""
        self.start_count_speed = True
        self.session = requests.Session()
        self.is_file = False
        self.retry_map = defaultdict(int)
        self.error_map = dict()

        # Other configurations for download
        self.single_file = single_file
        self.time_wait = time_wait
        self.queue_size = queue_size
        self.download_queue = queue.Queue(self.queue_size)
        self.big_file_download_queue = queue.Queue()
        self.downloaded_count = 0
        self.has_next = True
        self.next_marker = ""

        # move here to save some time!
        self.csb_file_server = self._get_file_server_endpoint()
        self._bucket_auth()

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
        self.__init__(endpoint=state_tuple[0], api=state_tuple[1], bucket_name=state_tuple[2],
                      region=state_tuple[3], app_token=state_tuple[4], vendor=state_tuple[5], root=state_tuple[6])

    def __getstate__(self):
        return self.endpoint, self.api, self.region, self.vendor, self.root

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
        key = "".join((self.root, key))
        final_content = self.get_bytes(key)
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
        key = "".join((self.root, key))
        content = bytes(memoryview(content))
        try:
            self._set(key, content)
        except Exception:
            for i in range(self.retry_times):
                try:
                    self._set(key, content)
                    logger.info(f'{key} successfully set after {i + 1} retries')
                    return
                except Exception:
                    pass
            logger.info(f'{key} fails in set_item...')

    def __delitem__(self, key):
        pass

    def clear(self, prefix=""):
        pass

    def get_bytes(self, path: str, start_byte: Optional[int] = None, end_byte: Optional[int] = None, ):
        """Gets the object present at the path within the given byte range.

        Args:
            path (str): The path relative to the root of the provider.
            start_byte (int, optional): If only specific bytes starting from ``start_byte`` are required.
            end_byte (int, optional): If only specific bytes up to end_byte are required.

        Returns:
            bytes: The bytes of the object present at the path within the given byte range.
        """
        try:
            return self._get_bytes(path, start_byte, end_byte)
        except Exception:
            # Retry for several times
            for i in range(self.retry_times):
                try:
                    content = self._get_bytes(path, start_byte, end_byte)
                    logger.info(f'{path} successfully got after {i + 1} retries')
                    return content
                except Exception:
                    pass
            logger.info(f'{path} fails in get_bytes...')

    def get_object_from_full_url(self, url: str):
        try:
            content = self._get_single_obj_by_key(url)
            return content
        except Exception:
            # Retry for several times
            for i in range(self.retry_times):
                try:
                    content = self._get_single_obj_by_key(url)
                    logger.info(f'{url} successfully got after {i + 1} retries')
                    return content
                except Exception:
                    logger.info("fails in get_object_from_full_url")
            logger.info(f'{url} fails in get_object_from_full_url...')

    def get_items(self, keys: Set[str], ignore_key_error: bool = False, multi_retry: int = 3):
        # Sherry: To improve the remote loading of tensor_meta.json files
        if ignore_key_error:
            raise NotImplementedError(f"ignore_key_error=True is not implemented for {self.__class__.__name__}")
        content_dict = {}
        try:
            with ThreadPoolExecutor(max_workers=self.thread_num + 4, thread_name_prefix="Python Downloader") as pool:
                ls_thread = []
                for file in keys:
                    path = "".join((self.root, file))
                    ls_thread.append(pool.submit(self._get_object_with_return_key, path))
                results = as_completed(ls_thread)
                for re in results:
                    content_dict.update({re.result()[0][len(self.root):]: re.result()[1]})
            return content_dict
        except Exception:
            logger.info("fails in get items")

    def set_items(self, contents: Dict[str, Any], multi_retry=3):
        # Sherry: To improve the remote uploading of multiple meta files
        try:
            with ThreadPoolExecutor(max_workers=self.thread_num + 4, thread_name_prefix="Python Uploader") as pool:
                ls_thread = []
                for file, content in contents.items():
                    path = "".join((self.root, file))
                    ls_thread.append(pool.submit(self._set_single_obj_by_key, path, content))
            return
        except Exception:
            multi_retry -= 1
            if multi_retry:
                time.sleep(10)
                try:
                    return self.set_items(contents, multi_retry)
                except Exception:
                    logger.info("fails in set items")
            logger.info(f'fails in set_items: {contents.keys()}...')

    def del_items(self, keys: Set[str]):
        raise NotImplementedError(f"del_items is not implemented for {self.__class__.__name__}")

    def _all_keys(self):
        """Helper function that lists all the objects present at the root of the S3Provider.

        Returns:
            set: set of all the objects found at the root of the S3Provider.

        Raises:
            S3ListError: Any S3 error encountered while listing the objects.
        """
        result_dict = self._get_object_key(next_marker="", bucket_path=self.root)
        key_list = []
        for detailed_obj in result_dict["objectKeys"]:
            key_list.append(detailed_obj["objectKey"])
        return key_list

    def _get_bytes(self, path, start_byte: Optional[int] = None, end_byte: Optional[int] = None):
        if start_byte is not None and end_byte is not None:
            if start_byte == end_byte:
                return b""
            range = (start_byte, end_byte)
        elif start_byte is not None:
            range = (start_byte,)
        elif end_byte is not None:
            range = (0, end_byte)
        else:
            range = None
        content = self._get_single_obj_by_key(path)
        if range is None:
            return content
        return content[range[0]:range[1]]

    def _set(self, file_name, content):
        b64_object_name = base64.urlsafe_b64encode(file_name.encode()).decode()
        path = f"/rest/boto3/s3/{self.vendor}/{self.region}/{self.app_token}/{self.bucket_name}/{b64_object_name}"

        headers = {
            "Content-Type": "application/json",
            "csb-token": self.app_token,
            'Connection': 'close'
        }
        resp = self.session.put(self.csb_file_server + path, data=content, headers=headers)
        if resp.status_code != 200:
            raise Exception(
                f"upload file error. return code {resp.status_code}, exception:{resp.content.decode('utf-8')} ")

    def _get_file_server_endpoint(self):
        try:
            url = self.endpoint + self.api
            param = "?" + "bucketid=" + self.bucket_name + "&token=" \
                    + self.app_token + "&vendor=" + self.vendor + "&region=" + self.region
            req = request.Request(url=url + param)
            res = request.urlopen(req, context=self.context)
            result = res.read().decode(encoding='utf-8')

            if self.debug:
                logger.info("request file server endpoint result: " + result)
            result_dict = json.loads(result)
            if result_dict["success"]:
                return result_dict["result"]
            raise Exception(result_dict["msg"])
        except Exception:
            traceback.print_exc()

    def _bucket_auth(self):
        bucket_auth_endpoint = self.csb_file_server + '/rest/boto3/s3/bucket-auth?vendor=' + self.vendor \
                               + '&region=' + self.region + '&bucketid=' + self.bucket_name \
                               + '&apptoken=' + self.app_token
        req = request.Request(url=bucket_auth_endpoint)
        res = request.urlopen(req)
        result = res.read().decode(encoding='utf-8')
        if self.debug:
            logger.info("bucket auth result: " + result)
        result_dict = json.loads(result)
        if not result_dict["success"]:
            raise Exception(result_dict["msg"])

    def _do_download_queue(self, bucket_path):
        result_dict = self._get_object_key("", bucket_path)
        while result_dict["nextmarker"]:
            try:
                result_dict = self._get_object_key(result_dict["nextmarker"], bucket_path)
            except Exception:
                self.error_map[result_dict["nextmarker"]] = time.time()
                time.sleep(5)
                result_dict = self._get_object_key(result_dict["nextmarker"], bucket_path)

    def _get_object_key(self, next_marker, bucket_path):
        path = "/rest/boto3/s3/list/bucket/objectkeys?"
        object_key = base64.urlsafe_b64encode(bucket_path.encode()).decode()
        param = f"vendor={self.vendor}&region={self.region}&bucketid={self.bucket_name}" \
                f"&apptoken={self.app_token}&objectkey={object_key}&nextmarker={next_marker}"
        headers = {
            "Content-Type": "application/json",
            "csb-token": self.app_token
        }
        result = self.session.get(self.csb_file_server + path + param, headers=headers)
        result_dict = json.loads(result.content)

        if not result_dict["nextmarker"]:
            self.has_next = False
        if not result_dict["success"]:
            logger.error(result.text)
            self.error_map["next_marker" + "@" + next_marker] = result.text
            raise Exception("get object key failed")
        self.file_count += len(result_dict["objectKeys"])
        for object_key in result_dict["objectKeys"]:
            if int(object_key["size"]) > self.big_file:
                self.big_file_download_queue.put(object_key)
                self.file_count -= 1
            else:
                self.download_queue.put(object_key)
        return result_dict

    def _get_delta(self, sleep_time: int):
        if self.show_speed:
            while self.start_count_speed:
                before = psutil.net_io_counters().bytes_sent
                time.sleep(sleep_time)
                now = psutil.net_io_counters().bytes_sent
                delta = (now - before) / (1024 * 1024 * 1)
                self.current_speed = "  {:.3f} MB/s".format(delta)

    def _do_get(self):
        logger.info("download_queue:", self.download_queue, self.download_queue.qsize())
        download_content_list = []
        while self.download_queue.qsize() > 0 or self.has_next:
            try:
                download_content = self.download_queue.get(timeout=self.time_wait)
                logger.info("download_content", download_content)
            except Exception:
                break
            object_key = download_content["objectKey"]
            try:
                single_content = self._get_object(object_key)
                download_content.update({"content": single_content})
                download_content_list.append(download_content)
                self.downloaded_count += 1
                with self.print_lock:
                    logger.info("\r", end="")
                    logger.info("Download progress: {}/{} - {}%: ".format(self.downloaded_count, self.file_count,
                                                                int((self.downloaded_count / self.file_count) * 100)),
                          "▋" * (int(self.downloaded_count / self.file_count * 50)) + self.current_speed, end="")
                    sys.stdout.flush()
            except Exception:
                traceback.print_exc()

                self.retry_map[str(download_content)] += 1
                if self.retry_map[str(download_content)] > self.retry_times:
                    logger.error(f"{download_content} try {self.retry_times} times failed")
                    self.error_map[str(download_content)] = traceback.format_exc()
                    continue
                self.download_queue.put(download_content)
        self.start_count_speed = False
        return download_content_list

    def _get_object_with_return_key(self, object_name: str):
        # Sherry: To improve the remote loading of tensor_meta.json files
        b64_object_name = base64.urlsafe_b64encode(object_name.encode()).decode()
        path = f"/rest/boto3/s3/{self.vendor}/{self.region}/{self.app_token}/{self.bucket_name}/{b64_object_name}"
        headers = {
            "Content-Type": "application/json",
            "csb-token": self.app_token,
            'Connection': 'close'
        }

        with self.session.get(self.csb_file_server + path, headers=headers) as result:
            if result.status_code > 200:
                raise Exception(
                    f"download file error. return code {result.status_code}, "
                    f"exception:{result.content.decode('utf-8')} ")
            return (object_name, result.content)

    def _get_single_obj_by_key(self, object_name):
        # Get the key
        b64_object_name = base64.urlsafe_b64encode(object_name.encode()).decode()
        path = f"/rest/boto3/s3/{self.vendor}/{self.region}/{self.app_token}/{self.bucket_name}/{b64_object_name}"
        headers = {
            "Content-Type": "application/json",
            "csb-token": self.app_token,
            'Connection': 'close'
        }

        with self.session.get(self.csb_file_server + path, headers=headers) as result:
            if result.status_code > 200:
                raise Exception(
                    f"download file error. return code {result.status_code}, "
                    f"exception:{result.content.decode('utf-8')} ")
            return result.content

    def _set_single_obj_by_key(self, object_name, object_val):
        b64_object_name = base64.urlsafe_b64encode(object_name.encode()).decode()
        path = f"/rest/boto3/s3/{self.vendor}/{self.region}/{self.app_token}/{self.bucket_name}/{b64_object_name}"

        headers = {
            "Content-Type": "application/json",
            "csb-token": self.app_token,
            'Connection': 'close'
        }
        resp = self.session.put(self.csb_file_server + path, data=object_val, headers=headers)
        if resp.status_code != 200:
            raise Exception(
                f"upload file error. return code {resp.status_code}, exception:{resp.content.decode('utf-8')} ")

# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import base64
import hmac
import os
import time
import uuid
from functools import wraps
from hashlib import sha256
from typing import Callable, Optional, Dict
import json
import requests

from muller.util.exceptions import InvalidPermissionError


def validate_permissions(func: Callable):
    """ 校验用户权限信息 """
    @wraps(func)
    def inner(*args, **kwargs):
        env_file_path = '/tmp/user/env.info'
        if 'path' not in kwargs and '/' not in args[0] and os.path.exists(env_file_path):
            with open(env_file_path, 'r') as file:
                env_data = json.load(file)
            service_host = env_data.get('dsw.service.host')
            url = f"{service_host}/dataset-mgmt/tensor-dataset/permissions/query"

            request_data = {
                "datasetId": args[0]
            }
            response = requests.post(url, json=request_data, headers=generator_headers())
            if response.status_code != 200:
                raise InvalidPermissionError(response.status_code)

            response_data = response.json()
            tensor_ds_permission = response_data.get("dataPermission", {})
            obs_path = tensor_ds_permission.get("obsPath", "")
            permissions = tensor_ds_permission.get("permissions", {})
            if not permissions:
                raise InvalidPermissionError(response.status_code)

            if permissions.get("read", False) and not permissions.get("write", True):
                kwargs['read_only'] = True
            elif not all(permissions.get(permission, False) for permission in permissions):
                raise InvalidPermissionError(response.status_code)

            args = list(args)
            args[0] = obs_path
            args = tuple(args)

        return func(*args, **kwargs)

    return inner


def get_string_to_sign(appid, appsecret, timestamp, requestid):
    """ 生成 sign """
    string_to_sign = "appId={}&appSecret={}&timestamp={}&requestId={}".format(appid, appsecret, timestamp, requestid)
    return string_to_sign


def get_signature_string(appsecret, stringtosign):
    """ 生成签名信息 """
    key = appsecret.encode('utf-8')
    data = stringtosign.encode('utf-8')
    sign = base64.b64encode(hmac.new(key, data, digestmod=sha256).digest())
    sign = str(sign, 'utf-8')
    return sign


def generator_headers(headers: Optional[Dict] = None) -> Dict:
    """ 生成 headers """
    if headers is None:
        headers = {}

    # Obtain timestamp
    time_stamp = str(int(round(time.time() * 1000)))
    # Obtain UUID
    request_id = str(uuid.uuid4())

    usr_file_path = '/tmp/user/user.info'
    try:
        with open(usr_file_path, 'r') as file:
            user_info = json.load(file)
    except FileNotFoundError as e:
        raise RuntimeError(f"File {usr_file_path} not found.") from e
    operatorid = user_info['uid']
    tenant = user_info['tenant']
    app_id = user_info['appId']
    app_secret = user_info['appSecret']
    authorization = 'HmacSHA256 ' + get_signature_string(app_secret,
                                                         get_string_to_sign(app_id, app_secret, time_stamp, request_id))

    # Append headers
    append_headers = {'AppId': app_id,
                      'Authorization': authorization,
                      'RequestId': request_id,
                      'Timestamp': time_stamp,
                      'operatorid': operatorid,
                      'tenant': tenant,
                      'Content-Type': 'application/json'}

    headers.update(append_headers)
    return headers

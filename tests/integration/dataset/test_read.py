# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import logging
import os
import sys

import pytest

import muller


@pytest.mark.skipif("--reading_test==local" not in sys.argv,
                    reason="It should be skipped if we do not need to test muller.read().")
def test_read_from_local(storage):
    # Read from raw files
    """
        The demo of the read function. The supported file types include:
        * Image: "bmp", "dib", "gif", "ico", "jpeg", "jpeg2000", "pcx", "png", "ppm", "sgi", "tga",
                 "tiff", "webp", "wmf", "xbm"
        * Audio: "flac", "mp3", "wav"
        * Video: "mp4", "mkv", "avi"
        * Dicom: "dcm"
        * Nifti: "nii", "nii.gz"
    """
    muller_read_path = "/data/muller_test_data_01/test/"
    if os.environ.get('MULLER_TEST_DATA_PATH'):
        muller_read_path = os.path.join(os.environ.get('MULLER_TEST_DATA_PATH'), muller_read_path[1:])
    if not os.path.exists(muller_read_path):
        raise Exception("Please prepare your testing dataset to execute `test_read_from_local`.")
    sample_files = {
        "jpg_file": os.path.join(muller_read_path, "dog.157.jpg"),
        "png_file": os.path.join(muller_read_path, "0208.png"),
        "bmp_file": os.path.join(muller_read_path, "carsgraz_420.bmp"),
        "mp3_file": os.path.join(muller_read_path, "example_5MB.mp3"),
        "wav_file": os.path.join(muller_read_path, "gettysburg10.wav"),
        "mp4_file": os.path.join(muller_read_path, "sample-10s.mp4"),
        "avi_file": os.path.join(muller_read_path, "example_24MB.avi"),
        "dcm_file": os.path.join(muller_read_path, "MR000000.dcm"),
        "nii_file": os.path.join(muller_read_path, "ExBox13/T1_brain.nii.gz"),
    }
    test_sample_0 = muller.read(path=sample_files["jpg_file"])
    assert test_sample_0.array[0][0][0] == 183
    assert test_sample_0.shape == (400, 500, 3)
    assert test_sample_0.dtype == "uint8"

    test_sample_1 = muller.read(path=sample_files["png_file"])
    assert test_sample_1.array[0][0][0] == 5
    assert test_sample_1.shape == (208, 208, 4)
    assert test_sample_1.dtype == "uint8"

    test_sample_2 = muller.read(path=sample_files["bmp_file"])
    assert test_sample_2.array[0][0][0] == 9
    assert test_sample_2.shape == (480, 640, 3)
    assert test_sample_2.dtype == "uint8"

    test_sample_3 = muller.read(path=sample_files["mp3_file"])
    assert test_sample_3.shape == (5863680, 2)
    assert test_sample_3.dtype == "float32"

    test_sample_4 = muller.read(path=sample_files["wav_file"])
    assert test_sample_4.shape == (220568, 1)
    assert test_sample_4.dtype == "float32"

    test_sample_5 = muller.read(path=sample_files["mp4_file"])
    assert test_sample_5.array[0][0][0][0] == 2
    assert test_sample_5.shape == (303, 1080, 1920, 3)
    assert test_sample_5.dtype == "uint8"

    test_sample_6 = muller.read(path=sample_files["avi_file"])
    assert test_sample_6.array[0][0][0][0] == 206
    assert test_sample_6.shape == (1344, 720, 1280, 3)
    assert test_sample_6.dtype == "uint8"

    test_sample_6 = muller.read(path=sample_files["dcm_file"])
    assert test_sample_6.array[0][0][0] == [0]
    assert test_sample_6.shape == (512, 512, 1)
    assert test_sample_6.dtype == "uint16"

    test_sample_7 = muller.read(path=sample_files["nii_file"])
    assert test_sample_7.shape == (290, 320, 208)
    assert test_sample_7.dtype == "float32"


@pytest.mark.skipif("--reading_test==http" not in sys.argv,
                    reason="It should be skipped if we do not need to test muller.read().")
def test_read_from_http(storage):
    credential_info = {
        "proxies": {"http": "xxx",
                    "https": "xxx"},
    }
    sample_files = {
        "https_file": "https://onlinejpgtools.com/images/examples-onlinejpgtools/cat-front.jpg",
    }
    if credential_info["proxies"]["http"].find("account") == -1:
        test_sample_0 = muller.read(path=sample_files["https_file"], creds=credential_info)
        assert test_sample_0.array[0][0][0] == 129
        assert test_sample_0.shape == (425, 640, 3)
        assert test_sample_0.dtype == "uint8"
    else:
        logging.info("Please specify your account and pwd to execute `test_read_from_http`")


@pytest.mark.skipif("--reading_test==roma" not in sys.argv,
                    reason="It should be skipped if we do not need to test muller.read().")
def test_read_from_roma(storage):
    credential_info = {
        "bucket_name": "xxx",
        "region": "xxx",
        "app_token": "xxx",
        "vendor": "xxx"
    }
    sample_files = {
        "roma_file": "roma://muller_test_data_01/test/dog.157.jpg"
    }
    try:
        test_sample_0 = muller.read(path=sample_files["roma_file"], creds=credential_info)
        assert test_sample_0.array[0][0][0] == 183
        assert test_sample_0.shape == (400, 500, 3)
        assert test_sample_0.dtype == "uint8"
    except ConnectionError as e:
        logging.info(f"ConnectionError in roma. You may need to unset http/https proxies. Detailed info: {e}")


if __name__ == '__main__':
    pytest.main(["-s", "--storage", "local", "test_read.py"])

# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Bingyu Liu

import time
import logging
import threading
import pytest
import muller
from muller.util.exceptions import DatasetCorruptError
from muller.util.version_control import integrity_check
from tests.utils import official_path, official_creds
from tests.constants import TEST_PROCESS_INTERRUPT


def long_running_task(stop_event, ds):
    """test interrupt append samples in single process"""
    for i in range(10000):
        ds.labels.extend([i])
        ds.new_labels.extend([i])
        if stop_event.is_set():
            logging.info("long_running_task Task interrupted!")
            return


def many_append_with_checkpoint(stop_event, ds):
    """Function to append multi-process with checkpoint."""
    @muller.compute
    def add_data(values, sample_out):
        if stop_event.is_set():
            logging.info("many_append_with_checkpoint Task interrupted!")
            return
        sample_out.labels.append(values)
        sample_out.new_labels.append(values)

    values = [0] * 10000
    with ds:
        add_data().eval(values, ds, num_workers=2, progressbar=True, scheduler="processed", checkpoint_interval=1000)


def many_append_without_checkpoint(stop_event, ds):
    """Function to append multi-process without checkpoint."""
    @muller.compute
    def add_data(values, sample_out):
        if stop_event.is_set():
            logging.info("many_append_without_checkpoint Task interrupted!")
            return
        sample_out.labels.append(values)
        sample_out.new_labels.append(values)

    values = [0] * 10000
    with ds:
        add_data().eval(values, ds, num_workers=2, progressbar=True, scheduler="processed")


def many_pop(stop_event, ds):
    """Function to pop many times."""
    for _ in range(10000):
        if stop_event.is_set():
            logging.info("many_pop Task interrupted!")
            return
        ds.pop(0)


def test_single_termination(storage):
    """Function to test single process append."""
    stop_event = threading.Event()
    ds = muller.dataset(path=official_path(storage, TEST_PROCESS_INTERRUPT),
                       creds=official_creds(storage), overwrite=True)
    ds.create_tensor(name="labels", htype="generic", dtype="int")
    ds.create_tensor(name="new_labels", htype="generic", dtype="int")
    ds.labels.extend([0, 1, 2, 3, 4])
    ds.new_labels.extend([0, 1, 2, 3, 4])
    ds.commit()
    task_thread = threading.Thread(target=long_running_task, args=(stop_event, ds))

    # Start a thread
    task_thread.start()

    time.sleep(2)

    # Make sure the thread is alive
    assert task_thread.is_alive()

    # Send a stop signal
    stop_event.set()

    # Wait for the thread to be killed
    task_thread.join(timeout=2)

    # Make sure the thread is killed
    assert not task_thread.is_alive()

    # Check the results
    try:
        integrity_check(ds)
        assert True, "No exception raises"
    except DatasetCorruptError as e:
        assert False, f"Raises DatasetCorruptError {e}"
    ds.reset()
    assert len(ds.labels.numpy(aslist=True)) == len(ds.new_labels.numpy(aslist=True)) == 5


def test_multi_termination(storage):
    """Function to test multi-process append with checkpoint."""
    stop_event = threading.Event()
    ds = muller.dataset(path=official_path(storage, TEST_PROCESS_INTERRUPT),
                       creds=official_creds(storage), overwrite=False)
    assert len(ds.labels.numpy(aslist=True)) == len(ds.new_labels.numpy(aslist=True)) == 5
    task_thread = threading.Thread(target=many_append_with_checkpoint, args=(stop_event, ds))

    # Start a thread
    task_thread.start()

    time.sleep(0.5)

    # Make sure the thread is still alive
    assert task_thread.is_alive()

    # Send a stop signal
    stop_event.set()

    # Wait for the thread to be killed
    task_thread.join(timeout=2)

    # Make sure the thread is killed
    assert not task_thread.is_alive()

    # Check the results
    try:
        integrity_check(ds)
        assert True, "No exception raises"
    except DatasetCorruptError as e:
        assert False, f"Raises DatasetCorruptError {e}"
    assert len(ds.labels.numpy(aslist=True)) == len(ds.new_labels.numpy(aslist=True))


def test_multi_termination_2(storage):
    """Function to test multi-process append without check point"""
    stop_event = threading.Event()
    ds = muller.dataset(path=official_path(storage, TEST_PROCESS_INTERRUPT),
                       creds=official_creds(storage), overwrite=True)
    ds.create_tensor(name="labels", htype="generic", dtype="int")
    ds.create_tensor(name="new_labels", htype="generic", dtype="int")
    ds.labels.extend([0, 1, 2, 3, 4])
    ds.new_labels.extend([0, 1, 2, 3, 4])
    ds.commit()

    task_thread = threading.Thread(target=many_append_without_checkpoint, args=(stop_event, ds))

    # Start a thread
    task_thread.start()

    time.sleep(0.1)

    # Make sure the thread is still alive
    assert task_thread.is_alive()

    # Send a stop signal
    stop_event.set()

    # Wait for the thread to be killed
    task_thread.join(timeout=2)

    # Make sure the thread is killed
    assert not task_thread.is_alive()

    # Check the results
    try:
        integrity_check(ds)
        assert True, "No exception raises"
    except DatasetCorruptError as e:
        assert False, f"Raises DatasetCorruptError {e}"
    assert len(ds.labels.numpy(aslist=True)) == len(ds.new_labels.numpy(aslist=True))


def test_pop_termination(storage):
    """Function to test pop termination."""
    stop_event = threading.Event()
    ds = muller.dataset(path=official_path(storage, TEST_PROCESS_INTERRUPT),
                       creds=official_creds(storage), overwrite=True)
    ds.create_tensor(name="labels", htype="generic", dtype="int")
    ds.create_tensor(name="new_labels", htype="generic", dtype="int")
    many_append_with_checkpoint(stop_event, ds)
    assert len(ds.labels.numpy(aslist=True)) == len(ds.new_labels.numpy(aslist=True)) == 10000

    task_thread = threading.Thread(target=many_pop, args=(stop_event, ds))

    # Start a thread
    task_thread.start()

    time.sleep(1)

    # Make sure the thread is alive
    assert task_thread.is_alive()

    # Send a stop signal
    stop_event.set()

    # Wait for the thread to be killed
    task_thread.join(timeout=2)

    # Make sure the thread is killed
    assert not task_thread.is_alive()

    # Check the results
    try:
        integrity_check(ds)
        assert True, "No exception raises"
    except DatasetCorruptError as e:
        assert False, f"Raises DatasetCorruptError {e}"
    assert len(ds.labels.numpy(aslist=True)) == len(ds.new_labels.numpy(aslist=True))


if __name__ == '__main__':
    pytest.main()

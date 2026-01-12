# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

from multiprocessing import Process, Queue

from mindspore.mindrecord import FileWriter

import muller
from muller.util.compute import get_compute_provider


def create_schema(tensor_dict):
    """Create mindrecord schema."""
    schema_json = {}
    for tensor_name in tensor_dict:
        tensor_object = tensor_dict[tensor_name]
        tensor_htype = tensor_object.htype
        tensor_dtype = tensor_object.dtype.name
        if tensor_htype == 'text':
            schema_json[tensor_name] = {"type": "string"}
        elif tensor_htype in ['int', 'class_label'] or tensor_dtype == 'int32':
            schema_json[tensor_name] = {"type": "int32"}
        elif tensor_htype == 'float' or tensor_dtype == 'float32':
            schema_json[tensor_name] = {"type": "float32"}
        elif tensor_htype == 'image':
            schema_json[tensor_name] = {"type": "bytes"}
    return schema_json


def producer(q, ds, batch_size, scheduler):
    """Producer keeps getting numpy data from MULLER, and putting them in to a queue."""
    tensor_dict = ds.tensors
    for start in range(0, len(ds), batch_size):
        numpy_data = {}
        end = start + batch_size if start + batch_size < len(ds) else len(ds)

        def get_numpy(tensor_name, s=start, e=end):
            result = {}
            tensor_object = tensor_dict[tensor_name]
            result[tensor_name] = tensor_object[s: e].numpy()
            return result

        compute = get_compute_provider(scheduler=scheduler, num_workers=len(tensor_dict))
        result = compute.map(get_numpy, list(tensor_dict))
        for element in result:
            numpy_data.update(element)
        compute.close()

        q.put(_generate_data(ds, end, start, tensor_dict, numpy_data))

    q.put(None)


def consumer(q, writer):
    """Consumer keeps getting data from queue and writing as MindRecord."""
    while True:
        data = q.get()
        if data is None:
            break
        writer.write_raw_data(data)
    writer.commit()


def create_mindrecord(dataset: muller.Dataset, file_name, shard_num, batch_size, overwrite, scheduler):
    """Create mindrecord dataset."""
    writer = FileWriter(file_name, shard_num, overwrite)
    schema = create_schema(dataset.tensors)
    writer.add_schema(schema, file_name)
    q = Queue()
    p1 = Process(target=producer, args=(q, dataset, batch_size, scheduler,))
    c1 = Process(target=consumer, args=(q, writer,))
    p1.start()
    c1.start()
    p1.join()
    c1.join()


def _generate_data(ds, end, start, tensor_dict, numpy_data):
    data = []
    with ds:
        for i in range(end - start):
            temp = {}
            for tensor_name in tensor_dict:
                temp[tensor_name] = numpy_data[tensor_name][i].item()
            data.append(temp)
    return data

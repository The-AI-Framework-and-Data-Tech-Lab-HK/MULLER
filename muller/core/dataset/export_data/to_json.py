# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import json
from multiprocessing import Pool

import jsonlines


def process_sub_dataset(temp_ds, tensors=None):
    """ Convert the target temp_dataset into a list of dict,
        while each dict contains the tensor name as key and the real value of the tensor as item.
    """
    if tensors:
        target_tensors = tensors
    else:
        target_tensors = temp_ds.tensors
    new_result = [
        dict(zip(target_tensors, items))
        for items in zip(*[temp_ds[k].numpy_continuous().flatten().tolist()
                           if temp_ds.tensors[k].htype != "list"
                           else [arr.tolist() for arr in temp_ds[k].numpy(aslist=True)]
                           for k in target_tensors])
    ]
    return new_result


def dump_to_path(data, path):
    """ Dump the data to a json file or a jsonl file in the target path."""
    if path.endswith("jsonl"):
        with jsonlines.open(path, 'w') as writer:
            for line in data:
                writer.write(line)
    else:  # path.endswith("json"):
        with open(path, 'w') as f:
            json.dump(data, f, ensure_ascii=False)


def to_json(ds, target_path, tensors=None, num_workers=1):
    """Export the dataset to a json file"""
    def divide_ds(ds, num_workers):
        """Divide the dataset into chunks according to the number of workers"""
        sub_ds_list = []
        batch_size = len(ds)//num_workers
        for i in range(0, num_workers):
            if i == num_workers-1:
                temp_ds = ds[i*batch_size:]
            else:
                temp_ds = ds[i*batch_size:(i+1)*batch_size]
            sub_ds_list.append(temp_ds)
        return sub_ds_list
    sub_ds_list = divide_ds(ds, num_workers)
    final_results = []
    results = []
    pool = Pool(processes=num_workers)
    for sub_ds in sub_ds_list:
        results.append(pool.apply_async(process_sub_dataset, (sub_ds, tensors)))
    pool.close()
    pool.join()  # 注：进程池中进程执行完毕后再关闭。
    pool.terminate()

    for result in results:
        final_results.extend(result.get())

    dump_to_path(final_results, target_path)

# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import json
import os
import re
import shutil
from multiprocessing import Process, Queue, Pool

import tantivy

from muller.constants import DEFAULT_MAX_SEARCH_LIMIT
from muller.util.exceptions import UnsupportedInvertedIndexError


def decide_tantivy_field_type(htype, dtype):
    if htype == "generic":
        if dtype == "float64":
            return "float"
        if dtype == "int64":
            return "integer"
        return None
    if htype == "class_label":
        return "integer"
    if htype == "text" or (htype == "list" and dtype == "<U0"):
        return "text"
    return None


def get_schema(htype, dtype):
    schema_builder = tantivy.SchemaBuilder()
    tantivy_filed_type = decide_tantivy_field_type(htype, dtype)
    if tantivy_filed_type == "text":
        schema_builder.add_text_field("content", stored=False)
    elif tantivy_filed_type == "integer":
        schema_builder.add_integer_field("content", stored=False, indexed=True)
    elif tantivy_filed_type == "float":
        schema_builder.add_float_field("content", stored=False, indexed=True)
    else:
        raise UnsupportedInvertedIndexError(htype, dtype)
    schema_builder.add_text_field("id", stored=True, index_option="basic")
    schema = schema_builder.build()
    return schema


def custom_tokenize(text):
    pattern = re.compile(r'([a-zA-Z]+)|([\u4e00-\u9fff])|([。，！？])') # find english words and chinese characters
    text = pattern.sub(r' \1\2\3 ', text)
    text = re.sub(r'\s+', ' ', text).strip()  # remove redundant spaces
    return text


def process_data(data):
    uuid, content = data
    if isinstance(content, list):
        content_cut = []
        for element in content:
            content_cut.append(custom_tokenize(element))
    else:
        content_cut = [custom_tokenize(content)]
    return uuid, content_cut


class InvertedIndex:
    def __init__(self, parent_path, branch_name):
        self.path = os.path.join(parent_path, "indexdir", branch_name)
        self._index_cache = {}
        self.meta_path = os.path.join(self.path, "meta.json")
        self.batch_size = 100000

    def get_indexed_tensors(self):
        json_meta = {}
        if os.path.exists(self.meta_path):
            with open(self.meta_path, 'r') as f:
                json_meta = json.load(f)
        return list(json_meta.keys())

    def search(self, tensor_name, query, htype, dtype):
        schema = get_schema(htype, dtype)
        idx = self._get_idx(tensor_name, schema)
        ids = set()
        searcher = idx.searcher()
        if htype in ('text', 'list'):
            query_parse = idx.parse_query(f"'{custom_tokenize(query)}'", ["content"])
        else:
            query_parse = idx.parse_query(f"{query}", ["content"])
        docs_hit = searcher.search(query_parse, DEFAULT_MAX_SEARCH_LIMIT).hits
        if len(docs_hit) == 0:
            return ids
        for doc_tuple in docs_hit:
            _, address = doc_tuple
            doc = searcher.doc(address)
            ids.add(doc["id"][0])
        return ids

    def execute_tantivy_index_job(self, columns, commit_id, json_meta, produce_func, consume_func):
        queue_dict, producer_dict, consumer_dict = {}, {}, {}
        for tensor_name in columns:
            json_meta[tensor_name] = commit_id  # record this tensor's last commit id
            queue_dict[tensor_name] = Queue()
            producer_dict[tensor_name] = Process(target=produce_func,
                                                 args=(queue_dict[tensor_name], tensor_name))
            consumer_dict[tensor_name] = Process(target=consume_func,
                                                 args=(queue_dict[tensor_name], tensor_name))
        try:
            for tensor_name in columns:
                producer_dict[tensor_name].start()
                consumer_dict[tensor_name].start()

            for tensor_name in columns:
                producer_dict[tensor_name].join()
                consumer_dict[tensor_name].join()
        except Exception as e:
            raise e
        finally:
            with open(self.meta_path, 'w') as f:
                json.dump(json_meta, f)

    def create_index(self, ds, columns, commit_id, procs=16, cache_size=900_000_000, overwrite=False):
        if overwrite and os.path.exists(self.path):
            shutil.rmtree(self.path)
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            json_meta = {}
        else:
            with open(self.meta_path, 'r') as f:
                json_meta = json.load(f)

        def produce(q, tensor_name):
            """Producer keeps reading data from MULLER."""
            tensor_dict = ds.tensors
            tensor_object = tensor_dict[tensor_name]

            uuids = [str(uuid) for uuid in tensor_object._sample_id_tensor.numpy().flatten()]
            search_field_type = decide_tantivy_field_type(ds.tensors[tensor_name].htype, ds.tensors[tensor_name].dtype)
            with Pool(procs) as pool:
                for i in range(0, len(uuids), self.batch_size):
                    batch_uuids = uuids[i:i+self.batch_size]
                    batch_contents = tensor_object[i:i+self.batch_size].data()["value"]
                    data_list = zip(batch_uuids, batch_contents)
                    if search_field_type == "text":
                        for result in pool.imap_unordered(process_data, data_list):
                            q.put(result)
                    else:
                        for result in data_list:
                            q.put((result[0], result[1][0]))
            q.put(None)

        def consume(q, tensor_name):
            """Consumer keeps creating index for data."""
            path_tensor_index = os.path.join(self.path, tensor_name)
            if not os.path.exists(path_tensor_index):
                os.makedirs(path_tensor_index)
            ix = tantivy.Index(get_schema(ds.tensors[tensor_name].htype, ds.tensors[tensor_name].dtype),
                               path=path_tensor_index)
            writer = ix.writer(cache_size)
            while True:
                data = q.get()
                if data is None:
                    break
                uuid, content = data
                writer.add_document(tantivy.Document(id=uuid, content=content))
            writer.commit()
            writer.wait_merging_threads()

        self.execute_tantivy_index_job(columns, commit_id, json_meta, produce, consume)

    def update_index(self, ds, diff, columns, commit_id, procs=16, cache_size=900_000_000):
        """ Update documents from the difference.
            Replace changes, or add document if no existing document matches the unique fields of
            the document that needs to be updated.
        """
        schema_dict = {}
        for tensor_name in columns:
            schema_dict[tensor_name] = get_schema(ds.tensors[tensor_name].htype, ds.tensors[tensor_name].dtype)
        with open(self.meta_path, 'r') as f:
            json_meta = json.load(f)

        def produce(q, tensor_name):
            data_list = []
            for id_add, content in diff[tensor_name]["added"].items():
                data_list.append((id_add, content))
            for id_up, content in diff[tensor_name]["updated"].items():
                data_list.append((id_up, content))
            search_field_type = decide_tantivy_field_type(ds.tensors[tensor_name].htype, ds.tensors[tensor_name].dtype)
            with Pool(procs) as pool:
                if search_field_type == "text":
                    for result in pool.imap_unordered(process_data, data_list):
                        q.put(result)
                else:
                    for result in data_list:
                        q.put(result)
            q.put(None)

        def consume(q, tensor_name):
            path_tensor_index = os.path.join(self.path, tensor_name)
            ix = tantivy.Index(schema_dict[tensor_name], path=path_tensor_index)
            writer = ix.writer(cache_size)
            for id_update, _ in diff[tensor_name]["updated"].items():
                writer.delete_documents(field_name="id", field_value=id_update)
            for id_del in diff[tensor_name]["deleted"]:
                writer.delete_documents(field_name="id", field_value=id_del)
            while True:
                data = q.get()
                if data is None:
                    break
                uuid, content = data
                writer.add_document(tantivy.Document(id=uuid, content=content))
            writer.commit()
            writer.wait_merging_threads()

        self.execute_tantivy_index_job(columns, commit_id, json_meta, produce, consume)


    def _get_idx(self, tensor_name, schema):
        try:
            self._index_cache[tensor_name] = tantivy.Index(schema, path=os.path.join(self.path, tensor_name))
        except Exception as e:
            raise Exception(f"Tensor \"{tensor_name}\" has not been indexed.") from e
        return self._index_cache[tensor_name]

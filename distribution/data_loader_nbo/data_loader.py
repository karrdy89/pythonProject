# import sys
#
# from db import DBUtil

# db = DBUtil()
# q = "CREATE TABLE TEST (" \
#     "CUST_NO VARCHAR(13) NOT NULL," \
#     "ORGN_DTM DATE NOT NULL," \
#     "EVNT_ID VARCHAR(6)," \
#     "EVNT_NM VARCHAR(9)," \
#     "SYS_EVNT_ID VARCHAR(10)," \
#     "SYS_EVNT_NM VARCHAR(20))"
# q = "DROP TABLE TEST"
# db.execute_query(q)

# import datetime
# import random
# test_data = []
# for _ in range(10000000):
#     test_data.append(("CUST" + str(_%300000).zfill(7), datetime.datetime.now(), "EVT" + str(random.randint(0, 250)).zfill(3),
#                       "T_EVNT" + str(random.randint(0, 250)).zfill(3), "C03-EVT" + str(random.randint(0, 250)).zfill(3),
#                       "T_CH" + str(random.randint(0, 50)).zfill(2) + "-T_EVNT" + str(random.randint(0, 250)).zfill(3)))
# print(test_data[:1])


# r = db.insert_many("INSERT_TEST_DATA", test_data)


import sys
import os
import json
from itertools import islice, chain
from typing import Any

import pandas as pd
import ray
from ray.util.multiprocessing import Pool

from db import DBUtil
from statics import Actors
from distribution.data_loader_nbo.utils import split_chunk, make_dataset


@ray.remote
class MakeDatasetNBO:
    def __init__(self):
        self.chunk_size: int = 0
        self.cur_buffer_size: int = 0
        self.num_chunks: int = 0
        self.count: int = 0
        self.file_count: int = 1
        self.split: list = []
        self.dataset: list = []
        self.information: list = []
        self.information_total: list = []

        self.logger: ray.actor = None
        self.shared_state: ray.actor = None
        self.file_size_limit: int = 10485760
        self.num_concurrency: int = 1
        self.labels: list | None = None
        self.key_index: int = 0
        self.x_index: list | None = None
        self.version: str = '0'
        self.dataset_name: str = "NBO"
        self.path: str = ''
        self.process_pool: Pool | None = None
        self.db: DBUtil | None = None
        self.act: ray.actor = None


        self.logger = ray.get_actor(Actors.LOGGER)
        self.shared_state = ray.get_actor(Actors.GLOBAL_STATE)

        self.file_size_limit = 10485760  # input, deal with percentage
        self.num_concurrency = 8  # deal with cpu count
        self.labels = ["EVT000", "EVT100", "EVT200", "EVT300", "EVT400", "EVT500", "EVT600", "EVT700", "EVT800",
                       "EVT900"]  # input
        self.key_index = 0  # input
        self.x_index = [1]  # input
        self.version = '0'    # input
        self.dataset_name = "NBO"   #input
        self.path = os.path.dirname(os.path.abspath(__file__))+"/dataset/"+self.dataset_name+"/"+self.version
        self.process_pool = Pool(self.num_concurrency)
        self.db = DBUtil()
        self.db.set_select_chunk(name="select_test", array_size=10000, prefetch_row=10000)


    def set_act(self, act: ray.actor, labels: list, version: str, path: str, ):
        self.act = act
        return 0

    def set_dataset(self, data: list, information: dict):
        self.dataset += data
        self.information.append(information)
        if len(self.information) == self.num_chunks:
            self.information_total += self.information
            self.export()
        return 0

    def set_split(self, data):
        self.split.append(data)
        if len(self.split) == self.num_chunks:
            self.merge()
        return 0

    def done(self):
        self.process_pool.close()
        max_len = 0
        classes = None
        for information in self.information_total:
            inform_max = information.get("max_len")
            if max_len < inform_max:
                max_len = inform_max
            if classes is not None:
                inform_classes = information.get("classes")
                for key in classes:
                    classes[key] += inform_classes[key]
            else:
                classes = information.get("classes")
        information = {"max_len": max_len, "class": classes}
        with open(self.path + "/information.json", 'w', encoding="utf-8") as f:
            json.dump(information, f, ensure_ascii=False, indent=4)

        # log to logger
        print("done")
        # request kill to global state
        ray.kill(ray.get_actor("dataset_maker"))

    def export(self):
        max_len = 0
        for info in self.information:
            if max_len < info.get("max_len"):
                max_len = info.get("max_len")
        fields = ["key"]
        for i in range(max_len):
            fields.append("feature" + str(i))
        fields.append("label")

        for data in self.dataset:
            data_len = len(data)
            r_max_len = max_len+2
            if data_len < r_max_len:
                label = data.pop(-1)
                for i in range(r_max_len - data_len):
                    data.append(None)
                data.append(label)
        try:
            df = pd.DataFrame(self.dataset, columns=fields)
            if not os.path.exists(self.path):
                os.makedirs(self.path)
            print(self.path)
            print(df.head(10))
            df.to_csv(self.path + "/" + str(self.file_count) + ".csv", sep=",", na_rep="NaN")
        except Exception as e:
            print(e)
            self.process_pool.close()
            # request kill to global state
            # kill actor
        else:
            self.information = []
            self.dataset = []
            self.file_count += 1
            self.operation_data()

    def merge(self):
        left_over = None
        split_len = len(self.split)
        for i in range(split_len):
            cur_chunk = None
            for chunk in self.split:
                if chunk[-2] == i:
                    cur_chunk = chunk[:-2]
            if left_over is not None:
                left_over_cust_id = left_over[0]
                if cur_chunk[0][0] == left_over_cust_id:
                    cur_chunk[0][1] = left_over[1] + cur_chunk[0][1]
                else:
                    cur_chunk.insert(0, left_over)
            if i < split_len - 1:
                left_over = cur_chunk.pop(-1)
            self.process_pool.apply_async(make_dataset, args=(cur_chunk, self.labels, self.act))
        self.split = []

    def fault_handle(self, msg):
        self.process_pool.close()
        # send to logger
        # request kill to global state
        # kill actor
        raise Exception(msg)

    def operation_data(self):
        self.cur_buffer_size = 0
        self.num_chunks = 0
        for i, chunk in enumerate(islice(self.db.select_chunk(), self.count, None)):
            if self.chunk_size == 0:
                self.chunk_size = sys.getsizeof(chunk) + sys.getsizeof(True)
            self.cur_buffer_size += self.chunk_size
            if self.cur_buffer_size + self.chunk_size < self.file_size_limit:
                self.process_pool.apply_async(split_chunk,
                                              args=(chunk, i, self.key_index, self.x_index, False, self.act))
            else:
                self.process_pool.apply_async(split_chunk,
                                              args=(chunk, i, self.key_index, self.x_index, True, self.act))
                self.num_chunks = i + 1
                self.count += i
                return 1
        self.done()
        return 0

#
# def split_chunk(chunk: list[tuple], chunk_index: int, key_index: int, x_index: list[int], is_buffer_end: bool):
#     print("@@@@@@@@@@@@")
#     split = []
#     temp = []
#     before_key = None
#     for data in chunk:
#         cur_key = data[key_index]
#         for i in x_index:
#             temp.append(data[i])
#         if before_key != data[key_index]:
#             split.append([cur_key, temp])
#             temp = []
#         before_key = data[key_index]
#     split.append(chunk_index)
#     split.append(is_buffer_end)
#     dataset_maker = ray.get_actor(Actors.DATA_MAKER_NBO)
#     result = ray.get(dataset_maker.set_split.remote(data=split))
#     if result != 0:
#         dataset_maker.fault_handle.remote(msg="failed to send split result")
#
#
# def make_dataset(datas: list, labels: list[str]):
#     max_len = 0
#     classes = {}
#     for label in labels:
#         classes[label] = 0
#     dataset = []
#     information = {}
#     for data in datas:
#         cust_id = data[0]
#         features = data[1]
#         matched_idx_before = 0
#         matched = False
#         for i, feature in enumerate(features):
#             if matched:
#                 matched = False
#                 matched_idx_before += 1
#                 continue
#             if feature in labels:
#                 matched_idx_current = i
#                 if matched_idx_current <= matched_idx_before + 1:
#                     matched_idx_before = matched_idx_current
#                 else:
#                     dataset.append([cust_id] + features[matched_idx_before:matched_idx_current] + [feature])
#                     if max_len < (matched_idx_current - matched_idx_before):
#                         max_len = matched_idx_current - matched_idx_before
#                     classes[feature] += 1
#                     matched_idx_before = matched_idx_current
#                 matched = True
#
#     information["max_len"] = max_len
#     information["classes"] = classes
#     dataset_maker = ray.get_actor(Actors.DATA_MAKER_NBO)
#     result = ray.get(dataset_maker.set_dataset.remote(data=dataset, information=information))
#     if result != 0:
#         dataset_maker.fault_handle.remote(msg="failed to send make dataset result")
#

# def make_nbo_dataset():
#     from ray.util import inspect_serializability
#     inspect_serializability(MakeDatasetNBO, name=Actors.DATA_MAKER_NBO)
#     inspect_serializability(split_chunk, name="split_chunk")
#     inspect_serializability(make_dataset, name="make_dataset")
#
#     svr2 = MakeDatasetNBO.options(name="dataset_maker").remote()
#     svr2.operation_data.remote()
# ray.init()
#
# from ray.util import inspect_serializability
#
# inspect_serializability(MakeDatasetNBO, name="dataset_maker")
# inspect_serializability(split_chunk, name="split_chunk")
# inspect_serializability(make_dataset, name="make_dataset")
#
#
# svr2 = MakeDatasetNBO.options(name="dataset_maker").remote()
# from timeit import default_timer as timer
#
# start = timer()
# svr2.operation_data.remote()
# end = timer()
# print(end - start)
# 20, 30sec, 21

# make dataloader -> call by endpoint
# get train data -> aibeem.dataset(datasetname, version) -> list of filepath, labels
# def iteration pipeline type

#
# from db import DBUtil
# from timeit import default_timer as timer
# db = DBUtil()
# db.set_select_chunk(name="select_test", array_size=10000, prefetch_row=10000)
# print(db.select("select_test"))
# start = timer()
# for i, c in enumerate(db.select_chunk()):
#     print(c[1])
# end = timer()
# print(end-start)
# 13sec

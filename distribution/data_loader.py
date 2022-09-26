import sys

from db import DBUtil

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

# d = None
# for _ in range(1):
#     d = {"CUSTNO": "CUST" + str(_).zfill(7),
#          "EVNT_ID": "EVT" + str(_%999).zfill(3),
#          "EVNT_NM": "T_EVNT" + str(_%999).zfill(3),
#          "SYS_EVNT_ID": "C03-EVT" + str(_%999).zfill(3),
#          "SYS_EVNT_NM": "T_CH" + str(_%99).zfill(2) + "-T_EVNT" + str(_%999).zfill(3)}
# print(d)
# from db.mapper import Mapper
# m = Mapper()
# md = db.parameter_mapping(m.get("insert_test"), d)
# print(md)

# r = db.insert_many("INSERT_TEST_DATA", test_data)
# r = db.execute_query("COMMIT")
# print(r.result())
# r = db.select("select_test")
# print(r.result())


# for _ in range(10000000):
#     d = {"CUSTNO": "CUST" + str(_).zfill(7),
#          "EVNT_ID": "EVT" + str(_%999).zfill(3),
#          "EVNT_NM": "T_EVNT" + str(_%999).zfill(3),
#          "SYS_EVNT_ID": "C03-EVT" + str(_%999).zfill(3),
#          "SYS_EVNT_NM": "T_CH" + str(_%99).zfill(2) + "-T_EVNT" + str(_%999).zfill(3)}
#     r = db.insert("insert_test", d)
#     print(str(_) + " / 9999999")

# print(test_data[9999999])
# r = db.insert_many("INSERT_TEST_DATA", test_data[:10000])
#
#
# from timeit import default_timer as timer
# start = timer()
# r = db.select("select_test")
# dt = r.result()

# db.set_select_chunk(name="select_test", array_size=1500, prefetch_row=1500)
# dt = []
# for chunk in db.select_chunk():
#     dt.append(chunk)
# end = timer()
# print(end - start)
# print(sys.getsizeof(dt))
# fetchall without tuning : take avg 35 sec for 10,000,000, 40mb
# fetchmany with tuning(as=1500, pf=1500) : take avg 7.2 sec for 10,000,000, 40mb

# 1. def fetchmany(yield) and fetch size, prefetch : done
# 2. def split size by memory and config, anyway maximum filesize is n% of memery size <-
# -> append chunk to filesize max
# -> if n(num of concurrent)/filesize exceeded, write to next buffer, next is set get and wright left over buffer and flush
# 3. distribution job
# -> yield from fetch, iteration
# -> 1. revolver
# -> 2. ray with cpu count
# -> 3. non concurrency
# 4. def iteration pipeline

# chain with pendings, if finished check pending
import concurrent.futures
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import asyncio

import pandas as pd

# it will be ray actor
# class MakeDatasetNBO:
#     def __init__(self):
#         # make some buffers of spliced chunk, if exceed max size of buffer, then write to next buffer and start merge, keep last(future)
#         # if appended something to buffer queue pop and start splice(if can atomic submit to processpool)
#         # if merge(if can atomic process task) done(add to process task), export to file except last, append last to chunking task, read from next buffer
#         # repeat
#         self.mem_limit = 10485760
#         self.num_concurrency = 8
#         self.executor = ProcessPoolExecutor(self.num_concurrency)
#         self.c_buffer_num = 2
#         self.chunk_size = 0
#         self.c_buffer_size_cur = 0
#         self.c_buffer_size_limit = self.mem_limit / self.c_buffer_num
#         self.db = DBUtil()
#         self.c_buffer_list: list[list] = [] # list of buffer queue or list(whatever atomic) if buffer in -> trigger callback
#         for i in range(self.c_buffer_num):
#             self.c_buffer_list.append([])
#
#         self.c_split = [] # list of dict, dict = {uid: data, last_flag:n}
#         self.working_buffer_idx = 0 # working buffer, writing buffer
#         self.write_buffer_idx = 0
#         self.is_first_chunk = True
#         self.split_futures = []
#         self.merge_futures = [] # list of futures
#         self.c_info = []
#         self.c_leftovers = [] # list of dict
#         self.c_datas = [] # list of dict
#
#         self.db.set_select_chunk(name="select_test", array_size=2000, prefetch_row=2000)
#
#     def operation(self, ar):
#         print(f"foo {ar}")
#         time.sleep(3)
#         print("bar")
#
#     def get_chunks(self):
#         self.executor.submit(self.operation, ar=3)
# for i, chunk in enumerate(self.db.select_chunk()):
#     if self.chunk_size == 0:
#         self.chunk_size = sys.getsizeof(chunk) + sys.getsizeof("N")
#     self.c_buffer_size_cur += self.chunk_size
#     if self.c_buffer_size_cur + self.chunk_size < self.c_buffer_size_limit:
#         self.c_buffer_list[self.write_buffer_idx].append([chunk, "N"])
#         self.split_futures.append(self.executor.submit(self.split_chunk, index=i))
#     else:
#         self.c_buffer_list[self.write_buffer_idx].append([chunk, "Y"])
#         self.executor.submit(self.split_chunk, index=i)
#         self.split_futures.append(self.executor.submit(self.split_chunk, index=i))
#
#         print("@")
#         break

# print(self.chunk_size)
# print(str(sys.getsizeof(self.c_buffer_list[self.write_buffer_idx]))+"/"+str(self.c_buffer_size))

#     # self.split_futures.append(self.executor.submit(self.split_chunk, index=i))
#     # self.split_futures[0].result()
#     # self.write_buffer_idx += 1
#     break


# t = MakeDatasetNBO()
# t.get_chunks()
# from timeit import default_timer as timer
# start = timer()
# t.get_chunks()
# end = timer()
# print(end - start)
#
#
from db import DBUtil
# db = DBUtil()
#
#
# db.set_select_chunk(name="select_test", array_size=2000, prefetch_row=2000)
# for c in db.select_chunk():
#     print(c)

import os

from multiprocessing import Pool
import ray
from ray.util.multiprocessing import Pool


@ray.remote
class MakeDatasetNBO:
    def __init__(self):
        self.file_size_limit = 10485760
        self.num_concurrency = 8
        self.chunk_size = 0
        self.cur_buffer_size = 0
        self.num_chunks = 0
        self.split = []  # list of dict, dict = {uid: data, last_flag:n}

        self.split_futures = []
        self.merge_futures = []  # list of futures
        self.c_info = []
        self.c_leftovers = []  # list of dict
        self.c_datas = []  # list of dict
        self.labels = ["EVT000", "EVT100", "EVT200", "EVT300", "EVT400", "EVT500", "EVT600", "EVT700", "EVT800",
                       "EVT900"]
        self.key_index = 0
        self.x_index = [1]
        self.process_pool = Pool(self.num_concurrency)
        self.split_result = None
        self.merge_result = None
        self.db = DBUtil()
        self.db.set_select_chunk(name="select_test", array_size=10000, prefetch_row=10000)
        self.count = 0

    def set_leftover(self, data):
        pass

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
            if i < split_len-1:
                left_over = cur_chunk.pop(-1)
            # do dataset stuff

    def set_split(self, data):
        self.split.append(data)
        if len(self.split) == self.num_chunks:
            self.merge()
        return 0

    def fault_handle(self, msg):
        raise Exception(msg)

    def operation_data(self):
        # get inspection
        # if merge

        for i, chunk in enumerate(self.db.select_chunk()):
            if self.chunk_size == 0:
                self.chunk_size = sys.getsizeof(chunk) + sys.getsizeof(True)
            self.cur_buffer_size += self.chunk_size
            if self.cur_buffer_size + self.chunk_size < self.file_size_limit:
                self.process_pool.apply_async(split_chunk,
                                              args=(chunk, i, self.key_index, self.x_index, False))
            else:
                self.process_pool.apply_async(split_chunk,
                                              args=(chunk, i, self.key_index, self.x_index, True))
                self.num_chunks = i + 1
                break


def split_chunk(chunk: list[tuple], chunk_index: int, key_index: int, x_index: list[int], is_buffer_end: bool):
    split = []
    temp = []
    before_key = None
    for data in chunk:
        cur_key = data[key_index]
        for i in x_index:
            temp.append(data[i])
        if before_key != data[key_index]:
            split.append([cur_key, temp])
            temp = []
        before_key = data[key_index]
    split.append(chunk_index)
    split.append(is_buffer_end)
    dataset_maker = ray.get_actor("dataset_maker")
    result = ray.get(dataset_maker.set_split.remote(data=split))
    if result != 0:
        dataset_maker.fault_handle.remote(msg="failed to send split result")


def merge():
    pass


from ray.util import inspect_serializability

inspect_serializability(MakeDatasetNBO, name="dataset_maker")
inspect_serializability(split_chunk, name="split_chunk")

ray.init()

svr2 = MakeDatasetNBO.options(name="dataset_maker").remote()
from timeit import default_timer as timer

start = timer()
ray.get(svr2.operation_data.remote())
end = timer()
print(end - start)
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

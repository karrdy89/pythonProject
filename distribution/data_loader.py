# import sys
#
# from db import DBUtil
#
# db = DBUtil()
# q = "CREATE TABLE TEST (" \
#     "CUST_NO VARCHAR(13) NOT NULL," \
#     "ORGN_DTM DATE NOT NULL," \
#     "EVNT_ID VARCHAR(6)," \
#     "EVNT_NM VARCHAR(9)," \
#     "SYS_EVNT_ID VARCHAR(10)," \
#     "SYS_EVNT_NM VARCHAR(20))"
# q = "DROP TABLE TEST"
# r = db.execute_query(q)
# print(r.result())

# import datetime
# test_data = []
# for _ in range(10000000):
#     test_data.append(("CUST" + str(_%1000).zfill(7), datetime.datetime.now(), "EVT" + str(_%999).zfill(3),
#                       "T_EVNT" + str(_%999).zfill(3), "C03-EVT" + str(_%999).zfill(3),
#                       "T_CH" + str(_%99).zfill(2) + "-T_EVNT" + str(_%999).zfill(3)))
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
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from db import DBUtil


#it will be ray actor
class MakeDatasetNBO:
    def __init__(self):
        # make some buffers of spliced chunk, if exceed max size of buffer, then write to next buffer and start merge, keep last(future)
        # if appended something to buffer queue pop and start splice(if can atomic submit to processpool)
        # if merge(if can atomic process task) done(add to process task), export to file except last, append last to chunking task, read from next buffer
        # repeat
        self.mem_limit = 10485760
        self.num_concurrency = 8
        self.executor = ProcessPoolExecutor(self.num_concurrency)
        self.c_buffer_num = 2
        self.chunk_size = None
        self.c_buffer_size = self.mem_limit / self.c_buffer_num
        self.db = DBUtil()
        self.c_buffer_list: list[list] = [] # list of buffer queue or list(whatever atomic) if buffer in -> trigger callback
        for i in range(self.c_buffer_num):
            self.c_buffer_list.append([])

        self.c_split = [] # list of dict, dict = {uid: data, last_flag:n}
        self.working_buffer_idx = 0 # working buffer, writing buffer
        self.write_buffer_idx = 0
        self.is_first_chunk = True
        self.merge_futures = [] # list of futures
        self.c_info = []
        self.c_leftovers = [] # list of dict
        self.c_datas = [] # list of dict

        self.db.set_select_chunk(name="select_test", array_size=1500, prefetch_row=1500)

    def get_chunks(self):
        c_buffer = self.c_buffer_list[self.write_buffer_idx]
        for chunk in self.db.select_chunk():
            # print(chunk)
            self.chunk_size = sys.getsizeof(chunk) + sys.getsizeof("Y")
            print(self.chunk_size)
            if sys.getsizeof(c_buffer) + self.chunk_size < self.c_buffer_size:
                c_buffer.append([chunk, "N"])
            else:
                c_buffer.append([chunk, "Y"])
                self.executor.submit(self.split_chunk)
                # self.write_buffer_idx += 1
                break

    def split_chunk(self):
        temp = self.c_buffer_list[self.write_buffer_idx].pop(0)
        print(temp)

    def execute_split(self, data, flag):
        pass

    def ordered_data_processing(self, job_num, flag, data):
        # convert data to dataframe
        # drop unnecessary
        # set left over to left_overs by flag
        # check
        pass

    def disordered_data_processing(self, job_num, flag, data):
        # convert data to dataframe
        # set data to c_datas : key is num and set inspect to c_info key is num
        # all task is done -> do merge task threadpool may use with amount of intersection
        # -> task_1 : {unique_1, count}, task_2 : {unique_1, count}, task_3 : {unique_1, count}
        # -> slice each data object with intersection data, save leftovers = list of dict
        # -> merge each data
        # if merge done
        # make dataset with list of label and export to file
        # this method will not work, so test ray memory limit and precess all data at once -> query always ordered
        # make some buffers of spliced chunk, if exceed max size of buffer, then write to next buffer and start merge, keep last(future)
        # if appended something to buffer queue pop and start splice(if can atomic submit to processpool)
        # if merge(if can atomic process task) done(add to process task), export to file except last, append last to chunking task, read from next buffer
        # repeat
        pass



t = MakeDatasetNBO()
t.get_chunks()

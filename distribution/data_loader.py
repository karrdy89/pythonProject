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
import pandas as pd
from db import DBUtil
from concurrent.futures import ProcessPoolExecutor

class MakeDatasetNBO:
    def __init__(self):
        self.mem_limit = 1000
        self.num_concurrency = 10
        self.executor = ProcessPoolExecutor(self.num_concurrency)
        self.c_buffer_size = self.mem_limit / self.num_concurrency
        self.db = DBUtil()
        self.c_buffer = []
        self.c_container = []
        self.c_futures = []
        self.c_info = []
        self.c_leftovers = [] # list of dict
        self.c_datas = [] # list of dict

        self.db.set_select_chunk(name="select_test", array_size=1500, prefetch_row=1500)

    def get_chunks(self):
        if len(self.c_container) <= self.num_concurrency:
            for chunk in self.db.select_chunk():
                if sys.getsizeof(self.c_buffer) < self.c_buffer_size:
                    self.c_buffer.append(chunk) # don't need to do this just call process, add current mem
                    # if chunk set call task executor
                else:
                    self.c_container.append(self.c_buffer)
                    self.c_buffer = []  # don't need to do this, just calculate current mem
                    if len(self.c_container) >= self.num_concurrency:
                        break

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
        pass




t = MakeDatasetNBO()
t.get_chunks()
print(t.c_container)
print(len(t.c_container))

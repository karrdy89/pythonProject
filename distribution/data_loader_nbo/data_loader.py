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
import configparser
import sys
import os
import json
import psutil
from itertools import islice, chain

import pandas as pd
import ray
from ray.util.multiprocessing import Pool

from db import DBUtil
from statics import Actors, ROOT_DIR
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
        self.dataset_name = "NBO"
        self.is_petch_end = False

        self.logger: ray.actor = None
        self.shared_state: ray.actor = None
        self.mem_limit: int = 10485760
        self.num_concurrency: int = 1
        self.labels: list | None = None
        self.key_index: int = 0
        self.x_index: list[int] | None = None
        self.version: str = '0'
        self.dataset_name: str = "NBO"
        self.path: str = ''
        self.process_pool: Pool | None = None
        self.db: DBUtil | None = None
        self.act: ray.actor = None




        # self.mem_limit = 10485760  # config, deal with percentage
        # self.num_concurrency = 8  # config, deal with cpu count
        # self.labels = ["EVT000", "EVT100", "EVT200", "EVT300", "EVT400", "EVT500", "EVT600", "EVT700", "EVT800",
        #                "EVT900"]  # input
        # self.key_index = 0  # input
        # self.x_index = [1]  # input
        # self.version = '0'    # input
        # self.path = os.path.dirname(os.path.abspath(__file__))+"/dataset/"+self.dataset_name+"/"+self.version
        # self.process_pool = Pool(self.num_concurrency)

    def init(self, act: ray.actor, labels: list, version: str, key_index: int, x_index: list[int]):
        self.act = act
        self.labels = labels
        self.version = version
        self.key_index = key_index
        self.x_index = x_index
        try:
            self.logger = ray.get_actor(Actors.LOGGER)
            self.shared_state = ray.get_actor(Actors.GLOBAL_STATE)
        except Exception as exc:
            print("an error occur when set actors", exc)
            return -1
        try:
            self.db = DBUtil()
            self.db.set_select_chunk(name="select_test", array_size=10000, prefetch_row=10000)
        except Exception as exc:
            print("an error occur when set DBUtil", exc)
            return -1
        config_parser = configparser.ConfigParser()
        try:
            config_parser.read("config/config.ini")
            mem_limit_percentage = int(config_parser.get("DATASET_MAKER", "MEM_LIMIT_PERCENTAGE"))
            mem_limit_percentage = mem_limit_percentage / 100
            concurrency_percentage = int(config_parser.get("DATASET_MAKER", "CONCURRENCY_PERCENTAGE_CPU"))
            concurrency_percentage = concurrency_percentage / 100
            base_path = str(config_parser.get("DATASET_MAKER", "BASE_PATH"))
        except configparser.Error as exc:
            print("an error occur when read config", exc)
            return -1
        else:
            mem_total = psutil.virtual_memory().total
            self.mem_limit = int(mem_total * mem_limit_percentage)
            cpus = psutil.cpu_count(logical=False)
            self.num_concurrency = int(cpus * concurrency_percentage)
            if self.num_concurrency < 1:
                self.num_concurrency = 1
            self.process_pool = Pool(self.num_concurrency)
            self.path = ROOT_DIR + base_path + "/" + self.dataset_name + "/" + self.version
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
            if self.cur_buffer_size + self.chunk_size < self.mem_limit:
                self.process_pool.apply_async(split_chunk,
                                              args=(chunk, i, self.key_index, self.x_index, False, self.act))
            else:
                self.process_pool.apply_async(split_chunk,
                                              args=(chunk, i, self.key_index, self.x_index, True, self.act))
                self.num_chunks = i + 1
                self.count += i
                return 1
        self.done() # if mem enough call this one first
        self.is_petch_end = True
        return 0

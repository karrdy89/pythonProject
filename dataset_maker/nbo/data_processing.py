import sys

# from db import DBUtil
#
# db = DBUtil()
# q = "CREATE TABLE TEST (" \
#     "CUST_NO VARCHAR(13) NOT NULL," \
#     "ORGN_DTM DATE NOT NULL," \
#     "EVNT_ID VARCHAR(10)," \
#     "EVNT_NM VARCHAR(9)," \
#     "SYS_EVNT_ID VARCHAR(10)," \
#     "SYS_EVNT_NM VARCHAR(20))"
# q = "DROP TABLE TEST"
# db.execute_query(q)

# import datetime
# import random
# test_data = []
# for _ in range(10000000):
#     test_data.append(("CUST" + str(_%300000).zfill(7), datetime.datetime.now(), "EVT" + str(random.randint(0, 250)).zfill(7),
#                       "T_EVNT" + str(random.randint(0, 250)).zfill(3), "C03-EVT" + str(random.randint(0, 250)).zfill(3),
#                       "T_CH" + str(random.randint(0, 50)).zfill(2) + "-T_EVNT" + str(random.randint(0, 250)).zfill(3)))
# print(test_data[:1])
# r = db.insert_many("INSERT_TEST_DATA", test_data)

import configparser
import sys
import os
import json
import psutil
import logging
from shutil import rmtree
from zipfile import ZipFile
from os.path import basename
from itertools import chain

import pandas as pd
import ray
from ray.util.multiprocessing import Pool

from db import DBUtil
from statics import Actors, ROOT_DIR, TrainStateCode
from dataset_maker.nbo.utils import split_chunk, make_dataset


@ray.remote
class MakeDatasetNBO:
    def __init__(self):
        self._worker = type(self).__name__
        self._chunk_size: int = 0
        self._num_data_limit: int | None = None
        self._num_data: int = 0
        self._cur_buffer_size: int = 0
        self._num_chunks: int = 0
        self._num_merged: int = 0
        self._file_count: int = 1
        self._split: list = []
        self._dataset: list = []
        self._information: list = []
        self._vocabs: list = []
        self._information_total: list = []
        self._dataset_name = "NBO"
        self._name = ''
        self._is_petch_end = False
        self._is_export_end = False
        self._is_operation_end = False
        self._is_merge_stop = False
        self._early_stopping = False

        self._total_processed_data = 0

        self._logger: ray.actor = None
        self._shared_state: ray.actor = None
        self._mem_limit: int = 104857600
        self._num_concurrency: int = 1
        self._labels: list | None = None
        self._key_index: int = 0
        self._x_index: list[int] | None = None
        self._version: str = '0'
        self._dataset_name: str = "NBO"
        self._path: str = ''
        self._process_pool: Pool | None = None
        self._db: DBUtil | None = None
        self._act: ray.actor = None

    def init(self, name: str, dataset_name: str, act: ray.actor, labels: list, version: str, key_index: int,
             x_index: list[int], num_data_limit: int | None, start_dtm: str, end_dtm: str):
        self._name = name
        self._dataset_name: str = dataset_name
        self._act = act
        self._labels = labels
        self._version = version
        self._key_index = key_index
        self._x_index = x_index
        self._num_data_limit = num_data_limit
        try:
            self._logger = ray.get_actor(Actors.LOGGER)
            self._shared_state = ray.get_actor(Actors.GLOBAL_STATE)
        except Exception as exc:
            print("an error occur when set actors", exc)
            return -1
        try:
            self._db = DBUtil()
            self._db.set_select_chunk(name="select_test", param={"START": start_dtm, "END": end_dtm},
                                      array_size=10000, prefetch_row=10000)
            # self._db.set_select_chunk(name="select_nbo", param={"START": start_dtm, "END": end_dtm},
            #                           array_size=10000, prefetch_row=10000)
        except Exception as exc:
            self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                    msg="an error occur when set DBUtil: " + exc.__str__())
            return -1
        config_parser = configparser.ConfigParser()
        try:
            config_parser.read("config/config.ini")
            mem_limit_percentage = int(config_parser.get("DATASET_MAKER", "MEM_LIMIT_PERCENTAGE"))
            mem_limit_percentage = mem_limit_percentage / 100
            concurrency_percentage = int(config_parser.get("DATASET_MAKER", "CONCURRENCY_PERCENTAGE_CPU"))
            concurrency_percentage = concurrency_percentage / 100
        except configparser.Error as exc:
            self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                    msg="an error occur when read config: " + exc.__str__())
            return -1
        else:
            mem_total = psutil.virtual_memory().total
            self._mem_limit = int(mem_total * mem_limit_percentage)
            cpus = psutil.cpu_count(logical=False)
            self._num_concurrency = int(cpus * concurrency_percentage)
            if self._num_concurrency < 1:
                self._num_concurrency = 1
            self._process_pool = Pool(self._num_concurrency)
            self._path = ROOT_DIR + "/dataset/" + self._dataset_name + "/" + self._version
        try:
            if os.path.exists(self._path):
                rmtree(self._path)
            os.makedirs(self._path)
        except Exception as exc:
            self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                    msg="an error occur when clean directory: " + exc.__str__())
            return -1
        return 0

    def set_dataset(self, data: list, information: dict):
        self._dataset += data
        self._information.append(information)
        processed_len = len(self._information)
        if self._early_stopping:
            self._num_merged += processed_len
            self._early_stopping = False
        if processed_len == self._num_merged:
            self._information_total += self._information
            self._export()
        return 0

    def set_split(self, data):
        self._split.append(data)
        if len(self._split) == self._num_chunks:
            self._merge()
        return 0

    def _done(self):
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="making nbo dataset: start finishing")
        self._process_pool.close()
        max_len = 0
        classes = None
        for information in self._information_total:
            inform_max = information.get("max_len")
            if max_len < inform_max:
                max_len = inform_max
            if classes is not None:
                inform_classes = information.get("classes")
                for key in classes:
                    classes[key] += inform_classes[key]
            else:
                classes = information.get("classes")
        vocabs = list(set(chain(*self._vocabs)))
        if None in vocabs:
            vocabs.remove(None)
            vocabs.append("")
        information = {"max_len": max_len, "class": classes, "vocabs": vocabs}
        try:
            with open(self._path + "/information.json", 'w', encoding="utf-8") as f:
                json.dump(information, f, ensure_ascii=False, indent=4)
        except Exception as exc:
            self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                    msg="an error occur when export information: " + exc.__str__())
            if os.path.exists(self._path):
                rmtree(self._path)
            ray.get(self._shared_state.set_make_dataset_result.remote(self._name, TrainStateCode.MAKING_DATASET_FAIL))
            self._shared_state.kill_actor.remote(self._name)
            return -1

        zip_name = self._dataset_name+'_'+self._version+".zip"
        try:
            with ZipFile(self._path+"/"+zip_name, 'w') as zipObj:
                for folderName, subfolders, filenames in os.walk(self._path):
                    for filename in filenames:
                        if filename == zip_name:
                            continue
                        else:
                            file_path = os.path.join(folderName, filename)
                            zipObj.write(file_path, basename(file_path))
        except Exception as exc:
            self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                    msg="an error occur when create zip archive: " + exc.__str__())
            if os.path.exists(self._path):
                rmtree(self._path)
            ray.get(self._shared_state.set_make_dataset_result.remote(self._name, TrainStateCode.MAKING_DATASET_FAIL))
            self._shared_state.kill_actor.remote(self._name)
            return -1

        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="making nbo dataset: finished")
        ray.get(self._shared_state.set_make_dataset_result.remote(self._name, TrainStateCode.MAKING_DATASET_DONE))
        self._shared_state.kill_actor.remote(self._name)

    def _export(self):
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="making nbo dataset: export csv")
        max_len = 0
        for info in self._information:
            if max_len < info.get("max_len"):
                max_len = info.get("max_len")
        fields = ["key"]
        for i in range(max_len):
            fields.append("feature" + str(i))
        fields.append("label")

        for data in self._dataset:
            data_len = len(data)
            r_max_len = max_len + 2
            if data_len < r_max_len:
                label = data.pop(-1)
                for i in range(r_max_len - data_len):
                    data.append(None)
                data.append(label)
        try:
            df = pd.DataFrame(self._dataset, columns=fields)
            one_column = []
            feature_df = df.iloc[:, 1:-1]
            for k in list(feature_df.keys()):
                one_column.append(feature_df[k])
            combined = pd.concat(one_column, ignore_index=True).tolist()
            self._vocabs.append(combined)
            df.to_csv(self._path + "/" + str(self._file_count) + ".csv", sep=",", na_rep="NaN", index=False)
        except Exception as exc:
            import traceback
            print(traceback.format_exc())
            self._process_pool.close()
            self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                    msg="an error occur when export csv: " + exc.__str__())
            ray.get(self._shared_state.set_make_dataset_result.remote(self._name, TrainStateCode.MAKING_DATASET_FAIL))
            self._shared_state.kill_actor.remote(self._name)
        else:
            self._information = []
            self._dataset = []
            self._file_count += 1
            self._is_export_end = True
            self.fetch_data()

    def _merge(self):
        if self._is_merge_stop:
            return
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="making nbo dataset: merging chunks")
        self._num_merged = 0
        left_over = None
        split_len = len(self._split)
        for i in range(split_len):
            cur_chunk = None
            for chunk in self._split:
                if chunk[-1] == i:
                    cur_chunk = chunk[:-1]
            if left_over is not None:
                left_over_cust_id = left_over[0]
                if cur_chunk[0][0] == left_over_cust_id:
                    cur_chunk[0][1] = left_over[1] + cur_chunk[0][1]
                else:
                    cur_chunk.insert(0, left_over)
            if i < split_len - 1:
                left_over = cur_chunk.pop(-1)
            len_chunk = len(cur_chunk)
            if self._num_data_limit is not None:
                diff_num = self._num_data_limit - self._num_data
                if len_chunk > diff_num:
                    cur_chunk = cur_chunk[0:diff_num]
                    self._num_merged = i
                    self._process_pool.apply_async(make_dataset, args=(cur_chunk, self._labels, self._act))
                    self._is_merge_stop = True
                    self._is_operation_end = True
                    self._early_stopping = True
                    self._is_petch_end = True
                    self._total_processed_data += len(cur_chunk)
                    break
            self._num_data += len_chunk
            self._num_merged = i
            self._process_pool.apply_async(make_dataset, args=(cur_chunk, self._labels, self._act))
            self._total_processed_data += len(cur_chunk)
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="total processed data: " + str(self._total_processed_data))
        self._split = []

    def fault_handle(self, msg):
        self._process_pool.close()
        self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                msg="making nbo dataset: an error occur when processing data: " + msg)
        ray.get(self._shared_state.set_make_dataset_result.remote(self._name, TrainStateCode.MAKING_DATASET_FAIL))
        self._shared_state.kill_actor.remote(self._name)

    def kill_process(self) -> int:
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="cancel make nbo dataset: run")
        try:
            self._process_pool.close()
            ray.get(self._shared_state.set_make_dataset_result.remote(self._name, TrainStateCode.MAKING_DATASET_FAIL))
            self._shared_state.kill_actor.remote(self._name)
        except Exception as exc:
            self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                    msg="cancel make nbo dataset: fail: " + exc.__str__())
            return -1
        else:
            return 0

    def fetch_data(self):
        self._cur_buffer_size = 0
        self._num_chunks = 0
        if not self._is_petch_end:
            self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                    msg="making nbo dataset: fetch data from db")
            for i, chunk in enumerate(self._db.select_chunk()):
                if len(chunk) == 0:
                    self._is_petch_end = True
                    break
                if self._chunk_size == 0:
                    self._chunk_size = sys.getsizeof(chunk) + sys.getsizeof(True)
                    if self._chunk_size >= self._mem_limit:
                        self.fault_handle("the chunk size of data exceed memory limit")
                        return -1
                self._cur_buffer_size += self._chunk_size
                if self._cur_buffer_size + self._chunk_size < self._mem_limit:
                    self._process_pool.apply_async(split_chunk,
                                                   args=(chunk, i, self._key_index, self._x_index, self._act))
                    self._num_chunks = i + 1
                else:
                    self._process_pool.apply_async(split_chunk,
                                                   args=(chunk, i, self._key_index, self._x_index, self._act))
                    self._num_chunks = i + 1
                    return 1
        if self._is_export_end and self._is_operation_end:
            self._done()
        self._is_operation_end = True
        return 0

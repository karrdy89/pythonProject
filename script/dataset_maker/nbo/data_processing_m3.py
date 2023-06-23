import configparser
from datetime import datetime
from datetime import timedelta
import sys
import os
import json
import psutil
import logging
from shutil import rmtree
from zipfile import ZipFile
from os.path import basename
from itertools import chain
from multiprocessing import Lock

import pandas as pd
import ray
from ray.util.multiprocessing import Pool

from db import DBUtil
from statics import Actors, ROOT_DIR, TrainStateCode
from dataset_maker.nbo.utils_m3 import split_chunk
from dataset_maker.arg_types import BasicTableType


@ray.remote
class MakeDatasetNBO:
    def __init__(self):
        self._worker = type(self).__name__
        self._chunk_size: int = 0
        self._num_data_limit: int | None = None
        self._num_data: int = 0
        self._cur_buffer_size: int = 0
        self._num_chunks: int = 0
        self._file_count: int = 1
        self._len_limit: int = 50
        self._label_data_total = {}
        self._label_data_total_c = {}
        self._max_len = 0
        self._dataset: list = []
        self._vocabs: list = []
        self._name = ''
        self._is_fetch_end = False
        self._total_processed_data = 0
        self._logger: ray.actor = None
        self._shared_state: ray.actor = None
        self._mem_limit: int = 104857600
        self._num_concurrency: int = 1
        self._labels: list | None = None
        self._query: str | None = None
        self._key_index: int = 0
        self._x_index: list[int] | None = None
        self._condition = None
        self._condition_index = None
        self._version: str = '0'
        self._dataset_name: str = "NBO"
        self._path: str = ''
        self._process_pool: Pool | None = None
        self._db: DBUtil | None = None
        self._act: ray.actor = None
        self._user_id: str = ''
        self._lock = Lock()
        self._total_read = 0
        self._is_early_stop = False
        self._is_mem_limit_fetch_end = False
        self._is_left_over_done = False
        self._left_over = {}
        self._mem_flush_left_over = None
        self._call_count_set_dataset = 0
        self._labels_ratio = {}

    def init(self, args: BasicTableType):
        self._name = args.actor_name
        self._dataset_name = args.dataset_name
        self._act = args.actor_handle
        self._labels = args.labels + ["UNK"]
        for label in self._labels:
            self._label_data_total[label] = 0
            self._label_data_total_c[label] = 0
        self._version = args.version
        self._key_index = args.key_index
        self._x_index = args.feature_index
        self._num_data_limit = args.num_data_limit
        self._query = args.query_name
        self._user_id = args.user_id
        self._condition_index = 1
        if self._num_data_limit is not None:
            total_label_num = len(self._labels)
            data_per_label = int(self._num_data_limit/total_label_num)
            diff = self._num_data_limit - data_per_label * total_label_num
            for label in self._labels:
                if label == "UNK":
                    self._labels_ratio[label] = data_per_label + diff
                else:
                    self._labels_ratio[label] = data_per_label
        try:
            self._logger = ray.get_actor(Actors.LOGGER)
            self._shared_state = ray.get_actor(Actors.GLOBAL_STATE)
        except Exception as exc:
            print("an error occur when set actors", exc)
            return -1
        try:
            self._db = DBUtil(db_info="FDS_DB")
            start_date_m3 = datetime.strptime(args.start_dtm, "%Y%m%d")
            self._condition = start_date_m3
            start_date = (start_date_m3 - timedelta(days=90)).strftime("%Y%m%d")
            self._db.set_select_chunk(name=self._query, param={"START": start_date, "END": args.end_dtm},
                                      array_size=50000, prefetch_row=50000)
        except Exception as exc:
            self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                    msg="an error occur when set DBUtil: " + exc.__str__())
            return -1
        config_parser = configparser.ConfigParser()
        try:
            config_parser.read(ROOT_DIR + "/config/config.ini")
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

    def get_committable(self, classes: dict):
        self._lock.acquire()
        committable = {}
        for label in self._labels:
            committable[label] = 0
        for label in self._labels:
            diff = self._labels_ratio[label] - self._label_data_total_c[label]
            if diff > 0:
                is_filled = False
                if classes[label] >= diff:
                    committable[label] = diff
                    self._label_data_total_c[label] += diff
                else:
                    committable[label] = classes[label]
                    self._label_data_total_c[label] += classes[label]
        self._lock.release()
        return committable

    def set_dataset(self, data: dict = None, information: dict = None, nc: bool = False):
        self._lock.acquire()
        if not nc:
            self._call_count_set_dataset += 1
        if information is not None:
            for label, num in information.items():
                if data is not None:
                    self._label_data_total[label] += len(data[label])
                    self._dataset += data[label]
        for label in self._labels:
            if self._label_data_total[label] < self._labels_ratio[label]:
                self._is_early_stop = False
                break
        else:
            self._is_early_stop = True
        self._lock.release()
        if self._is_early_stop:
            if self._call_count_set_dataset == self._num_chunks:
                self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                        msg="making nbo dataset: early stopping....")
                if self._is_early_stop:
                    self._export()
                    return 0
        if self._is_mem_limit_fetch_end or self._is_fetch_end:
            if self._call_count_set_dataset == self._num_chunks:
                if not self._is_left_over_done:
                    self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                            msg="making nbo dataset: process left over....")
                    self._process_left_over()
                else:
                    self._export()
        return 0

    def set_left_over(self, data: dict):
        self._lock.acquire()
        self._left_over.update(data)
        self._lock.release()
        return 0

    def _process_left_over(self):
        if self._left_over:
            chunk_idx_list = list(self._left_over.keys())
            chunk_idx_list.sort()
            b_cust_id = None
            left_over_data = []
            if self._mem_flush_left_over is not None:
                if self._left_over[0][0] == self._mem_flush_left_over[0]:
                    self._left_over[0] = self._mem_flush_left_over + self._left_over[0][1:]
                else:
                    left_over_data.append(self._mem_flush_left_over)
            for chunk_idx in chunk_idx_list:
                for chunk in self._left_over[chunk_idx]:
                    if chunk[0] == b_cust_id:
                        left_over_data[-1] = left_over_data[-1] + chunk[1:]
                    else:
                        left_over_data.append(chunk)
                    b_cust_id = chunk[0]
            if self._is_mem_limit_fetch_end:
                self._mem_flush_left_over = left_over_data.pop(-1)
            self._call_count_set_dataset = 0
            self._num_chunks = 1
            self._is_left_over_done = True
            self._process_pool.apply_async(split_chunk,
                                           args=(left_over_data, 0, self._key_index, self._x_index,
                                                 self._condition_index, self._condition,
                                                 self._labels, self._len_limit, False, self._act))
        else:
            self._is_left_over_done = True
            self.set_dataset(nc=True)

    def _done(self):
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="making nbo dataset: done: start")
        self._process_pool.close()

        vocabs = list(set(chain(*self._vocabs)))
        classes = {}
        for k, v in self._label_data_total.items():
            label_num = v
            classes[k] = label_num
            self._total_processed_data += label_num
        if None in vocabs:
            vocabs.remove(None)
            vocabs.append("")
        information = {"max_len": self._max_len, "class": classes, "vocabs": vocabs}
        try:
            with open(self._path + "/information.json", 'w', encoding="utf-8") as f:
                json.dump(information, f, ensure_ascii=False, indent=4)
        except Exception as exc:
            if os.path.exists(self._path):
                rmtree(self._path)
            self.fault_handle(msg="an error occur when export information: " + exc.__str__())
            return -1

        zip_name = self._dataset_name + '_' + self._version + ".zip"
        try:
            with ZipFile(self._path + "/" + zip_name, 'w') as zipObj:
                for folderName, subfolders, filenames in os.walk(self._path):
                    for filename in filenames:
                        if filename == zip_name:
                            continue
                        else:
                            file_path = os.path.join(folderName, filename)
                            zipObj.write(file_path, basename(file_path))
        except Exception as exc:
            if os.path.exists(self._path):
                rmtree(self._path)
            self.fault_handle(msg="an error occur when create zip archive: " + exc.__str__())
            return -1

        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="making nbo dataset: done: finished | total read: " + str(self._total_read)
                                + " | total processed: " + str(self._total_processed_data))
        self._shared_state.set_make_dataset_result.remote(self._name, self._user_id, TrainStateCode.MAKING_DATASET_DONE)
        self._shared_state.kill_actor.remote(self._name)

    def _export(self):
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="making nbo dataset: export csv: start")
        self._is_left_over_done = False
        for data in self._dataset:
            dt_len = len(data) - 2
            if self._max_len < dt_len:
                self._max_len = dt_len

        fields = ["key"]
        for i in range(self._max_len):
            fields.append("feature" + str(i))
        fields.append("label")
        for data in self._dataset:
            data_len = len(data)
            r_max_len = self._max_len + 2
            if data_len < r_max_len:
                label = data.pop(-1)
                for i in range(r_max_len - data_len):
                    data.append(None)
                data.append(label)
        try:
            mdf_flag = False
            df = pd.DataFrame(self._dataset, columns=fields)
            self._dataset = []
            one_column = []
            feature_df = df.iloc[:, 1:-1]
            for k in list(feature_df.keys()):
                one_column.append(feature_df[k])
            if len(one_column) > 1:
                combined = pd.concat(one_column, ignore_index=True).tolist()
                self._vocabs.append(combined)
                df.to_csv(self._path + "/" + str(self._file_count) + ".csv", sep=",", na_rep="NaN", index=False)
            else:
                mdf_flag = True
        except Exception as exc:
            self.fault_handle(msg="an error occur when export csv: " + exc.__str__())
        else:
            if mdf_flag:
                self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                        msg="making nbo dataset: export csv: no concatenable data frame")
                self.fault_handle(msg="an error occur when export csv: " + "no concatenable data frame")
            if self._is_early_stop or self._is_fetch_end:
                self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                        msg="making nbo dataset: export csv: finish")
                self._done()
            elif self._is_mem_limit_fetch_end:
                self._file_count += 1
                self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                        msg="making nbo dataset: export csv: finish")
                self.fetch_data()

    def fault_handle(self, msg):
        self._process_pool.close()
        self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                msg="making nbo dataset: an error occur when processing data: " + msg)
        self._shared_state.set_make_dataset_result.remote(self._name, self._user_id, TrainStateCode.MAKING_DATASET_FAIL)
        self._shared_state.set_error_message.remote(self._name, msg)
        self._shared_state.kill_actor.remote(self._name)

    def kill_process(self) -> int:
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="cancel make nbo dataset: run")
        try:
            self._process_pool.close()
            self._shared_state.set_make_dataset_result.remote(self._name, self._user_id,
                                                              TrainStateCode.MAKING_DATASET_FAIL)
            self._shared_state.kill_actor.remote(self._name)
        except Exception as exc:
            self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                    msg="cancel make nbo dataset: fail: " + exc.__str__())
            return -1
        else:
            return 0

    def fetch_data(self):
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="making nbo dataset: fetch data: start")
        if not self._is_fetch_end and not self._is_early_stop:
            self._call_count_set_dataset = 0
            self._cur_buffer_size = 0
            self._num_chunks = 0
            self._is_mem_limit_fetch_end = False
            self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                    msg="making nbo dataset: fetch data: read from db")
            i = 0
            for chunk in self._db.select_chunk():
                if self._is_early_stop:
                    return 0
                self._total_read += len(chunk)
                print("read: ", self._total_read)
                if len(chunk) == 0:
                    self._is_fetch_end = True
                    print("end of table")
                    break
                if self._chunk_size == 0:
                    self._chunk_size = sys.getsizeof(chunk) + sys.getsizeof(True)
                    if self._chunk_size >= self._mem_limit:
                        self.fault_handle("the chunk size of data exceed memory limit")
                        break
                self._cur_buffer_size += self._chunk_size
                self._num_chunks = i + 1
                if self._cur_buffer_size + self._chunk_size < self._mem_limit:
                    self._process_pool.apply_async(split_chunk,
                                                   args=(chunk, i, self._key_index, self._x_index,
                                                         self._condition_index, self._condition,
                                                         self._labels, self._len_limit, False, self._act))
                else:
                    self._is_mem_limit_fetch_end = True
                    self._process_pool.apply_async(split_chunk,
                                                   args=(chunk, i, self._key_index, self._x_index,
                                                         self._condition_index, self._condition,
                                                         self._labels, self._len_limit, False, self._act))
                    self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                            msg="making nbo dataset: fetch data: ended by limitation of memory")
                    break
                i += 1
        elif self._is_fetch_end and self._total_read == 0:
            self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                    msg="making nbo dataset: fetch data: empty table")
            self._process_pool.close()
            self._shared_state.set_make_dataset_result.remote(self._name, self._user_id,
                                                              TrainStateCode.MAKING_DATASET_FAIL)
            self._shared_state.kill_actor.remote(self._name)

        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="making nbo dataset: fetch data: end")
        return 0

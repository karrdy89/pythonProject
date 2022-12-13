import configparser
import math
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
from dataset_maker.nbo.utils import split_chunk, make_dataset
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
        self._split: list = []
        self._label_data_total = {}
        self._max_len = 0
        self._dataset: list = []
        self._vocabs: list = []
        self._name = ''
        self._is_f_end = False
        self._is_fetch_end = False
        self._is_export_end = False
        self._is_operation_end = False
        self._total_processed_data = 0

        self._logger: ray.actor = None
        self._shared_state: ray.actor = None
        self._mem_limit: int = 104857600
        self._num_concurrency: int = 1
        self._labels: list | None = None
        self._query: str | None = None
        self._key_index: int = 0
        self._x_index: list[int] | None = None
        self._version: str = '0'
        self._dataset_name: str = "NBO"
        self._path: str = ''
        self._process_pool: Pool | None = None
        self._db: DBUtil | None = None
        self._act: ray.actor = None
        self._user_id: str = ''
        self._lock = Lock()
        self._flush = 100
        self._total_read = 0
        self._is_fetch_data_end = False
        self._is_data_limit = False
        self._is_merge = False
        self._is_export = False
        self._call_count_set_split = 0
        self._call_count_set_dataset = 0
        self._labels_ratio = {}
        self._cur_labels_num = {}

    def init(self, args: BasicTableType):
        self._name = args.actor_name
        self._dataset_name = args.dataset_name
        self._act = args.actor_handle
        self._labels = args.labels + ["UNK"]
        for label in self._labels:
            self._label_data_total[label] = 0
            self._cur_labels_num[label] = 0
        self._version = args.version
        self._key_index = args.key_index
        self._x_index = args.feature_index
        self._num_data_limit = args.num_data_limit
        self._query = args.query_name
        self._user_id = args.user_id
        if self._num_data_limit is not None:
            total_label_num = len(self._labels)
            data_per_label = int(self._num_data_limit/total_label_num)
            diff = self._num_data_limit - data_per_label * total_label_num
            for label in self._labels:
                if label == "UNK":
                    self._labels_ratio[label] = data_per_label + diff
                else:
                    self._labels_ratio[label] = data_per_label
        digit = (int(math.log10(self._num_data_limit)) + 1)
        self._flush = int(math.e ** digit * int(str(self._num_data_limit)[:1]))
        try:
            self._logger = ray.get_actor(Actors.LOGGER)
            self._shared_state = ray.get_actor(Actors.GLOBAL_STATE)
        except Exception as exc:
            print("an error occur when set actors", exc)
            return -1
        try:
            self._db = DBUtil(db_info="MANAGE_DB")
            self._db.set_select_chunk(name=self._query, param={"START": args.start_dtm, "END": args.end_dtm},
                                      array_size=10000, prefetch_row=10000)
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

    def is_committable(self, labels_num: dict) -> list:
        if self._is_operation_end:
            return []
        appendable_labels = []
        if self._labels_ratio:
            self._lock.acquire()
            for labels, limit in self._labels_ratio.items():
                af_num = self._cur_labels_num[labels] + labels_num[labels]
                if af_num <= limit:
                    self._cur_labels_num[labels] = af_num
                    appendable_labels.append(labels)
            if not appendable_labels:
                self._is_operation_end = True
        self._lock.release()
        return appendable_labels

    def set_dataset(self, data: dict = None, information: dict = None, f_end: bool = False):
        self._lock.acquire()
        if self._is_export or self._is_f_end:
            self._lock.release()
            return 0
        self._is_f_end = f_end
        self._call_count_set_dataset += 1

        for label, num in information["classes"].items():
            if self._label_data_total[label] < self._labels_ratio[label]:
                ovf = self._label_data_total[label] + num - self._labels_ratio[label]
                if ovf > 0:
                    diff = self._labels_ratio[label] - self._label_data_total[label]
                    self._label_data_total[label] += len(data[label][:diff])
                    self._dataset += data[label][:diff]
                else:
                    self._label_data_total[label] += len(data[label])
                    self._dataset += data[label]

        if f_end:
            self._split = []
            self._is_merge = False
            self._is_export = True
            self._lock.release()
            self._export()
            return 0

        if self._call_count_set_dataset == len(self._split):
            self._split = []
            self._is_merge = False
            self._is_export = True
            self._lock.release()
            self._export()
            return 0
        self._lock.release()
        return 0

    def set_split(self, data=None, nc=False):
        self._lock.acquire(block=True, timeout=5)
        if self._is_merge:
            self._lock.release()
            return 0
        if not nc:
            self._call_count_set_split += 1
        if data is not None:
            self._split.append(data)
        if self._is_fetch_data_end:
            if self._call_count_set_split == self._num_chunks:
                self._is_merge = True
                self._lock.release()
                self._merge()
                return 0
        self._lock.release()
        return 0

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
            df = pd.DataFrame(self._dataset, columns=fields)
            one_column = []
            feature_df = df.iloc[:, 1:-1]
            for k in list(feature_df.keys()):
                one_column.append(feature_df[k])
            combined = pd.concat(one_column, ignore_index=True).tolist()
            self._vocabs.append(combined)
            df.to_csv(self._path + "/" + str(self._file_count) + ".csv", sep=",", na_rep="NaN", index=False)
        except Exception as exc:
            self.fault_handle(msg="an error occur when export csv: " + exc.__str__())
        else:
            self._information = []
            self._dataset = []
            self._file_count += 1
            self._is_export_end = True
            self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                    msg="making nbo dataset: export csv: finish")
            self.fetch_data()

    def _merge(self):
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="making nbo dataset: merge: start")
        self._call_count_set_dataset = 0
        left_over = None
        split_len = len(self._split)
        for i in range(split_len):
            if self._is_operation_end:
                self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                        msg="making nbo dataset: merge: end by early stop")
                return 0
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
            self._num_data += len_chunk
            self._process_pool.apply_async(make_dataset, args=(cur_chunk, self._labels, self._len_limit,
                                                               self._labels_ratio, self._is_operation_end, self._act))
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="making nbo dataset: merge: end")
        return 0

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
        self._is_export = False
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="making nbo dataset: fetch data: start")
        if not self._is_fetch_end and not self._is_operation_end:
            self._call_count_set_split = 0
            self._cur_buffer_size = 0
            self._num_chunks = 0
            self._is_fetch_data_end = False
            self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                    msg="making nbo dataset: fetch data: read from db")
            i = 0
            for chunk in self._db.select_chunk():
                if self._is_operation_end:
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
                        return -1
                self._cur_buffer_size += self._chunk_size
                self._num_chunks = i + 1
                if self._cur_buffer_size + self._chunk_size < self._mem_limit:
                    self._process_pool.apply_async(split_chunk,
                                                   args=(chunk, i, self._key_index, self._x_index, self._act))
                    if self._num_chunks >= self._flush:
                        self._is_fetch_data_end = True
                        self.set_split(nc=True)
                        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                                msg="making nbo dataset: fetch data: ended by flush")
                        return 1
                else:
                    self._process_pool.apply_async(split_chunk,
                                                   args=(chunk, i, self._key_index, self._x_index, self._act))
                    self._is_fetch_data_end = True
                    self.set_split(nc=True)
                    self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                            msg="making nbo dataset: fetch data: ended by limitation of memory")
                    return 1
                i += 1
            self._is_fetch_data_end = True
            self.set_split(nc=True)
        if (self._is_export_end and self._is_operation_end) or (self._is_export_end and self._is_fetch_end):
            self._done()

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

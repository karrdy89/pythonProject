# *********************************************************************************************************************
# Program Name : db_util
# Creator : yum kiyeon
# Create Date : 2022. 11. 10
# Modify Desc :
# *********************************************************************************************************************
# ---------------------------------------------------------------------------------------------------------------------
# Date  | Updator   | Remark
#
# ---------------------------------------------------------------------------------------------------------------------
import re
import concurrent.futures
import configparser
from typing import Optional, Any

import oracledb

from db.mapper import Mapper
from statics import ROOT_DIR
from cryptography.fernet import Fernet


class DBUtil:
    """
    A DB utility for oracle

    Attributes
    ----------
    _worker: str
        The class name of instance.
    _mapper: Mapper
        The Mapper class for mapping query.
    _session_pool
        The oracle connection pool.
    _chunk_cursor:
        The cursor object for petch chunk data.
    _chunk_conn:
        The oracle connection for petch chunk data.
    _dsn: str
        The dsn for oracle access information.
    concurrency: bool
        if true query will execute on ThreadPoolExecutor and return futures.
    _executor: ThreadPoolExecutor
        The ThreadPoolExecutor for executing query parallely
    _USER: str
        The username for oracle access information.
    _PASSWORD: str
        The password for oracle access information.
    _IP: str
        THE IP  for oracle access information.
    _PORT: str
        THE PORT for oracle access information.
    _SID: str
        The SID for oracle access information.
    _MAX_WORKER: int
        The max worker of ThreadPoolExecutor
    _SESSION_POOL_MIN: int
        The range of session pool
    _SESSION_POOL_MAX: int
        The range of session pool
    """
    def __init__(self, db_info: str, concurrency: bool = False):
        """
        init class attribute
        :param concurrency:
            if true query will execute on ThreadPoolExecutor and return futures
        """
        self._worker = type(self).__name__
        self._mapper = Mapper()
        self._session_pool = None
        self._chunk_cursor = None
        self._chunk_conn = None
        self._dsn: str = ''
        self.concurrency = concurrency
        if self.concurrency:
            from concurrent.futures import ThreadPoolExecutor
            self._executor: ThreadPoolExecutor | None = None
        self._USER: str = ''
        self._PASSWORD: str = ''
        self._IP: str = ''
        self._PORT: str = ''
        self._SID: str = ''
        self._MAX_WORKER: int = 5
        self._SESSION_POOL_MIN: int = 2
        self._SESSION_POOL_MAX: int = 30
        self._DSN = "None"
        oracledb.init_oracle_client()
        try:
            self.init(db_info=db_info)
        except Exception as exc:
            raise exc

    def init(self, db_info: str) -> None:
        """
        Initiate DBUtil from config
        :rtype: None
        """
        config_parser = configparser.ConfigParser()
        try:
            config_parser.read(ROOT_DIR + "/config/config.ini")
            self._USER = str(config_parser.get(db_info, "USER"))
            self._PASSWORD = str(config_parser.get(db_info, "PASSWORD"))
            key_path = ROOT_DIR + "/script/db/refKey.txt"
            f = open(key_path, "rb")
            key = f.read()
            f.close()
            self._PASSWORD = bytes(self._PASSWORD, 'utf-8')
            dec_key = Fernet(key)
            self._PASSWORD = (dec_key.decrypt(self._PASSWORD)).decode("utf-8")
            self._IP = str(config_parser.get(db_info, "IP"))
            self._PORT = int(config_parser.get(db_info, "PORT"))
            self._SID = str(config_parser.get(db_info, "SID"))
            self._MAX_WORKER = int(config_parser.get(db_info, "MAX_WORKER"))
            self._SESSION_POOL_MIN = int(config_parser.get(db_info, "SESSION_POOL_MIN"))
            self._SESSION_POOL_MAX = int(config_parser.get(db_info, "SESSION_POOL_MAX"))
            self._DSN = str(config_parser.get(db_info, "DSN"))
        except configparser.Error as exc:
            raise exc
        print(f"DB_ACCESS_INFO: USER: {self._USER}, SID: {self._SID}, IP: {self._IP}, PORT: {self._PORT}")
        if self.concurrency:
            from concurrent.futures import ThreadPoolExecutor
            self._executor = ThreadPoolExecutor(max_workers=self._MAX_WORKER)
        if self._DSN == "None":
            self._dsn = oracledb.makedsn(host=self._IP, port=self._PORT, sid=self._SID)
        else:
            self._dsn = self._DSN
        self._session_pool = oracledb.SessionPool(user=self._USER, password=self._PASSWORD, dsn=self._dsn,
                                                  min=self._SESSION_POOL_MIN, max=self._SESSION_POOL_MAX,
                                                  increment=1, encoding="UTF-8")

    def connection_test(self) -> None:
        """
        Check DB connection
        :rtype: None
        """
        try:
            test_connection = self._session_pool.acquire()
            self._session_pool.release(test_connection)
        except Exception as exc:
            raise exc

    def select(self, name: str, param: Optional[dict] = None) -> concurrent.futures.Future | Any:
        """
        Mapping query with given name and execute
        :param name: str
            A name of query defined in query.yaml
        :param param: param
            A params be mapped in query.yaml
        :return:  concurrent.futures.Future | Any
            if self.concurrency is true, then it will return future. if not return execute result
        """
        query = self._mapper.get(name)
        if param is not None:
            query = self._parameter_mapping(query, param)
        if self.concurrency:
            return self._executor.submit(self._execute_select, query)
        else:
            return self._execute_select(query)

    def _execute_select(self, query: str):
        """
        Execute select with given query
        :return: object
        """
        print(query)
        with self._session_pool.acquire() as conn:
            conn.autocommit = True
            cursor = conn.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            cursor.close()
            return result

    def set_select_chunk(self, name: str, param: Optional[dict] = None,
                         prefetch_row: Optional[int] = None, array_size: Optional[int] = None) -> None:
        """
        Set attributes of cursor for petch chunk data.
        :param name: str
            A name of query defined in query.yaml
        :param param: param
            A params be mapped in query.yaml
        :param prefetch_row: int
            Set prefetch rows
        :param array_size: int
            Set prefetch array size
        """
        self._chunk_conn = self._session_pool.acquire()
        self._chunk_cursor = self._chunk_conn.cursor()
        if prefetch_row is not None:
            self._chunk_cursor.prefetchrows = prefetch_row
        if array_size is not None:
            self._chunk_cursor.arraysize = array_size
        query = self._mapper.get(name)
        if param is not None:
            query = self._parameter_mapping(query, param)
        print(query)
        self._chunk_cursor.execute(query)

    def select_chunk(self) -> list:
        """
        Execute select with given query
        :return: list
        """
        cursor = self._chunk_cursor
        while True:
            results = cursor.fetchmany(numRows=self._chunk_cursor.arraysize)
            if not results:
                cursor.close()
                self._session_pool.release(self._chunk_conn)
                self._chunk_cursor = None
                self._chunk_conn = None
                yield []
                break
            yield results

    def execute_query(self, query: str) -> concurrent.futures.Future | Any:
        if self.concurrency:
            return self._executor.submit(self._execute_query, query)
        else:
            return self._execute_query(query)

    def _execute_query(self, query: str):
        with self._session_pool.acquire() as conn:
            conn.autocommit = True
            cursor = conn.cursor()
            result = cursor.execute(query)
            result = result.fetchall()
            cursor.close()
            return result

    def insert(self, name: str, param: Optional[dict] = None) -> concurrent.futures.Future | Any:
        query = self._mapper.get(name)
        if param is not None:
            query = self._parameter_mapping(query, param)
        if self.concurrency:
            return self._executor.submit(self._execute_insert, query)
        else:
            return self._execute_insert(query)

    def _execute_insert(self, query: str):
        print(query)
        with self._session_pool.acquire() as conn:
            conn.autocommit = True
            cursor = conn.cursor()
            result = cursor.execute(query)
            if result is None:
                cursor.close()
                return result
            else:
                raise Exception("query failed")

    def insert_many(self, name: str, data: list[tuple]) -> concurrent.futures.Future | Any:
        query = self._mapper.get(name)
        if self.concurrency:
            return self._executor.submit(self._execute_many, query, data)
        else:
            return self._execute_many(query, data)

    def _execute_many(self, query: str, data: list[tuple]):
        with self._session_pool.acquire() as conn:
            conn.autocommit = True
            cursor = conn.cursor()
            result = cursor.executemany(query, data, batcherrors=True)
            for error in cursor.getbatcherrors():
                print(error.message)
            cursor.close()
            return result

    def _parameter_mapping(self, query: str, param: dict) -> str:
        """
        Mapping of params to given query.
        :param query: str
            Query string
        :param param: dict
            variable of query string
        :rtype: str
        """
        mapped = re.sub(r"\#\{(.*?)\}", lambda m: self._mapping(m.group(1), param), query)
        return mapped

    def _mapping(self, s: str, param: dict) -> str:
        """
        Mapping of params to given string.
        :param s: str
            Query string
        :param param: dict
            variable of query string
        :rtype: str
        """
        v = param[s]
        if v is None:
            return "NULL"
        elif type(v) == int or type(param[s]) == float:
            return str(v)
        elif type(v) == str:
            v = v.replace("'", '"')
            return "'" + v + "'"

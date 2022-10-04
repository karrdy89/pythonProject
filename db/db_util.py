import re
import concurrent.futures
import configparser
from typing import Optional, Any

import oracledb

from db.mapper import Mapper


class DBUtil:
    def __init__(self, concurrency: bool = False):
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
        try:
            self.init()
        except Exception as exc:
            raise exc

    def init(self):
        config_parser = configparser.ConfigParser()
        try:
            config_parser.read("config/config.ini")
            self._USER = str(config_parser.get("DB", "USER"))
            self._PASSWORD = str(config_parser.get("DB", "PASSWORD"))
            self._IP = str(config_parser.get("DB", "IP"))
            self._PORT = int(config_parser.get("DB", "PORT"))
            self._SID = str(config_parser.get("DB", "SID"))
            self._MAX_WORKER = int(config_parser.get("DB", "MAX_WORKER"))
            self._SESSION_POOL_MIN = int(config_parser.get("DB", "SESSION_POOL_MIN"))
            self._SESSION_POOL_MAX = int(config_parser.get("DB", "SESSION_POOL_MAX"))
        except configparser.Error as exc:
            raise exc
        if self.concurrency:
            from concurrent.futures import ThreadPoolExecutor
            self._executor = ThreadPoolExecutor(max_workers=self._MAX_WORKER)
        self._dsn = oracledb.makedsn(host=self._IP, port=self._PORT, sid=self._SID)
        self._session_pool = oracledb.SessionPool(user=self._USER, password=self._PASSWORD, dsn=self._dsn,
                                                  min=self._SESSION_POOL_MIN, max=self._SESSION_POOL_MAX,
                                                  increment=1, encoding="UTF-8")

    def connection_test(self):
        try:
            test_connection = self._session_pool.acquire()
            self._session_pool.release(test_connection)
        except Exception as exc:
            raise exc

    def select(self, name: str, param: Optional[dict] = None) -> concurrent.futures.Future | Any:
        query = self._mapper.get(name)
        if param is not None:
            query = self._parameter_mapping(query, param)
        if self.concurrency:
            return self._executor.submit(self._execute_select, query)
        else:
            return self._execute_select(query)

    def _execute_select(self, query: str):
        with self._session_pool.acquire() as conn:
            conn.autocommit = True
            cursor = conn.cursor()
            result = cursor.execute(query)
            result = result.fetchall()
            cursor.close()
            return result

    def set_select_chunk(self, name: str, param: Optional[dict] = None,
                         prefetch_row: Optional[int] = None, array_size: Optional[int] = None) -> None:
        self._chunk_conn = self._session_pool.acquire()
        self._chunk_cursor = self._chunk_conn.cursor()
        if prefetch_row is not None:
            self._chunk_cursor.prefetchrows = prefetch_row
        if array_size is not None:
            self._chunk_cursor.arraysize = array_size
        query = self._mapper.get(name)
        if param is not None:
            query = self._parameter_mapping(query, param)
        self._chunk_cursor.execute(query)

    def select_chunk(self) -> list:
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
        mapped = re.sub(r"\#\{(.*?)\}", lambda m: self._mapping(m.group(1), param), query)
        return mapped

    def _mapping(self, s: str, param: dict) -> str:
        v = param[s]
        if v is None:
            raise Exception("parameter isn't matched")
        elif type(v) == int or type(param[s]) == float:
            return str(v)
        elif type(v) == str:
            return "'" + v + "'"

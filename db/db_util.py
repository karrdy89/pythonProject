import re
import concurrent
import configparser
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

# import uvloop
import oracledb

from db.mapper import Mapper


class DBUtil:
    def __init__(self):
        # asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        self._worker = type(self).__name__
        self._mapper = Mapper()
        self._session_pool = None
        self._dsn: str = ''
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
        self._executor = ThreadPoolExecutor(max_workers=self._MAX_WORKER)

        self._dsn = oracledb.makedsn(host=self._IP, port=self._PORT, sid=self._SID)
        self._session_pool = oracledb.SessionPool(user=self._USER, password=self._PASSWORD, dsn=self._dsn,
                                                  min=self._SESSION_POOL_MIN, max=self._SESSION_POOL_MAX,
                                                  increment=1, encoding="UTF-8")
        try:
            test_connection = self._session_pool.acquire()
            self._session_pool.release(test_connection)
        except Exception as exc:
            raise exc

    def execute_query(self, query: str) -> concurrent.futures.Future:
        return self._executor.submit(self._execute, query)

    def _execute(self, query: str):
        with self._session_pool.acquire() as conn:
            cursor = conn.cursor()
            result = cursor.execute(query)
            if result is not None:
                result = result.fetchall()
            else:
                cursor.execute("commit")
            return result

    def select(self, name: str, param: Optional[dict] = None) -> concurrent.futures.Future:
        query = self._mapper.get(name)
        if param is not None:
            query = self._parameter_mapping(query, param)
        return self._executor.submit(self._execute_select, query)

    def _execute_select(self, query: str):
        with self._session_pool.acquire() as conn:
            cursor = conn.cursor()
            result = cursor.execute(query)
            result = result.fetchall()
            return result

    def insert(self, name: str, param: Optional[dict] = None) -> concurrent.futures.Future:
        query = self._mapper.get(name)
        if param is not None:
            query = self._parameter_mapping(query, param)
        return self._executor.submit(self._execute_insert, query)

    def _execute_insert(self, query: str):
        with self._session_pool.acquire() as conn:
            cursor = conn.cursor()
            result = cursor.execute(query)
            if result is None:
                result = cursor.execute("commit")
            else:
                raise Exception("query failed")
            return result

    def insert_many(self, name: str, data: list[tuple]) -> concurrent.futures.Future:
        query = self._mapper.get(name)
        return self._executor.submit(self._execute_many, query, data)

    def _execute_many(self, query: str, data: list[tuple]):
        with self._session_pool.acquire() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, data, batcherrors=True)
            result = cursor.execute("commit")
            return result

    def _parameter_mapping(self, query: str, param: dict) -> str:
        mapped = re.sub(r"\#\{(.*?)\}", lambda m: self._get_mapped_value(m.group(1), param), query)
        return mapped

    def _get_mapped_value(self, s: str, param: dict) -> str:
        v = param[s]
        if v is None:
            raise Exception("parameter isn't matched")
        elif type(v) == int or type(param[s]) == float:
            return str(v)
        elif type(v) == str:
            return "'"+v+"'"


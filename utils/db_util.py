import concurrent
import configparser
import datetime
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

import uvloop
import oracledb


class DBUtil:
    def __init__(self):
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        self._worker = type(self).__name__
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
        try:
            self._dsn = oracledb.makedsn(host=self._IP, port=self._PORT, sid=self._SID)
        except Exception as exc:
            raise exc
        try:
            self._session_pool = oracledb.SessionPool(user=self._USER, password=self._PASSWORD, dsn=self._dsn,
                                                      min=self._SESSION_POOL_MIN, max=self._SESSION_POOL_MAX)
        except ConnectionError as exc:
            raise exc
        self._executor = ThreadPoolExecutor(max_workers=self._MAX_WORKER)

    def execute_query(self, query: str) -> concurrent.futures:
        with self._executor as executor:
            return executor.submit(self._execute, query)

    def _execute(self, query: str):
        with self._session_pool.acquire() as conn:
            cursor = conn.cursor()
            result = cursor.execute(query).fetchall()
            time.sleep(5)
            return result


dbutil = DBUtil()
q = "select user from dual"
f = dbutil.execute_query(q)
print(f.result())
# dbutil.set_connection(host="192.168.72.128", user="system", password="oracle1234", port=1521, sid="sid")

# q = "select user from dual"
# q = "create table TEST (" \
#     "TRSCDTM TIMESTAMP NOT NULL primary key," \
#     "CUSTNO NUMBER NOT NULL," \
#     "CHNID VARCHAR2(20) NOT NULL," \
#     "SESID VARCHAR2(20) NOT NULL," \
#     "CHNEVTID VARCHAR2(20) NOT NULL," \
#     "REFKEY VARCHAR2(20)," \
#     "EVTID VARCHAR2(20) NOT NULL)"
# dbutil = DBUtil()
# dbutil.set_connection(host="192.168.72.128", user="system", password="oracle1234", port=1521, sid="sid")
# result1 = dbutil.select(q)
# result2 = dbutil.select(q)
# print(result1)
# print(result2)
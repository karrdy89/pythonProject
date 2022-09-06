import configparser
import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

import uvloop
import oracledb


class DBUtil:
    # async class -> thread pool task executor
    # put in execute query to executor
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
        except Exception as e:
            raise Exception(e)

        # self._execute_select = ExecuteSelect()
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
        except configparser.Error as e:
            raise e
        try:
            self._dsn = oracledb.makedsn(host=self._IP, port=self._PORT, sid=self._SID)
        except Exception as e:
            raise e
        try:
            self._session_pool = oracledb.SessionPool(user=self._USER, password=self._PASSWORD, dsn=self._dsn,
                                                      min=self._SESSION_POOL_MIN, max=self._SESSION_POOL_MAX)
        except ConnectionError as e:
            raise e

    def select(self, query: str):
        if self._session_pool is None:
            #log
            print("session not initiated")
            return -1
        else:
            return asyncio.get_event_loop().run_until_complete(self.__async__select(query))

    async def __async__select(self, query: str):
        # async with self._execute_select as selected:
        async with ExecuteSelect(self._session_pool, query) as selected:
            return selected

    def insert_list(self, query: str, data: list[tuple] ):
        with self._session_pool.acquire() as conn:
            cursor = conn.cursor()
            result = cursor.excutemany(query, data)
            return result.fetchall()


class ExecuteSelect:
    def __init__(self, session_pool, query: str):
        self._session_pool = session_pool
        self.query = query

    async def __aenter__(self):
        return await self.execute_select()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def execute_select(self):
        with self._session_pool.acquire() as conn:
            cursor = conn.cursor()
            result = cursor.execute(self.query).fetchall()
            await asyncio.sleep(3)
            print(datetime.datetime.now())
            return result


# dbutil = DBUtil()
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
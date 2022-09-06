import datetime
import asyncio

import uvloop
import oracledb

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


class DBUtil:
    # async class
    def __init__(self):
        self._session_pool = None
        self._dsn = None
        # self._execute_select = ExecuteSelect()

    def set_connection(self, user: str, password: str, host: str, port: int, sid: str):
        self._dsn = oracledb.makedsn(host=host, port=port, sid=sid)
        try:
            self._session_pool = oracledb.SessionPool(user=user, password=password, dsn=self._dsn, min=2, max=30)
        except ConnectionError:
            #log
            print("connection err", ConnectionError)

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
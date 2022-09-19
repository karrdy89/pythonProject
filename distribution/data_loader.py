import sys

from db import DBUtil

db = DBUtil()
# q = "CREATE TABLE TEST (" \
#     "CUST_NO VARCHAR(13) NOT NULL," \
#     "ORGN_DTM DATE NOT NULL," \
#     "EVNT_ID VARCHAR(6)," \
#     "EVNT_NM VARCHAR(9)," \
#     "SYS_EVNT_ID VARCHAR(10)," \
#     "SYS_EVNT_NM VARCHAR(20)," \
#     "CONSTRAINT TEST_PK PRIMARY KEY(CUST_NO, ORGN_DTM))"
# q = "DROP TABLE TEST"
# r = db.execute_query(q)
# print(r.result())

# test_data = []
# for _ in range(10000000):
#     test_data.append(("CUST_NO" + str(_), "SYSDATE", "EVT" + str(_%999).zfill(3),
#                       "T_EVNT" + str(_%999).zfill(3), "C03-EVT" + str(_%999).zfill(3),
#                       "T_CH" + str(_%99).zfill(2) + "-T_EVNT" + str(_%999).zfill(3)))
# for _ in range(10000000):
#     d = {"CUSTNO": "CUST" + str(_).zfill(7),
#          "EVNT_ID": "EVT" + str(_%999).zfill(3),
#          "EVNT_NM": "T_EVNT" + str(_%999).zfill(3),
#          "SYS_EVNT_ID": "C03-EVT" + str(_%999).zfill(3),
#          "SYS_EVNT_NM": "T_CH" + str(_%99).zfill(2) + "-T_EVNT" + str(_%999).zfill(3)}
#     r = db.insert("insert_test", d)
#     print(str(_) + " / 9999999")

# print(test_data[9999999])
# r = db.insert_many("INSERT_TEST_DATA", test_data[:10000])
#
#
from timeit import default_timer as timer
start = timer()
# r = db.select("select_test")
# dt = r.result()

db.set_select_chunk(name="select_test", array_size=1500, prefetch_row=1500)
dt = []
for chunk in db.select_chunk():
    dt.append(chunk)
end = timer()
print(end - start)
print(sys.getsizeof(dt))
# fetchall without tuning : take avg 35 sec for 10,000,000, 40mb
# fetchmany with tuning(as=1500, pf=1500) : take avg 7.2 sec for 10,000,000, 40mb

# 1. def fetchmany(yield) and fetch size, prefetch : done
# 2. def split size by memory and config <-
# 3. distribution job
# -> yield from fetch, iteration
# -> 1. revolver
# -> 2. ray with cpu count
# -> 3. non concurrency

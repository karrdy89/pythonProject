# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 1. read data from db
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import oracledb
import pandas as pd

# db connection info
# USER = "S_AIB_U"
# PASSWORD = "Csxdsxzes4#"
# IP = "10.10.10.42"
# PORT = 1525
# SID = "FDSD"

USER = "SYSTEM"
PASSWORD = "oracle1234"
IP = "192.168.113.1"
PORT = 3000
SID = "KY"

dsn = oracledb.makedsn(host=IP, port=PORT, sid=SID)
session_pool = oracledb.SessionPool(user=USER, password=PASSWORD, dsn=dsn,
                                    min=2, max=10,
                                    increment=1, encoding="UTF-8")

# check db connection
try:
    test_connection = session_pool.acquire()
except Exception as exc:
    print("connection fail")
    raise exc
else:
    print("connection success")
    session_pool.release(test_connection)

# read data from db
START_DATE = "20220601"
END_DATE = "20230130"
# query = "SELECT /*+ INDEX_DESC(A IXTBCHN3001H03) */ \
#                 A.CUST_NO \
#                 , A.ORGN_DTM \
#                 , A.CHNL_ID \
#                 , A.EVNT_ID \
#             FROM \
#                 S_AIB.TBCHN3001H A \
#             WHERE 1=1 \
#             AND CUST_NO IS NOT NULL \
#             AND ORGN_DTM BETWEEN " + START_DATE + " || '000000' AND " + END_DATE + " || '999999'"

query = "SELECT CUST_NO, EVNT_ID \
        FROM TEST \
        WHERE 1=1 AND ORGN_DTM BETWEEN TO_DATE("+ START_DATE +", 'YYYYMMDD') AND TO_DATE("+ END_DATE +", 'YYYYMMDD') ORDER BY CUST_NO ASC, ORGN_DTM ASC"


with session_pool.acquire() as conn:
    cursor = conn.cursor()
    cursor.execute(query)
    df = pd.DataFrame(cursor.fetchall())
    df.columns = [x[0] for x in cursor.description]
    cursor.close()

assert len(df) != 0, "No value retrieved"
print(df)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 2. transform data
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
cust_nos = df["CUST_NO"].unique().tolist()
df_sep_cust_no = []
for cust_no in cust_nos:
    df_sep_cust_no.append(df[df["CUST_NO"] == cust_no].reset_index(drop=True))



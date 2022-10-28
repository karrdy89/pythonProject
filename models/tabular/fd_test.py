# read dataset

import pandas as pd
from statics import ROOT_DIR

dataset_path = ROOT_DIR + "/dataset/fd_test/fd_dataset.csv"
df = pd.read_csv(dataset_path)
print(df.keys())
df = df[["MID", "IP", "PID", "PR", "Class"]]

# CCM -> CCW check it is good to go
df["MID"] = df["MID"].str.replace("CCW", "CCM")
df[["PID"]] = df[["PID"]].fillna(value="NA")
df["EVT"] = df.MID.str.cat(df.PID, sep='-')

# check null
assert not df.isnull().values.any()

# uniques to col
events = df["EVT"].unique().tolist()
len(events)

# sep with ip
ips = df["IP"].unique().tolist()
tdfs = []
for ip in ips:
    tdfs.append(df[df['IP'] == ip].reset_index())

login_event = "CCMLO0101"

for tdf in tdfs:
    ip = tdf["IP"].iloc[0]
    label = tdf["Class"].iloc[0]
    start_idxes = tdf.index[tdf["MID"] == login_event].tolist()
    end_idxes = tdf.index[tdf["PR"] != 0].tolist()
    for end_idx in end_idxes:
        pass
    print(start_idxes, start_idxes)

# print(df.values.tolist())


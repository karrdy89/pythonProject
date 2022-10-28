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
dfs = []
for ip in ips:
    dfs.append(df[df['IP'] == ip])

# start with login(MID) if class == 0 end with PR is not 0, else to end of ip from sep
login_event = "CCMLO0101"
for df in dfs:
    pass

# print(df.values.tolist())


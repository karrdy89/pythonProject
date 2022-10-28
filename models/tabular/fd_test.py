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
    tdfs.append(df[df['IP'] == ip].reset_index(drop=True))

login_event = "CCMLO0101"

dataset = pd.DataFrame(columns=["IP"] + events + ["label"])

for tdf in tdfs:
    ip = tdf["IP"].iloc[0]
    label = tdf["Class"].iloc[0]
    start_idxes = tdf.index[tdf["MID"] == login_event].tolist()
    end_idxes = tdf.index[tdf["PR"] != 0].tolist()
    data_row = {"IP": ip, "label": label}
    start_idx = start_idxes[0]
    for end_idx in end_idxes:
        temp_df = tdf[start_idx:end_idx+1]
        counted_items = (dict(temp_df["EVT"].value_counts()))
        data_row.update(counted_items)
        data_row = pd.DataFrame([data_row])
        dataset = pd.concat([dataset, data_row], ignore_index=True)
        print(dataset)
        break
    print(start_idxes, start_idxes)

# print(df.values.tolist())

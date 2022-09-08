# plot data distribution
# preprocess
# make dataset
# build model
# train
# check result
# add each stop to pipeline

# import pandas as pd
# import tensorflow as tf
# import matplotlib.pyplot as plt


# write model
# if distribution pipe -> store(best param) -> split train function

import pandas as pd
from tensorflow.keras.layers.experimental import preprocessing
df = pd.read_csv("dataset/CJ_train.csv")

df.head(5)
labels = df['Target'].unique()
labels = labels.tolist()
samples = {}
for i in labels:
    samples[i] = df["Target"].value_counts()[i]
min_sample = min(samples, key=samples.get)

sep_frames = []
for label in labels:
    if label is not min_sample:
        downed = df[df['Target'] == label]
        downed = downed.sample(n=samples[min_sample].item(), random_state=0)
        sep_frames.append(downed)

df = pd.concat(sep_frames, axis=0)
df.fillna('', inplace=True)

Y = df.pop("Target")
df = df.iloc[:, ::-1]
df_list = df.values.tolist()
X = []
for j in df_list:
    X.append(j)

oneCol = []
for k in df:
    oneCol.append(df[k])
combined = pd.concat(oneCol, ignore_index=True)

events = combined.unique()
le = preprocessing.StringLookup()
le.adapt(events)




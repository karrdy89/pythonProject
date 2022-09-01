# plot data distribution
# preprocess
# make dataset
# build model
# train
# check result
# add each stop to pipeline

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt



df = pd.read_csv("dataset/CJ_train.csv")
df.head(5)
u = df['Target'].unique()
u = u.tolist()
d = {}
for i in u:
    d[i] = df["Target"].value_counts()[i]

# split each unique, copy to min idx, concat


y = df.pop("Target")
df = df.iloc[:, ::-1]
df.head(5)

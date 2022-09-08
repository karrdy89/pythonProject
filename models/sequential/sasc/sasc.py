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

# split each unique, copy to min idx, concat
# use ray dataset and do sampling or sampling with skr and to file or use it or pandas -> using pandas
# get lowest -> sep col -> concat
# write model
# if distribution pipe -> store(best param) -> split train function

import pandas as pd
df = pd.read_csv("dataset/CJ_train.csv")

df.head(5)
labels = df['Target'].unique()
labels = labels.tolist()
samples = {}
for i in labels:
    samples[i] = df["Target"].value_counts()[i]
min_sample = min(samples, key=samples.get)

print(samples[2])
# split dataset with min key




y = df.pop("Target")
df = df.iloc[:, ::-1]
df.head(5)


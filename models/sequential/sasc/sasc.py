# load data
# plot data distribution
# preprocess
# make dataset
# build model
# train
# check result
# add each stop to pipeline

import pandas as pd
import tensorflow as tf

df = pd.read_csv("dataset/CJ_train.csv")
df.head(5)

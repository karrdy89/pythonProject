# *********************************************************************************************************************
# Program Name : NBO
# Creator : yum kiyeon
# Create Date : 2022. 11. 10
# Modify Desc :
# *********************************************************************************************************************
# ---------------------------------------------------------------------------------------------------------------------
# Date  | Updator   | Remark
#
# ---------------------------------------------------------------------------------------------------------------------
import json
import os
import math

from models.sequential.sasc.modules import LabelLayer
from pipeline import Input, TrainInfo, PipelineComponent
from pipeline import split_ratio
from pipeline import base_callbacks, evaluation_callback
from models.sequential.sasc import sasc
from statics import ROOT_DIR


@PipelineComponent
def train_NBO_model(train_info: Input[TrainInfo]):
    """
    A PipelineComponent train NBO model
    :param train_info: TrainInfo
        definition of train params
    :return: None
    """
    import tensorflow as tf
    import pandas as pd
    import numpy as np
    from tensorflow.keras.layers.experimental import preprocessing
    from sklearn.model_selection import train_test_split

    sp_model_info = train_info.name.split(':')
    model_name = sp_model_info[0]
    mn_version = sp_model_info[-1].split('.')[0]
    base_dataset_path = ROOT_DIR + "/dataset/" + model_name + "/" + mn_version + "/"
    datafiles = []
    information_json = None
    for folderName, subfolders, filenames in os.walk(base_dataset_path):
        for filename in filenames:
            ext = filename.split('.')[-1]
            if ext == "csv":
                datafiles.append(filename)
            if ext == "json":
                try:
                    with open(base_dataset_path + filename) as json_file:
                        information_json = json.load(json_file)
                except Exception as exc:
                    raise exc
    if len(datafiles) == 0 or information_json is None:
        raise FileNotFoundError

    max_len = information_json["max_len"] + 1
    vocabs = information_json["vocabs"]
    classes = information_json["class"]
    labels = []
    class_weight = {}
    class_total = 0
    left_over_sample = {}
    for idx, (k, v) in enumerate(classes.items()):
        labels.append(k)
        class_weight[idx] = v
        left_over_sample[k] = v
        class_total += v

    lowest_class = min(class_weight, key=class_weight.get)
    lowest_class_weight = 1

    for k, v in class_weight.items():
        if k == lowest_class:
            class_weight[k] = (1 / v) * (class_total / 2.0) * lowest_class_weight
        else:
            class_weight[k] = (1 / v) * (class_total / 2.0)
    print(class_weight)

    labels = list(set(labels))

    mask_token = vocabs.pop(vocabs.index(""))
    le = preprocessing.StringLookup()
    le.adapt(vocabs)

    vocab_size = len(le.get_vocabulary()) + 1
    num_labels = len(labels)
    model = sasc.get_model(vocab_size=vocab_size, vocabulary=vocabs, max_len=max_len, num_labels=num_labels,
                           mask_token=mask_token)

    b_steps = 0
    b_monitor = None
    min_delta = 0.01
    p_count = 5
    t_steps = None
    dataset_per_file = {}

    for e in range(1, train_info.epoch + 1):
        c_epoch = e
        steps = []
        monitors = []
        for current_file_num, filename in enumerate(datafiles):
            dataset_path = base_dataset_path + filename
            if filename not in dataset_per_file:
                df = pd.read_csv(dataset_path)
                df.fillna("", inplace=True)
                y = df[["label"]]
                df.drop(["label", "key"], axis=1, inplace=True)
                for idx, label in enumerate(labels):
                    y.loc[y["label"] == label] = idx
                y = y["label"].tolist()
                X = df.stack().groupby(level=0).apply(list).tolist()
                train_ratio, validation_ratio, test_ratio = split_ratio(train_info.data_split)
                X_train, X_valid_test, y_train, y_valid_test = train_test_split(X, y,
                                                                                test_size=validation_ratio + test_ratio,
                                                                                shuffle=True)
                X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test, y_valid_test,
                                                                    test_size=(1 - test_ratio / (
                                                                            validation_ratio + test_ratio)),
                                                                    shuffle=True)
                X_train = np.array(X_train, dtype='object')
                y_train = np.array(y_train).astype(np.int)

                X_valid = np.array(X_valid, dtype='object')
                y_valid = np.array(y_valid).astype(np.int)

                X_test = np.array(X_test, dtype='object')
                y_test = np.array(y_test).astype(np.int)

                dataset_per_file[filename] = {"X_train": X_train, "y_train": y_train,
                                              "X_valid": X_valid, "y_valid": y_valid,
                                              "X_test": X_test, "y_test": y_test}

            # else:
            #     test_dataset_X = np.concatenate((test_dataset_X, X_test))
            # else:
            #     test_dataset_y = np.concatenate((test_dataset_y, y_test))

            train_callback = base_callbacks(train_info, b_steps=b_steps,
                                            t_epoch=train_info.epoch, c_epoch=c_epoch, t_steps=t_steps,
                                            t_file_count=len(datafiles), c_file_count=current_file_num+1)
            # model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=train_info.epoch,
            #           batch_size=train_info.batch_size, callbacks=train_callback, verbose=1)
            hist = model.fit(dataset_per_file[filename]["X_train"], dataset_per_file[filename]["y_train"],
                             validation_data=(dataset_per_file[filename]["X_valid"],
                                              dataset_per_file[filename]["y_valid"]),
                             epochs=1,
                             batch_size=train_info.batch_size, callbacks=train_callback, verbose=1,
                             class_weight=class_weight)
            monitors.append(hist.history['val_loss'][0])
            c_steps = (math.ceil(len(dataset_per_file[filename]["X_train"]) / train_info.batch_size))
            steps.append(c_steps)
            b_steps += c_steps
        if len(steps) > 1:
            t_steps = sum(steps)
        if train_info.early_stop == 'Y':
            c_monitor = sum(monitors) / len(monitors)
            if b_monitor is not None:
                if b_monitor > (c_monitor - min_delta):
                    p_count -= 1
                else:
                    p_count = 5
            b_monitor = c_monitor
            if p_count <= 0:
                print("early stopping")
                break

    X_test = None
    y_test = None
    for k in dataset_per_file:
        if X_test is None:
            X_test = dataset_per_file[k]["X_test"]
            y_test = dataset_per_file[k]["y_test"]
        else:
            X_test = np.concatenate((X_test, dataset_per_file[k]["X_test"]))
            y_test = np.concatenate((y_test, dataset_per_file[k]["y_test"]))

    test_callback = evaluation_callback(train_info)
    model.evaluate(X_test, y_test, callbacks=test_callback, batch_size=train_info.batch_size)
    labeled_output = LabelLayer(labels, num_labels, name='result')(model.outputs)
    model = tf.keras.Model(model.input, labeled_output)
    model.save(train_info.save_path)

    # predict
    threshold = 0.65
    pred = []
    truth = []
    proba = []
    for idx in range(len(X_test)):
        print(idx, len(X_test))
        label, prob = model.predict(tf.constant([X_test[idx]]))
        label = label.tolist()
        prob = prob.tolist()
        if label[0] != b"UNK":
            if prob[0] <= threshold:
                p_res = b"UNK"
            else:
                p_res = label[0]
        else:
            p_res = b"UNK"
        pred.append(p_res)
        proba.append(prob[0])
        truth.append(labels[y_test[idx]])

    import pandas as pd
    df = pd.DataFrame((zip(truth, pred, proba)), columns=['truth', 'pred', 'proba'])

    label_acc = {}
    df = df.astype({'truth': 'string'})
    df['pred'] = df['pred'].str.decode("utf-8")
    df = df.astype({'pred': 'string'})
    df.to_csv("./pred.csv")
    print(df.dtypes)
    for label in labels:
        label_df = df.loc[df['truth'] == label]
        total = len(label_df)
        correct = len(label_df.query('truth == pred'))
        label_acc[label] = correct/total
    print(label_acc)
    print("acc_total", len(df.query('truth == pred'))/len(df))
    print(df.loc[df['pred'] == "UNK"])
    print(len(df.loc[df['pred'] == "UNK"]), len(df))
    # get acc per class

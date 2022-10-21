import json
import os

from models.sequential.sasc.modules import LabelLayer
from pipeline import Input, TrainInfo, PipelineComponent
from pipeline.util import split_ratio
from pipeline.callbacks import base_callbacks, evaluation_callback
from models.sequential.sasc import sasc
from statics import ROOT_DIR


@PipelineComponent
def train_NBO_model(train_info: Input[TrainInfo]):
    import tensorflow as tf
    import pandas as pd
    import numpy as np
    from tensorflow.keras.layers.experimental import preprocessing
    from tensorflow.keras import layers
    from sklearn.model_selection import train_test_split

    sp_model_info = train_info.name.split(':')
    model_name = sp_model_info[0]
    nm_version = sp_model_info[-1].split('.')[-1]
    base_dataset_path = ROOT_DIR + "/dataset/" + model_name + "/" + nm_version + "/"
    datafiles = []
    information_json = None
    for folderName, subfolders, filenames in os.walk(base_dataset_path):
        for filename in filenames:
            ext = filename.split('.')[-1]
            if ext == "csv":
                datafiles.append(filename)
            if ext == "json":
                try:
                    with open(base_dataset_path+filename) as json_file:
                        information_json = json.load(json_file)
                except Exception as exc:
                    raise exc
    if len(datafiles) == 0 or information_json is None:
        raise FileNotFoundError

    max_len = information_json["max_len"]+1
    vocabs = information_json["vocabs"]
    classes = information_json["class"]
    labels = []
    min_class_num = 0
    for k, v in classes.items():
        labels.append(k)
        if min_class_num < v:
            min_class_num = v

    mask_token = vocabs.pop(vocabs.index(""))
    le = preprocessing.StringLookup()
    le.adapt(vocabs)
    label_vocab = preprocessing.StringLookup()
    label_vocab.adapt(labels)

    vocab_size = len(le.get_vocabulary())
    num_labels = len(label_vocab.get_vocabulary())
    model = sasc.get_model(vocab_size=vocab_size, vocabulary=vocabs, max_len=max_len, num_labels=num_labels,
                           mask_token=mask_token)
    test_dataset_X = None
    test_dataset_y = None

    for current_file_num, filename in enumerate(datafiles):
        dataset_path = base_dataset_path + filename
        df = pd.read_csv(dataset_path)
        df.fillna("", inplace=True)
        y = df.pop("label")
        X = df.stack().groupby(level=0).apply(list).tolist()
        train_ratio, validation_ratio, test_ratio = split_ratio(train_info.data_split)
        X_train, X_valid_test, y_train, y_valid_test = train_test_split(X, y, test_size=validation_ratio + test_ratio,
                                                                        shuffle=True)
        X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test, y_valid_test,
                                                            test_size=(1 - test_ratio / (
                                                                        validation_ratio + test_ratio)),
                                                            shuffle=True)
        X_train = np.array(X_train, dtype='object')
        y_train = np.array(y_train).astype(str)
        y_train = label_vocab(y_train)

        X_valid = np.array(X_valid, dtype='object')
        y_valid = np.array(y_valid).astype(str)
        y_valid = label_vocab(y_valid)

        X_test = np.array(X_test, dtype='object')
        y_test = np.array(y_test).astype(str)
        y_test = label_vocab(y_test)

        if test_dataset_X is None:
            test_dataset_X = X_test
        else:
            test_dataset_X = np.concatenate((test_dataset_X, X_test))
        if test_dataset_y is None:
            test_dataset_y = y_test
        else:
            test_dataset_y = np.concatenate((test_dataset_y, y_test))

        train_callback = base_callbacks(train_info, monitor="val_loss", dataset_num=len(datafiles),
                                        cur_dataset_num=current_file_num+1)
        model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=train_info.epoch,
                  batch_size=train_info.batch_size, callbacks=train_callback, verbose=1)

    test_callback = evaluation_callback(train_info)
    model.evaluate(test_dataset_X, test_dataset_y, callbacks=test_callback, batch_size=train_info.batch_size)
    labeled_output = LabelLayer(label_vocab.get_vocabulary(), len(label_vocab.get_vocabulary()), name='result')(model.outputs)
    model = tf.keras.Model(model.input, labeled_output)
    model.save(train_info.save_path)

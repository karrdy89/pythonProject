from models.sequential.sasc.modules import LabelLayer
from pipeline import Input, Dataset, TrainInfo, PipelineComponent
from pipeline.util import split_ratio
from pipeline.callbacks import base_callbacks, evaluation_callback
from models.sequential.sasc import sasc


@PipelineComponent
def train_NBO_model(dataset: Input[Dataset], train_info: Input[TrainInfo]):
    import tensorflow as tf
    import pandas as pd
    import numpy as np
    from tensorflow.keras.layers.experimental import preprocessing
    from tensorflow.keras import layers
    from sklearn.model_selection import train_test_split

    train_info = train_info
    df = dataset.data
    labels = df['Target'].unique()
    y = df.pop("Target")
    df = df.iloc[:, ::-1]
    X = df.stack().groupby(level=0).apply(list).tolist()
    one_col = []

    for k in df:
        one_col.append(df[k])
    combined = pd.concat(one_col, ignore_index=True)

    events = combined.unique().tolist()
    mask_token = events.pop(events.index(''))
    le = preprocessing.StringLookup()
    le.adapt(events)
    label_vocab = preprocessing.IntegerLookup(dtype='int64')
    label_vocab.adapt(labels)

    train_ratio, validation_ratio, test_ratio = split_ratio(train_info.data_split)
    X_train, X_valid_test, y_train, y_valid_test = train_test_split(X, y, test_size=validation_ratio+test_ratio,
                                                                    shuffle=False)
    X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test, y_valid_test,
                                                        test_size=(1-test_ratio/(validation_ratio+test_ratio)),
                                                        shuffle=False)
    X_train = np.array(X_train, dtype='object')
    X_valid = np.array(X_valid, dtype='object')
    X_test = np.array(X_test, dtype='object')
    y_train = np.array(y_train).astype(int)
    y_train = label_vocab(y_train)
    y_valid = np.array(y_valid).astype(int)
    y_valid = label_vocab(y_valid)
    y_test = np.array(y_test).astype(int)
    y_test = label_vocab(y_test)

    vocab_size = len(le.get_vocabulary())
    max_len = len(df.columns)
    num_labels = len(label_vocab.get_vocabulary())

    model = sasc.get_model(vocab_size=vocab_size, vocabulary=events, max_len=max_len, num_labels=num_labels, mask_token=mask_token)

    train_callback = base_callbacks(train_info, monitor="val_loss")
    test_callback = evaluation_callback(train_info)

    model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=train_info.epoch,
              batch_size=train_info.batch_size, callbacks=train_callback, verbose=1)
    model.evaluate(X_test, y_test, callbacks=test_callback, batch_size=train_info.batch_size)
    labeled_output = LabelLayer(label_vocab.get_vocabulary(), len(label_vocab.get_vocabulary()), name='result')(model.outputs)
    model = tf.keras.Model(model.input, labeled_output)
    print(model.summary())
    model.save(train_info.save_path)

import numpy as np

from db import DBUtil


def transform_data(data: list, max_len: int | None = None):
    db = DBUtil(db_info="MANAGE_DB", concurrency=False)
    events = db.select(name="select_nbo_event", param={"CUST_NO": data[0]})
    # if max_len:
    #     events = np.array(events[:max_len])
    events = np.ravel(events, order="C")
    events = events.tolist()
    X_len = len(events)
    if X_len >= max_len:
        events = events[-max_len:]
    else:
        pad_size = max_len - X_len
        events = [*events, *[""] * pad_size]
    return events


def transform_data_m2(data: list, max_len: int | None = None):
    db = DBUtil(db_info="MANAGE_DB", concurrency=False)
    events = db.select(name="select_nbo_event_m2", param={"CUST_NO": data[0]})
    # if max_len:
    #     events = np.array(events[:max_len])
    events = np.ravel(events, order="C")
    events = events.tolist()
    X_len = len(events)
    if X_len >= max_len:
        events = events[-max_len:]
    else:
        pad_size = max_len - X_len
        events = [*events, *[""] * pad_size]
    return events
    #
    # db = DBUtil(db_info="MANAGE_DB", concurrency=False)
    # events = db.select(name="select_nbo_event_m2", param={"CUST_NO": data[0]})
    # if max_len:
    #     events = np.array(events[:max_len])
    # events = np.ravel(events, order="C")
    # events = events.tolist()
    # return events

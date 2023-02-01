import numpy as np

from script.db.db_util import DBUtil


def transform_data(data: list, max_len: int | None = None):
    db = DBUtil(db_info="MANAGE_DB", concurrency=False)
    events = db.select(name="select_nbo_event", param={"CUST_NO": data[0]})
    if max_len:
        events = np.array(events[:max_len])
    events = np.ravel(events, order="C")
    events = events.tolist()
    return events

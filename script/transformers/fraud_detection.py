import pandas as pd
import datetime
from datetime import timedelta

EVENT_LIST = ["EVT0000668",
              "EVT0000665",
              "EVT0000874",
              "EVT0000574",
              "EVT0000572",
              "EVT0000868",
              "EVT0000688",
              "EVT0000589",
              "EVT0000779",
              "EVT0000481",
              "EVT0000462",
              "EVT0000548",
              "EVT0000490",
              "EVT0000575",
              "EVT0000775",
              "EVT0000479",
              "EVT0000851",
              "EVT0000889",
              "EVT0000570",
              "EVT0000512",
              "EVT0000954",
              "EVT0000487",
              "EVT0000461",
              "EVT0000488",
              "EVT0000463",
              "EVT0000700",
              "EVT0000701",
              "EVT0000465",
              "EVT0000604",
              "EVT0000703",
              "EVT0000713",
              "EVT0000848",
              "EVT0000603",
              "EVT0000689",
              "EVT0000643",
              "EVT0000464",
              "EVT0000698",
              "time_diff"]

START_EVENT = ["EVT0000512", "EVT0000461", "EVT0000570", "EVT0000464"]
END_EVENT = ["EVT0000688", "EVT0000953"]

# start_events = ["CCMLO0101", "CCWLO0101", "CCMMS0101SL01", "CCWMS0101SL01", "CCWSA0101", "CCMSA0101"]
# end_event = ["CCMLN0101PC01", "CCWLN0101PC01", "CCWRD0201PC01", "CCMRD0201PC01"]


def transform_data(db, data: list) -> list:
    data = db.select(name="select_fds_event", param={"CUST_NO": data[0]})
    if len(data) == 0:
        return []
    df = pd.DataFrame(data)
    df = df.loc[::-1].reset_index(drop=True)
    df.iloc[:, 1] = pd.to_datetime(df.iloc[:, 1], format='%Y%m%d%H%M%S')

    end_idx_list = df.index[df.iloc[:, 0].isin(END_EVENT)].tolist()
    if not end_idx_list:
        return []

    s_dtm = df.iloc[0, 1].strftime('%Y%m%d')
    t_dtm = df.iloc[[end_idx_list[-1]], 1].dt.strftime('%Y%m%d').iloc[0]
    ld_evt = df[(df.iloc[:, 1] >= t_dtm)]

    start_idx_list = ld_evt.index[ld_evt.iloc[:, 0].isin(START_EVENT)].tolist()

    reset_flag = False
    if start_idx_list:
        if start_idx_list[0] >= end_idx_list[-1]:
            reset_flag = True

    if not start_idx_list or reset_flag:
        t_dtm_obj = datetime.datetime.strptime(t_dtm, '%Y%m%d')
        t_dtm_obj = t_dtm_obj - timedelta(days=1)
        if t_dtm_obj >= datetime.datetime.strptime(s_dtm, '%Y%m%d'):
            t_dtm = t_dtm_obj.strftime('%Y%m%d')
            ld_evt = df[(df.iloc[:, 1] >= t_dtm)]
            start_idx_list = ld_evt.index[ld_evt.iloc[:, 0].isin(START_EVENT)].tolist()
        else:
            return []

    if start_idx_list:
        if start_idx_list[0] >= end_idx_list[-1]:
            return []
        df = df[start_idx_list[0]:end_idx_list[-1] + 1]
        if df.empty:
            return []
        event_list = df.iloc[:, 0].tolist()
        time_diff = (df.iloc[:, 1].iloc[-1] - df.iloc[:, 1].iloc[0]).total_seconds()
        counted_item = []
        for event in EVENT_LIST:
            if event != "time_diff":
                counted_item.append(event_list.count(event))
            else:
                counted_item.append(time_diff)
        return counted_item
    else:
        return []

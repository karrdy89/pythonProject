import pandas as pd

EVENT_LIST = ["EVT0000011",
              "EVT0000012",
              "EVT0000013",
              "EVT0000014",
              "EVT0000015",
              "EVT0000016",
              "EVT0000017",
              "EVT0000018",
              "EVT0000019",
              "EVT0000020",
              "EVT0000021",
              "EVT0000022",
              "EVT0000023",
              "EVT0000024",
              "EVT0000025",
              "EVT0000026",
              "EVT0000027",
              "EVT0000028",
              "EVT0000029",
              "EVT0000030",
              "EVT0000031",
              "EVT0000032",
              "EVT0000033",
              "time_diff",
              "EVT0000034",
              "EVT0000035",
              "EVT0000036",
              "EVT0000037",
              "EVT0000038",
              "EVT0000039",
              "EVT0000040",
              "EVT0000041",
              "EVT0000042",
              "EVT0000043",
              "EVT0000044",
              "EVT0000045",
              "EVT0000046",
              "EVT0000047",
              "EVT0000048",
              "EVT0000049",
              "EVT0000050",
              "EVT0000051",
              "EVT0000052",
              "EVT0000053"]
START_EVENT = ["EVT0000005", "EVT0000006", "EVT0000009", "EVT0000010"]  # if login many times? -> search from back wihtout main
END_EVENT = ["EVT0000001", "EVT0000002", "EVT0000003", "EVT0000004"]
# event = ["CCMLN0201CH01",
#          "CCMLN0102SL02",
#          "CCMRD0206SL01",
#          "CCMFD0101RG01",
#          "CCMVI0207SL01",
#          "CCMCP0801SE01",
#          "CCMES0403",
#          "CCMSA0101SE02",
#          "CCMSA0301SE01",
#          "CCMLN0101PC01",
#          "CCMLO0101SE01",
#          "CCMRD0202CH01",
#          "CCMSA0101SE01",
#          "CCMFD0101RG02",
#          "CCMLO0201",
#          "CCMCP0801",
#          "CCMRD0102SL02",
#          "CCMCM0103SE01",
#          "CCMLN0101CH01",
#          "CCMCP0601",
#          "CCMES0201SL01",
#          "CCMSA0101SL01",
#          "CCMLO0101SE05",
#          "CCMFD0101AT01",
#          "CCMCM0101",
#          "CCMSA0101",
#          "CCMMS0101SL01",
#          "CCMMS0101",
#          "CCMLO0101",
#          "CCMLN0101CL01",
#          "CCMCP0301SE01",
#          "CCMLN0101",
#          "CCMRD0201SL01",
#          "CCMLN0201",
#          "CCMLN0101SL02",
#          "CCMCP0801SE02",
#          "CCMCP0801SE03",
#          "CCMLN0101SL03",
#          "CCMCP0801IN01",
#          "CCMLN0101SL01",
#          "CCMFD0101AT02",
#          "CCMMA0201SE01",
#          "CCMMA0101SE01",
#          "time_diff"]
# start_events = ["CCMLO0101", "CCWLO0101", "CCWSA0101", "CCMSA0101"]
# end_event = ["CCMLN0101PC01", "CCWLN0101PC01", "CCWRD0201PC01", "CCMRD0201PC01"]



def transform_data(db, data: list) -> list:
    data = db.select(name="select_fds_event", param={"CUST_NO": data[0]})
    if len(data) == 0:
        return []
    df = pd.DataFrame(data)
    start_idx_list = df.index[df.iloc[:, 0].isin(START_EVENT)].tolist()
    end_idx_list = df.index[df.iloc[:, 0].isin(END_EVENT)].tolist()
    if not start_idx_list or not end_idx_list:
        return []
    start_idx_list.reverse()
    start_idx = None
    for s_idx in start_idx_list:
        if s_idx < end_idx_list[-1]:
            start_idx = s_idx
            break
    if start_idx is None:
        return []
    df = df[start_idx:end_idx_list[-1]+1]
    event_list = df.iloc[:, 0].tolist()
    time_diff = (df.iloc[:, 1].iloc[-1] - df.iloc[:, 1].iloc[0]).total_seconds()
    event_list.append(time_diff)
    counted_item = []
    for event in EVENT_LIST:
        if event != "time_diff":
            counted_item.append(event_list.count(event))
        else:
            counted_item.append(time_diff)
    return counted_item


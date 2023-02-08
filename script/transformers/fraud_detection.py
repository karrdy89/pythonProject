# read data from db
# count each pre-defined event
# make input
import random
import pandas as pd

MAPPING_LIST = {"EVT0000100": "EVT0000012",
                "EVT0000101": "EVT0000013",
                "EVT0000102": "EVT0000014"}
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
              "EVT0000053",
              "EVT0000053"]
START_EVENT = ["EVT0000005", "EVT0000006", "EVT0000009", "EVT0000010"]  # if login many times? -> search from back wihtout main
END_EVENT = ["EVT0000001", "EVT0000002", "EVT0000003", "EVT0000004"]
# event_list = []
# start_events = ["CCMLO0101", "CCWLO0101", "CCWSA0101", "CCMSA0101"]
# end_event = ["CCMLN0101PC01", "CCWLN0101PC01", "CCWRD0201PC01", "CCMRD0201PC01"]
# 1. feature CCMLN0201CH01 (0.125)
# 2. feature CCMLN0102SL02 (0.086)
# 3. feature CCMRD0206SL01 (0.065)
# 4. feature CCMFD0101RG01 (0.051)
# 5. feature CCMVI0207SL01 (0.049)
# 6. feature CCMCP0801SE01 (0.046)
# 7. feature CCMES0403 (0.046)
# 8. feature CCMSA0101SE02 (0.036)
# 9. feature CCMSA0301SE01 (0.035)
# 10. feature CCMLN0101PC01 (0.033)
# 11. feature CCMLO0101SE01 (0.027)
# 12. feature CCMRD0202CH01 (0.027)
# 13. feature CCMSA0101SE01 (0.027)
# 14. feature CCMFD0101RG02 (0.025)
# 15. feature CCMLO0201 (0.024)
# 16. feature CCMCP0801 (0.024)
# 17. feature CCMRD0102SL02 (0.020)
# 18. feature CCMCM0103SE01 (0.018)
# 19. feature CCMLN0101CH01 (0.017)
# 20. feature CCMCP0601 (0.017)
# 21. feature CCMES0201SL01 (0.015)
# 22. feature CCMSA0101SL01 (0.015)
# 23. feature CCMLO0101SE05 (0.015)
# 24. feature CCMFD0101AT01 (0.014)
# 25. feature CCMCM0101 (0.013)
# 26. feature CCMSA0101 (0.012)
# 27. feature CCMMS0101SL01 (0.011)
# 28. feature CCMMS0101 (0.011)
# 29. feature CCMLO0101 (0.011)
# 30. feature CCMLN0101CL01 (0.009)
# 31. feature CCMCP0301SE01 (0.009)
# 32. feature time_diff (0.008)
# 33. feature CCMLN0101 (0.008)
# 34. feature CCMRD0201SL01 (0.007)
# 35. feature CCMLN0201 (0.007)
# 36. feature CCMLN0101SL02 (0.006)
# 37. feature CCMCP0801SE02 (0.005)
# 38. feature CCMCP0801SE03 (0.005)
# 39. feature CCMLN0101SL03 (0.004)
# 40. feature CCMCP0801IN01 (0.004)
# 41. feature CCMLN0101SL01 (0.004)
# 42. feature CCMFD0101AT02 (0.003)
# 43. feature CCMMA0201SE01 (0.002)
# 44. feature CCMMA0101SE01 (0.002)


def trim_end_event(data: list) -> list | None:
    event_data = data.reverse()
    for end_event in END_EVENT:
        if end_event in event_data:
            event_data = event_data[event_data.index(end_event):]
            event_data = event_data.reverse()
            return event_data
    return None

def trim_start_event(data: list) -> list | None:
    for start_event in START_EVENT:
        if start_event in data:
            return data[data.index(start_event)]
    return None

def transform_data(db, data: str = None):
    f_event_data = None
    e_event_data = None
    s_time = None
    e_time = None
    time_diff = 0
    lst_s_event_data = []
    for days_ago in range(3):
        if days_ago == 0:
            data = db.select(name="select_nbo_event", param={"CUST_NO": data[0]})
            df = pd.DataFrame(data)
        else:
            data = db.select(name="select_nbo_event", param={"CUST_NO": data[0], "DAYS_AGO": days_ago})
            df = pd.DataFrame(data)
        if data is not None and e_event_data is None:
            event_data = trim_end_event(df.iloc[:, 1].tolist())
            if event_data is not None:
                e_event_data = event_data
                event_data = trim_start_event(event_data)
                if event_data is not None:
                    f_event_data = event_data
                    break
        elif data is not None and e_event_data is not None:
            event_data = trim_start_event(df.iloc[:, 1].tolist())
            if event_data is not None:
                lst_s_event_data += event_data
                f_event_data = lst_s_event_data + event_data
                break
            else:
                lst_s_event_data += data[0]

    if f_event_data is not None:
        input_vec = []
        for event in EVENT_LIST:
            if event == "time_diff":
                input_vec.append(time_diff)
            else:
                input_vec.append(f_event_data.count(event))
        return input_vec
    else:
        return None

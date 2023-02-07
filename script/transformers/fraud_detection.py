# read data from db
# count each pre-defined event
# make input
import random

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
START_EVENT = ["EVT0000005", "EVT0000006", "EVT0000007", "EVT0000008", "EVT0000009", "EVT0000010"]
END_EVENT = ["EVT0000001", "EVT0000002", "EVT0000003", "EVT0000004"]
# event_list = []
# start_events = ["CCMLO0101", "CCWLO0101", "CCMMS0101SL01", "CCWMS0101SL01", "CCWSA0101", "CCMSA0101"]
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


def transform_data(db, data: str = None):
    data = db.select(name="select_nbo_event", param={"CUST_NO": data[0]})
    return [random.randrange(0, 5)]*46

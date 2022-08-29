def split_ratio(data_split: str):
    ratio = data_split.split(":")
    train = float(ratio[0])*0.01
    val = float(ratio[1])*0.01
    test = float(ratio[2])*0.01
    return train, val, test

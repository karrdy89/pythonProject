# *********************************************************************************************************************
# Program Name : util
# Creator : yum kiyeon
# Create Date : 2022. 11. 10
# Modify Desc :
# *********************************************************************************************************************
# ---------------------------------------------------------------------------------------------------------------------
# Date  | Updator   | Remark
#
# ---------------------------------------------------------------------------------------------------------------------
def split_ratio(data_split: str):
    """
    Return train, validation, test ratio of dataset from given string. (ex. 80:10:10)

    Parameters
    ----------
    data_split : str
        Option of dataset split ratio. (formatted string ex. 80:10:10)
    """
    ratio = data_split.split(":")
    train = float(ratio[0])*0.01
    val = float(ratio[1])*0.01
    test = float(ratio[2])*0.01
    return train, val, test

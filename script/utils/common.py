# *********************************************************************************************************************
# Program Name : common
# Creator : yum kiyeon
# Create Date : 2022. 11. 10
# Modify Desc :
# *********************************************************************************************************************
# ---------------------------------------------------------------------------------------------------------------------
# Date  | Updator   | Remark
#
# ---------------------------------------------------------------------------------------------------------------------
def version_encode(version: str) -> int:
    """
    Return encoded version of given string (ex1. 1.0 -> 101, ex2. 1.23 -> 1232)
    can't exceed decimal point over 9

    Parameters
    ----------
    version : str

    """
    sv = version.split('.')
    if len(sv[-1]) > 9:
        raise Exception("can't exceed decimal point over 9")
    else:
        encoded = sv[0] + sv[-1] + str(len(sv[-1]))
        return int(encoded)


def version_decode(version: int) -> str:
    """
    Return decoded version of given int (ex1. 101 -> 1.0, ex2. 1232 -> 1.23)

    Parameters
    ----------
    version : int

    """
    decimal = version // 10 ** 0 % 10
    decoded = str(version)[:-1]
    if decoded == '':
        return '0.0'
    return str(int(decoded) / 10**decimal)


def encode_tf_input_meta(meta: dict) -> str:
    meta = str(meta)
    meta = meta.replace(" ", '')
    meta = meta.replace("{'", "_spds__spk_").replace('}', "_spde_")\
        .replace("':", "_spv_").replace(",'", "_spk_").replace("'", "_spq_").replace(".", "_spd_")
    return meta


def decode_tf_input_meta(meta: str) -> dict:
    meta = meta.replace("_spds__spk_", "{'").replace("_spde_", '}')\
        .replace("_spv_", "':").replace("_spk_", ",'").replace("_spq_", "'")
    meta = meta.replace("_spd_", ".")
    meta = eval(meta)
    return meta

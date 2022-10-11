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

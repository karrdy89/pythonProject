def version_encode(version: str) -> int:
    sv = version.split('.')
    if len(sv[-1]) > 9:
        raise Exception("can't exceed decimal point over 9")
    else:
        encoded = sv[0] + sv[-1] + str(len(sv[-1]))
        return int(encoded)


def version_decode(version: int) -> str:
    decimal = version // 10 ** 0 % 10
    decoded = str(version)[:-1]
    decimal = len(decoded) - decimal
    if decimal > 0:
        decoded = decoded[:decimal] + "." + decoded[decimal:]
    return decoded

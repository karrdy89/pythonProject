# *********************************************************************************************************************
# Program Name : exceptions
# Creator : yum kiyeon
# Create Date : 2022. 11. 10
# Modify Desc :
# *********************************************************************************************************************
# ---------------------------------------------------------------------------------------------------------------------
# Date  | Updator   | Remark
#
# ---------------------------------------------------------------------------------------------------------------------
class DefinitionNotFoundError(Exception):
    def __init__(self, exc: str = ''):
        self.exc = exc

    def __str__(self):
        return "there is no definition in file : " + self.exc


class DefinitionNotExistError(Exception):
    def __init__(self, exc: str = ''):
        self.exc = exc

    def __str__(self):
        return "Definition not exist in definition : " + self.exc


class SetDefinitionError(Exception):
    def __init__(self, exc: str = ''):
        self.exc = exc

    def __str__(self):
        return "an error occur when set definition : " + self.exc

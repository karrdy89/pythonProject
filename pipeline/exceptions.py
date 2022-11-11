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
class PipelineNotFoundError(Exception):
    def __str__(self):
        return "there is no pipeline in pipeline definition"


class SequenceNotExistError(Exception):
    def __str__(self):
        return "Sequence not exist in pipeline definition"


class SetSequenceError(Exception):
    def __str__(self):
        return "an error occur when set sequences"

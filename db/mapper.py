import xml.etree.ElementTree as ElemTree
import os


class Mapper:
    def __init__(self):
        self._path: str = os.path.dirname(os.path.abspath(__file__)) + "/query.xml"
        self._queries = None
        self._set_queries()

    def _set_queries(self):
        queries = ElemTree.parse(self._path)
        self._queries = queries.getroot()

    def get(self, name: str) -> str:
        query_strings = self._queries.findall(".//*[@name='"+name+"']")
        if len(query_strings) > 1:
            raise MapperException("Query is duplicated")
        if len(query_strings) == 0:
            raise MapperException("Query not exist")
        else:
            return query_strings[0].text


class MapperException(Exception):
    def __init__(self, msg: str):
        self.msg = msg

    def __str__(self):
        return self.msg

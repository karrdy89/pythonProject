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
            return "query id is duplicated. check query.xml"
        if len(query_strings) == 0:
            return ""
        else:
            return query_strings[0].text

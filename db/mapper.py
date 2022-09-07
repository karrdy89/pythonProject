import xml.etree.ElementTree as ET
import os


class QueryMapper:
    def __init__(self):
        self._path: str = os.path.dirname(os.path.abspath(__file__)) + "/query.xml"
        self._queries = None
        self._set_queries()

    def _set_queries(self):
        queries = ET.parse(self._path)
        queries = queries.getroot()
        print(queries)
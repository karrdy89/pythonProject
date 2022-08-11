import aiohttp
import json


class Http:
    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *err):
        await self._session.close()
        self._session = None

    async def get(self, url):
        async with self._session.get(url) as resp:
            if resp.status == 200:
                return await resp.read()
            else:
                print("get error. request code: " + str(resp.status))
                return None

    async def post(self, url, data: json = None):
        async with self._session.post(url=url, data=data) as resp:
            if resp.status == 200:
                return await resp.read()
            else:
                print("post error. request code: " + str(resp.status))
                return None

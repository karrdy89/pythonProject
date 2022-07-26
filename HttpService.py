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
            return await resp.read()

    async def post(self, url, data: json = None):
        async with self._session.post(url=url, data=data) as resp:
            return await resp.read()

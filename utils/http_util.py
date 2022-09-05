import aiohttp


class Http:
    """
    An async class for http request

    Methods
    -------
    __aenter__():
        Set client session entering async with.
    __aexit__(*err):
        Clear client session when exit async with
    get(url) -> int | str | None:
        update training progress to global data store when epoch end.
    post_json(url, data: dict = None) -> int | str | None:
        update training progress to global data store when batch end.
    def post(url, payload, header: dict) -> int | str | None:
    """
    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *err):
        await self._session.close()
        self._session = None

    async def get(self, url) -> int | str | None:
        try:
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    return await resp.text()
                else:
                    print("get error. request code: " + str(resp.status))
                    return None
        except aiohttp.ClientConnectionError as e:
            print("connection error")
            print(e)
            return -1

    async def post_json(self, url, data: dict = None) -> int | str | None:
        try:
            async with self._session.post(url=url, json=data, timeout=60) as resp:
                if resp.status == 200:
                    return await resp.text()
                else:
                    print("post error. request code: " + str(resp.status))
                    return None
        except aiohttp.ClientConnectionError as e:
            print("connection error")
            print(e)
            return -1

    async def post(self, url, payload, header: dict) -> int | str | None:
        try:
            async with self._session.post(url=url, data=payload, header=header, timeout=60) as resp:
                if resp.status == 200:
                    return await resp.text()
                else:
                    print("post error. request code: " + str(resp.status))
                    return None
        except aiohttp.ClientConnectionError as e:
            print("connection error")
            print(e)
            return -1

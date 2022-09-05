import aiohttp


class Http:
    """
    An async class for http request

    Methods
    -------
    __aenter__():
        Constructs all the necessary attributes.
    __aexit__(*err):
        update training progress to global data store when epoch begin.
    get(url) -> None:
        update training progress to global data store when epoch end.
    post_json(self, url, data: dict = None) -> None:
        update training progress to global data store when batch end.
    def post(self, url, payload, header: dict):
    """
    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *err):
        await self._session.close()
        self._session = None

    async def get(self, url):
        async with self._session.get(url) as resp:
            print(await resp.text())
            if resp.status == 200:
                print(await resp.text())
                return None
            else:
                print("get error. request code: " + str(resp.status))
                return None

    async def post_json(self, url, data: dict = None):
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

    async def post(self, url, payload, header: dict):
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

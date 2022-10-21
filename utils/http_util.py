import aiohttp
import ray
import logging
import json


class Http:
    """
    An async class for http request

    Methods
    -------
    __aenter__():
        Set client session entering async with.
    __aexit__(*err):
        Clear client session when exit async with.
    get(url) -> int | str | None:
        Send GET request to given url .
    post_json(url, data: dict = None) -> int | str | None:
        Send POST request to given url with json data.
    def post(url, payload, header: dict) -> int | str | None:
        Send get request to given url with data
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
                    return json.loads(await resp.text())
                else:
                    logger = ray.get_actor("logging_service")
                    logger.log.remote(level=logging.ERROR, worker=type(self).__name__,
                                      msg="http request error :" + str(resp.status))
                    return None
        except aiohttp.ClientConnectionError as e:
            logger = ray.get_actor("logging_service")
            logger.log.remote(level=logging.ERROR, worker=type(self).__name__,
                              msg="http connection error :" + str(e))
            return -1

    async def post_json(self, url, data: dict = None) -> int | str | dict | None:
        try:
            async with self._session.post(url=url, json=data, timeout=60) as resp:
                if resp.status == 200:
                    return json.loads(await resp.text())
                else:
                    print(await resp.text())
                    logger = ray.get_actor("logging_service")
                    logger.log.remote(level=logging.ERROR, worker=type(self).__name__,
                                      msg="http request error :" + str(resp.status))
                    return None
        except aiohttp.ClientConnectionError as e:
            logger = ray.get_actor("logging_service")
            logger.log.remote(level=logging.ERROR, worker=type(self).__name__,
                              msg="http connection error :" + str(e))
            return -1

    async def post(self, url, payload, header: dict) -> int | str | None:
        try:
            async with self._session.post(url=url, data=payload, header=header, timeout=60) as resp:
                if resp.status == 200:
                    return await resp.text()
                else:
                    logger = ray.get_actor("logging_service")
                    logger.log.remote(level=logging.ERROR, worker=type(self).__name__,
                                      msg="http request error :" + str(resp.status))
                    return None
        except aiohttp.ClientConnectionError as e:
            logger = ray.get_actor("logging_service")
            logger.log.remote(level=logging.ERROR, worker=type(self).__name__,
                              msg="http connection error :" + str(e))
            return -1

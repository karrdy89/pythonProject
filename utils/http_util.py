import aiohttp
import json


class Http:
    """
    An async class to monitor training progress.

    Attributes
    ----------
    _shared_state : actor
        an actor handle of global data store.
    _train_result : TrainResult
        a current result of training.
    name : str
        a name of pipeline.
    epoch_step : int
        a batch of each epoch.
    epoch : int
        current epoch of training.

    Methods
    -------
    __init__(name: str):
        Constructs all the necessary attributes.
    on_epoch_begin(epoch, logs=None) -> None:
        update training progress to global data store when epoch begin.
    on_epoch_end(epoch, logs=None) -> None:
        update training progress to global data store when epoch end.
    on_batch_end(batch, logs=None) -> None:
        update training progress to global data store when batch end.
    """
    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *err):
        await self._session.close()
        self._session = None

    async def get(self, url):
        async with self._session.get(url) as resp:
            print(await resp.read())
            if resp.status == 200:
                print(await resp.read())
                return None
                # return await resp.read()
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

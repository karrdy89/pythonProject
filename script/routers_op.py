# *********************************************************************************************************************
# Program Name : routers
# Creator : yum kiyeon
# Create Date : 2022. 11. 10
# Modify Desc :
# *********************************************************************************************************************
# ---------------------------------------------------------------------------------------------------------------------
# Date  | Updator   | Remark
#
# ---------------------------------------------------------------------------------------------------------------------
import ast
import os
import logging
import requests
import json

import ray
import httpx
from fastapi import FastAPI
from fastapi_utils.cbv import cbv
from fastapi_utils.inferring_router import InferringRouter
from fastapi.responses import FileResponse, PlainTextResponse
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import StreamingResponse
from starlette.background import BackgroundTask

from statics import Actors

project_path = os.path.dirname(os.path.abspath(__file__))
app_op = FastAPI()
app_op.add_middleware(HTTPSRedirectMiddleware)
app_op.add_middleware(CORSMiddleware)
router_op = InferringRouter()


@cbv(router_op)
class AIbeemRouter:
    """
    A class that runs and manages the tensorboard service.

    Attributes
    ----------
    _worker : str
        Class name of instance.
    _logger : actor
        An actor handle of global logger.
    _shared_state : actor
        An actor handle of global state manager.

    Methods
    -------
    __init__():
        Constructs all the necessary attributes.
    @router:
        API endpoints of server
    """

    def __init__(self):
        self._worker = type(self).__name__
        self._logger: ray.actor = ray.get_actor(Actors.LOGGER)
        self._shared_state: ray.actor = ray.get_actor(Actors.GLOBAL_STATE)

    @router_op.get("/dataset/{uid}")
    async def download_dataset(self, uid: str):
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="get request: download_dataset")
        path = await self._shared_state.get_dataset_path.remote(uid=uid)
        if path is not None:
            if os.path.exists(path):
                filename = path.split("/")[-1]
                return FileResponse(path, filename=filename)
            else:
                return "file not exist"
        else:
            return "invalid url"


async def _reverse_proxy(request: Request):
    path = request.url.path.split('/')
    port = path[2]
    path = '/'.join(path[3:])

    # validate session request to manage server
    shared_state = ray.get_actor(Actors.GLOBAL_STATE)
    session_info, url = await shared_state.get_session.remote(port=port)
    if session_info is None:
        return PlainTextResponse("invalid access")
    else:
        data = {"SESSION_ID": session_info}
        headers = {'Content-Type': 'application/json; charset=utf-8'}
        try:
            res = requests.post(url, data=json.dumps(data), headers=headers, timeout=10)
        except Exception as exc:
            return PlainTextResponse("failed to validate session:" + exc.__str__())
        else:
            if res.status_code == 200:
                body = ast.literal_eval(res.content.decode('utf-8'))
                is_valid = body.get("IS_VALID_SESSION_ID")
                if is_valid == 'Y':
                    client = httpx.AsyncClient(base_url="http://127.0.0.1:" + port)
                    url = httpx.URL(path=path,
                                    query=request.url.query.encode("utf-8"))
                    rp_req = client.build_request(request.method, url,
                                                  headers=request.headers.raw,
                                                  content=await request.body())
                    try:
                        rp_resp = await client.send(rp_req, stream=True)
                    except httpx.RequestError:
                        return PlainTextResponse("invalid URL")
                    except Exception as exc:
                        return f"An error occurred while requesting {exc.__str__()!r}."
                    else:
                        return StreamingResponse(
                            rp_resp.aiter_raw(),
                            status_code=rp_resp.status_code,
                            headers=rp_resp.headers,
                            background=BackgroundTask(rp_resp.aclose),
                        )
                else:
                    return PlainTextResponse("invalid access")
            else:
                return PlainTextResponse("failed to validate session")


app_op.add_route("/dashboard/{port}/{path:path}", _reverse_proxy, ["GET", "POST"])
app_op.add_route("/tensorboard/{port}/{path:path}", _reverse_proxy, ["GET", "POST"])
app_op.include_router(router_op)

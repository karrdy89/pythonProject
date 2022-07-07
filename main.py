import asyncio

import ray
import time
import json
from ray import serve
from fastapi import FastAPI, Request
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

import VO.trainVO

ray.init()
serve.start(detached=True, http_options={
    "host": "0.0.0.0",
    "port": 8080,
    "middlewares": [Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])]
})

app = FastAPI()


#health check API

@serve.deployment(route_prefix="/")
@serve.ingress(app)
class EndPointService:
    def __init__(self):
        self.count = 0

    @app.post("/train/data_info", reponse_model=VO.trainVO.DataInfoVO)
    async def data_info(self, request_data=VO.trainVO.DataInfoVO):
        # result_json = {}
        # request_body = await request.json()
        # if "table_arr" in request_body:
        #     table_arr = request_body["table_arr"]
        # else:
        #     return result_json
        # # await self.take_time()
        return request_data

    async def take_time(self):
        for i in range(10):
            await asyncio.sleep(1)
            print(i)


EndPointService.deploy()
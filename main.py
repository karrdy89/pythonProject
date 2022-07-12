import asyncio
import datetime

import ray
import time
import json

import uvicorn
from ray import serve
from fastapi import FastAPI, Request
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware

import VO.trainVO

# ray.init()
# serve.start(detached=True, http_options={
#     "host": "0.0.0.0",
#     "port": 8080,
#     "middlewares": [Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]),
#                     Middleware(HTTPSRedirectMiddleware)]
# })

app = FastAPI()
app.add_middleware(HTTPSRedirectMiddleware)


@app.get("/test")
async def test():
    await take_time()
    return datetime.datetime.now()


async def take_time():
    for i in range(10):
        await asyncio.sleep(1)
        print(i)

#
# @serve.deployment(route_prefix="/")
# @serve.ingress(app)
# class EndPointService:
#     def __init__(self):
#         self.count = 0
#     #
#     # @app.post("/train/data_info", reponse_model=VO.trainVO.DataInfoVO)
#     # async def data_info(self, request_data=VO.trainVO.DataInfoVO):
#     #     # result_json = {}
#     #     # request_body = await request.json()
#     #     # if "table_arr" in request_body:
#     #     #     table_arr = request_body["table_arr"]
#     #     # else:
#     #     #     return result_json
#     #     # # await self.take_time()
#     #     return request_data
#
#     @app.get("/test")
#     async def test(self):
#         # await self.take_time()
#         return datetime.datetime.now()
#
#     async def take_time(self):
#         for i in range(10):
#             await asyncio.sleep(1)
#             print(i)


# EndPointService.deploy()


if __name__ == '__main__':
    uvicorn.run("main:app",
                host="0.0.0.0",
                port=8080,
                ssl_keyfile="/home/ky/cert/key.pem",
                ssl_certfile="/home/ky/cert/cert.pem",
                ssl_keyfile_password="1234"
                )
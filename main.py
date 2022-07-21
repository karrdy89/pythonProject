import asyncio
import datetime
import json

import ray
import time
import json

import uvicorn
import multiprocessing
from ray import serve
from fastapi import FastAPI, Request
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware

from ModelServing import ModelServing

import VO.trainVO as VO

#REQUIRMENT
#ray fastapi uvicorn[standard] ray[serve]

# ray.init(ignore_reinit_error=True)
# ray.init(dashboard_host="0.0.0.0", dashboard_port=8265, include_dashboard=True)

app = FastAPI()
app.add_middleware(HTTPSRedirectMiddleware)


@app.get("/test")
async def test():
    await take_time()
    return datetime.datetime.now()


@app.get("/models")
async def get_models():
    server = ModelServing()
    return json.dumps(server.get_container_names())


@app.post("/deploy")
async def deploy(request_body: VO.DeployVO):
    # request_body = request_body.dict()
    # model_name = request_body.get("model_name")
    server = ModelServing()
    server.run_container(request_body.model_name)
    return datetime.datetime.now()


@app.post("/predict")
async def deploy(request_body: VO.PredictVO):
    result = 0
    model_name = request_body.model_name
    feature = request_body.feature
    server = ModelServing()
    result = await server.get_model_state(model_name)
    return result


async def take_time():
    for i in range(5):
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





if __name__ == "__main__":
    uvicorn.run("main:app",
                host="0.0.0.0",
                port=8080,
                ssl_keyfile="/home/ky/cert/key.pem",
                ssl_certfile="/home/ky/cert/cert.pem",
                ssl_keyfile_password="1234"
                )
#
# @ray.remote
# def printsome():
#     print("11")
#
# ray.get(printsome.remote())
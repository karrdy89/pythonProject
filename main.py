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

# REQUIRMENT
# ray fastapi uvicorn[standard] ray[serve]


app = FastAPI()
app.add_middleware(HTTPSRedirectMiddleware)


@app.get("/test")
async def test():
    await take_time()
    ray.get(print_current_datetime.remote())
    return datetime.datetime.now()


@app.get("/models")
async def get_models():
    server = ModelServing()
    return json.dumps(await server.get_container_names())


@app.get("/{model}/state")
async def get_model_state(model: str):
    server = ModelServing()
    result = await server.get_model_state(model)
    result = json.loads(result)
    return result


@app.post("/deploy")
async def deploy(request_body: VO.DeployVO):
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


async def run_uvicorn():
    #change path when deploy
    config = uvicorn.Config("main:app", host="0.0.0.0", port=8080
                            , ssl_keyfile="/home/ky/cert/key.pem", ssl_certfile="/home/ky/cert/cert.pem"
                            , ssl_keyfile_password="1234"
                            )
    server = uvicorn.Server(config)
    await server.serve()


@ray.remote
def uvicorn_wrapper():
    import asyncio
    asyncio.get_event_loop().run_until_complete(run_uvicorn())


@ray.remote
def print_current_datetime():
    time.sleep(0.3)
    current_datetime = datetime.datetime.now()
    print(current_datetime)
    return current_datetime


if __name__ == "__main__":
    ray.init(dashboard_host="0.0.0.0", dashboard_port=8265)
    ray.get(uvicorn_wrapper.remote())

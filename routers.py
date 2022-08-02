import datetime
import asyncio
import json

import ray
import httpx
from fastapi import FastAPI
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.requests import Request
from starlette.responses import StreamingResponse
from starlette.background import BackgroundTask

import VO.value_object as VO
from serving import ModelServing

app = FastAPI()
app.add_middleware(HTTPSRedirectMiddleware)


async def _reverse_proxy(request: Request):
    path = request.url.path.split('/')
    port = path[2]
    path = '/'.join(path[3:])
    client = httpx.AsyncClient(base_url="http://127.0.0.1:" + port)
    url = httpx.URL(path=path,
                    query=request.url.query.encode("utf-8"))
    rp_req = client.build_request(request.method, url,
                                  headers=request.headers.raw,
                                  content=await request.body())
    try:
        rp_resp = await client.send(rp_req, stream=True)
    except httpx.RequestError as exc:
        print(f"An error occurred while requesting {exc.request.url!r}.")
        return f"An error occurred while requesting {exc.request.url!r}."
    else:
        return StreamingResponse(
            rp_resp.aiter_raw(),
            status_code=rp_resp.status_code,
            headers=rp_resp.headers,
            background=BackgroundTask(rp_resp.aclose),
        )

app.add_route("/dashboard/{port}/{path:path}", _reverse_proxy, ["GET", "POST"])


@app.get("/ttt")
async def f():
    # app.add_route("/{path:path}", _reverse_proxy(url="aa"), ["GET", "POST"])
    await some_task.remote()
    return "Hello from the root!"


@app.get("/test")
async def test():
    await take_time()
    return datetime.datetime.now()


@app.get("/models")
async def get_models():
    server = ModelServing()
    return json.dumps(server.get_container_names())


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


@ray.remote
def some_task():
    return 1


async def take_time():
    for i in range(5):
        await asyncio.sleep(1)
        print(i)

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
server_test = ray.get_actor("model_serving")


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


@app.get("/reset")
async def reset():
    server_test.reset_version_config.remote()
    return 0


@app.get("/models")
async def get_models():
    result = await server_test.get_container_names.remote()
    return json.dumps(result)


@app.post("/deploy")
async def deploy(request_body: VO.DeployVO):
    server = ray.get_actor("model_serving")
    remote_job_obj = server.deploy.remote(model_id=request_body.model_id,
                                          model_version=request_body.model_version,
                                          container_num=request_body.container_num)
    result = await remote_job_obj
    return result


@app.get("/deploy/state")
async def get_deploy_state():
    return None


@app.post("/deploy/add_container")
async def add_container(request_body: VO.AddContainerVo):
    return None


@app.post("/deploy/delete_container")
async def delete_container(request_body: VO.DeleteContainerVO):
    return None


@app.post("/deploy/end_deploy")
async def end_deploy(request_body: VO.EndDeploy):
    return None


@app.post("/predict")
async def deploy(request_body: VO.PredictVO):
    return None

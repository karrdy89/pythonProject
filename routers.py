import os

import httpx
from fastapi import FastAPI
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.requests import Request
from starlette.responses import StreamingResponse
from starlette.background import BackgroundTask

import VO.request_vo as rvo
from tensorboard_service import TensorBoardTool
from utils.common import version_decode, version_encode

project_path = os.path.dirname(os.path.abspath(__file__))
app = FastAPI()
app.add_middleware(HTTPSRedirectMiddleware)
tensorboard_tool = TensorBoardTool()
server = None


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
app.add_route("/tensorboard/{port}/{path:path}", _reverse_proxy, ["GET", "POST"])


@app.post("/deploy")
async def deploy(request_body: rvo.Deploy):
    print("accept deploy request")
    remote_job_obj = server.deploy.remote(model_id=request_body.model_id,
                                          version=request_body.version,
                                          container_num=request_body.container_num)
    result = await remote_job_obj
    return result


@app.get("/deploy/state")
async def get_deploy_state():
    remote_job_obj = server.get_deploy_state.remote()
    result = await remote_job_obj
    return result


@app.post("/deploy/add_container")
async def add_container(request_body: rvo.AddContainer):
    remote_job_obj = server.add_container.remote(model_id=request_body.model_id, version=request_body.version,
                                                 container_num=request_body.container_num)
    result = await remote_job_obj
    return result


@app.post("/deploy/remove_container")
async def remove_container(request_body: rvo.RemoveContainer):
    remote_job_obj = server.remove_container.remote(model_id=request_body.model_id, version=request_body.version,
                                                    container_num=request_body.container_num)
    result = await remote_job_obj
    return result


@app.post("/deploy/end_deploy")
async def end_deploy(request_body: rvo.EndDeploy):
    remote_job_obj = server.end_deploy.remote(model_id=request_body.model_id, version=request_body.version)
    result = await remote_job_obj
    return result


@app.post("/predict")
async def deploy(request_body: rvo.Predict):
    remote_job_obj = server.predict.remote(model_id=request_body.model_id, version=request_body.version,
                                           data=request_body.feature)
    result = await remote_job_obj
    return result


@app.post("/tensorboard/create")
async def create_tensorboard(request_body: rvo.CreateTensorboard):
    version = request_body.version
    encoded_version = version_encode(version)
    model = request_body.model_id
    log_path = project_path + "/train_logs/" + model + "/" + str(encoded_version)
    port = tensorboard_tool.run(log_path)
    url = "/tensorboard/"+str(port)
    return url

import asyncio
import datetime
import json
import sys
import logging

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

import routers
from serving import ModelServing
from logger import Logger
from shared_state import SharedState

SSL_CERT_PATH = "/home/ky/cert" # from config

ray.init(dashboard_host="0.0.0.0", dashboard_port=8265)


@ray.remote
class UvicornServer(uvicorn.Server):
    def install_signal_handlers(self):
        pass

    async def run_server(self):
        await self.serve()


config = uvicorn.Config("routers:app",
                        host="0.0.0.0",
                        port=8080,
                        ssl_keyfile=SSL_CERT_PATH + "/key.pem",
                        ssl_certfile=SSL_CERT_PATH + "/cert.pem",
                        ssl_keyfile_password="1234"
                        )

# create service actor
logging_service = Logger.options(name="logging_service", max_concurrency=500).remote()
api_service = UvicornServer.options(name="API_service").remote(config=config)
# model_serving = ModelServing.options(name="model_serving").remote()
# shared_state = SharedState.options(name="shared_state").remote()
#
# # initiate all service
# init_processes = ray.get([model_serving.init.remote()])
# api_service.run_server.remote()
# if -1 in init_processes:
#     logging_service.log.remote(level=logging.ERROR, worker=__name__, msg="failed to initiate server. shut down")
#     ray.kill(api_service)
#     ray.kill(model_serving)
#     ray.kill(logging_service)
#     ray.kill(shared_state)
#     sys.exit()

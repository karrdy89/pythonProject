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

SSL_CERT_PATH = "/home/ky/cert"

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

API_service = UvicornServer.options(name="API_service").remote(config=config)
API_service.run_server.remote()

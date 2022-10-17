import sys
import os
import configparser
import traceback

import ray
import uvicorn

from serving import ModelServing
from db import DBUtil
from logger import Logger, BootLogger
from shared_state import SharedState
from statics import Actors

boot_logger = BootLogger().logger
boot_logger.info("(Main Server) init main server...")

SSL_CERT_PATH = ''
config_parser = configparser.ConfigParser()
try:
    config_parser.read("config/config.ini")
    SSL_CERT_PATH = os.path.dirname(os.path.abspath(__file__)) + str(config_parser.get("DEFAULT", "SSL_CERT_PATH"))
except configparser.Error as e:
    boot_logger.error("(Main Server) an error occur when set config...: " + str(e))
    sys.exit()

boot_logger.info("(Main Server) init ray...")
os.environ["RAY_LOG_TO_STDERR"] = "1"
# os.environ["RAY_LOG_TO_STDERR"] = "0"
ray.init(dashboard_host="127.0.0.1", dashboard_port=8265)


@ray.remote
class UvicornServer(uvicorn.Server):
    """
    A ray actor class for wrapping uvicorn server

    Methods
    -------
    install_signal_handlers():
        pass
    run_server() -> int
        run uvicorn server
    """
    def install_signal_handlers(self):
        pass

    async def run_server(self):
        await self.serve()


# config = uvicorn.Config("routers:app",
#                         host="0.0.0.0",
#                         port=8080,
#                         ssl_keyfile=SSL_CERT_PATH + "/key.pem",
#                         ssl_certfile=SSL_CERT_PATH + "/cert.pem",
#                         ssl_keyfile_password="1234"
#                         )

config = uvicorn.Config("routers:app",
                        host="0.0.0.0",
                        port=8080,
                        ssl_keyfile=SSL_CERT_PATH + "/newkey.pem",
                        ssl_certfile=SSL_CERT_PATH + "/cert.pem",
                        ssl_ca_certs=SSL_CERT_PATH + "/DigiCertCA.pem"
                        )


boot_logger.info("(Main Server) create actors...")
logging_service = Logger.options(name=Actors.LOGGER, max_concurrency=500).remote()

boot_logger.info("(Main Server) init logging_service...")
init_processes = ray.get(logging_service.init.remote())
if init_processes == -1:
    boot_logger.error("(Main Server) failed to init logging_service")
    sys.exit()

model_serving = ModelServing.options(name=Actors.MODEL_SERVER).remote()
shared_state = SharedState.options(name=Actors.GLOBAL_STATE).remote()

boot_logger.info("(Main Server) init services...")
init_processes = ray.get([model_serving.init.remote(), shared_state.init.remote()])
api_service = None
try:
    api_service = UvicornServer.options(name=Actors.SERVER).remote(config=config)
except Exception as exc:
    exc_str = ''.join(traceback.format_exception(None, exc, exc.__traceback__))
    init_processes.append(-1)

boot_logger.info("(Main Server) check database connection...")
try:
    db = DBUtil()
    result = db.connection_test()
except Exception as exc:
    boot_logger.error("(Main Server) can not connect to database...: " + exc.__str__())
    init_processes.append(-1)

if -1 in init_processes:
    boot_logger.error("(Main Server) failed to init services")
    ray.kill(api_service)
    ray.kill(model_serving)
    ray.kill(logging_service)
    ray.kill(shared_state)
    sys.exit()
else:
    boot_logger.info("(Main Server) run API server...")
    api_service.run_server.remote()
    boot_logger.info("(Main Server) server initiated successfully...")

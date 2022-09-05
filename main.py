import sys
import configparser

import ray
import uvicorn

from serving import ModelServing
from logger import Logger, BootLogger
from shared_state import SharedState


boot_logger = BootLogger().logger
boot_logger.info("(Main Server) init main server...")

SSL_CERT_PATH = ''
config_parser = configparser.ConfigParser()
try:
    config_parser.read("config/config.ini")
    SSL_CERT_PATH = str(config_parser.get("DEFAULT", "SSL_CERT_PATH"))
except configparser.Error as e:
    boot_logger.error("(Main Server) an error occur when set config...: " + str(e))
    sys.exit()

boot_logger.info("(Main Server) init ray...")
ray.init(dashboard_host="0.0.0.0", dashboard_port=8265)


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


config = uvicorn.Config("routers:app",
                        host="0.0.0.0",
                        port=8080,
                        ssl_keyfile=SSL_CERT_PATH + "/key.pem",
                        ssl_certfile=SSL_CERT_PATH + "/cert.pem",
                        ssl_keyfile_password="1234"
                        )

boot_logger.info("(Main Server) create actors...")
logging_service = Logger.options(name="logging_service", max_concurrency=500).remote()

boot_logger.info("(Main Server) init logging_service...")
init_processes = ray.get(logging_service.init.remote())
if init_processes == -1:
    boot_logger.error("(Main Server) failed to init logging_service")
    sys.exit()

model_serving = ModelServing.options(name="model_serving").remote()
shared_state = SharedState.options(name="shared_state").remote()
api_service = UvicornServer.options(name="API_service").remote(config=config)

boot_logger.info("(Main Server) init services...")
init_processes = ray.get([model_serving.init.remote()])
if -1 in init_processes:
    boot_logger.error("(Main Server) failed to init services")
    ray.kill(api_service)
    ray.kill(model_serving)
    ray.kill(logging_service)
    ray.kill(shared_state)
    sys.exit()
else:
    boot_logger.error("(Main Server) run API server...")
    ray.get(api_service.run_server.remote())
    boot_logger.error("(Main Server) server initiated successfully...")

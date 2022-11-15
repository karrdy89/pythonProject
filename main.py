# *********************************************************************************************************************
# Program Name : main
# Creator : yum kiyeon
# Create Date : 2022. 11. 10
# Modify Desc :
# *********************************************************************************************************************
# ---------------------------------------------------------------------------------------------------------------------
# Date  | Updator   | Remark
#
# ---------------------------------------------------------------------------------------------------------------------
import sys
import configparser
import traceback
import os
from shutil import copytree, copyfile

import ray
import uvicorn

from tf_serving_manger import TfServingManager
from onnx_serving_manager import OnnxServingManager
from db import DBUtil
from logger import Logger, BootLogger
from shared_state import SharedState
from statics import Actors, ROOT_DIR

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

boot_logger.info("(Main Server) patching pre-trained models ...")
pre_trained_dir = "/pre-trained_models/"
saved_model_dir = "/saved_models/"
try:
    for subdir, dirs, files in os.walk(ROOT_DIR + pre_trained_dir):
        saved_model_path = subdir.replace(pre_trained_dir, saved_model_dir)
        if os.path.isdir(saved_model_path):
            if files:
                for file in files:
                    saved_model_file = saved_model_path + "/" + file
                    pre_trained_model_file = subdir + "/" + file
                    if os.path.isfile(saved_model_file):
                        if os.path.getsize(saved_model_file) != os.path.getsize(pre_trained_model_file):
                            copyfile(pre_trained_model_file, saved_model_file)
                    else:
                        copyfile(pre_trained_model_file, saved_model_file)
        else:
            copytree(subdir, saved_model_path)
except Exception as exc:
    boot_logger.error("(Main Server) an error occur when copying pre-trained_model...: " + exc.__str__())
    sys.exit()


boot_logger.info("(Main Server) init ray...")
os.environ["RAY_LOG_TO_STDERR"] = "0"
os.environ["RAY_ROTATION_MAX_BYTES"] = "104857600"
os.environ["RAY_ROTATION_BACKUP_COUNT"] = "5"
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
        try:
            await self.serve()
        except Exception as uv_exc:
            raise uv_exc


# test
# config = uvicorn.Config("routers:app",
#                         host="0.0.0.0",
#                         port=8080,
#                         ssl_keyfile=SSL_CERT_PATH + "/key.pem",
#                         ssl_certfile=SSL_CERT_PATH + "/cert.pem",
#                         ssl_keyfile_password="1234"
#                         )

# build
config = uvicorn.Config("routers:app",
                        host="0.0.0.0",
                        port=8080,
                        ssl_keyfile=SSL_CERT_PATH + "/newkey.pem",
                        ssl_certfile=SSL_CERT_PATH + "/cert.pem",
                        ssl_ca_certs=SSL_CERT_PATH + "/DigiCertCA.pem"
                        )


boot_logger.info("(Main Server) check database connection...")
try:
    db = DBUtil()
    db.connection_test()
except Exception as exc:
    boot_logger.error("(Main Server) can not connect to database...: " + exc.__str__())
    sys.exit()

boot_logger.info("(Main Server) create actors...")
logging_service = Logger.options(name=Actors.LOGGER, max_concurrency=500).remote()

boot_logger.info("(Main Server) init logging_service...")
init_processes = ray.get(logging_service.init.remote())
if init_processes == -1:
    boot_logger.error("(Main Server) failed to init logging_service")
    sys.exit()

shared_state = SharedState.options(name=Actors.GLOBAL_STATE, max_concurrency=2000).remote()
tf_serving_manager = TfServingManager.options(name=Actors.TF_SERVING_MANAGER, max_concurrency=1000).remote()
onx_serving_manager = OnnxServingManager.options(name=Actors.ONNX_SERVING_MANAGER, max_concurrency=1000).remote()

boot_logger.info("(Main Server) init services...")
init_processes = ray.get([tf_serving_manager.init.remote(),
                          onx_serving_manager.init.remote(),
                          shared_state.init.remote()])
api_service = None
try:
    api_service = UvicornServer.options(name=Actors.SERVER, max_concurrency=1000).remote(config=config)
except Exception as exc:
    exc_str = ''.join(traceback.format_exception(None, exc, exc.__traceback__))
    init_processes.append(-1)

if -1 in init_processes:
    boot_logger.error("(Main Server) failed to init services")
    ray.kill(api_service)
    ray.kill(tf_serving_manager)
    ray.kill(logging_service)
    ray.kill(shared_state)
    sys.exit()
else:
    boot_logger.info("(Main Server) run API server...")
    api_service.run_server.remote()
    boot_logger.info("(Main Server) server initiated successfully...")

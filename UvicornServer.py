import contextlib
import time
import threading
import uvicorn

class Server(uvicorn.Server):
    def install_signal_handlers(self):
        pass

    @contextlib.contextmanager
    def run_in_thread(self):
        thread = threading.Thread(target=self.run)
        thread.start()
        try:
            while not self.started:
                time.sleep(1e-3)
            yield
        finally:
            self.should_exit = True
            thread.join()

config = uvicorn.Config("main:app", host="0.0.0.0", port=8081, log_level="debug",
                        ssl_keyfile="/home/ky/cert/key.pem", ssl_certfile="/home/ky/cert/cert.pem",
                        ssl_keyfile_password="1234")
server = Server(config=config)

with server.run_in_thread():
    while server.keep_running:
        pass
    # Server started.
# Server stopped.
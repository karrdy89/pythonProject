import logging
import sys
import os

from tensorboard import program, default, assets

# make tensorboard by given name(log path) and port
# return url
# set expire an hour
# - run -> add timer job, activate and reset(queueing the port and next, if same time or less activate if cron???? change cron to time, save time to activate
# get current time and compare??? or calculate interval ) -> how to reset time of scheduler
# (cur - 11:00, interval 1) -> activate on (cur - 12:00, interval ?)
# add tb on (cur - 11:30, interval 1) -> queueing (cur time diff next interval time -> set interval )
# kill in timer and delete in revers proxy

# set config file and read
# run actual model
# handle with db_util
# error handling
# doc string
# test packaging

TENSORBOARD_PORT_START = 6000
TENSORBOARD_THREAD_MAX = 500


class TensorBoardTool:
    def __init__(self):
        self._port: list[int] = []
        self._port_use: list[int] = []


    def get_port_http(self):
        port = self._port.pop()
        self._port_use.append(port)
        return port

    def release_port_http(self, port: int):
        self._port_use.remove(port)
        self._port.append(port)

    def init(self):
        for i in range(TENSORBOARD_THREAD_MAX):
            self._port.append(TENSORBOARD_PORT_START + i)

    def run(self, dir_path: str):
        # Remove http messages
        log = logging.getLogger('werkzeug').setLevel(logging.ERROR)
        # Start tensorboard server
        tb = program.TensorBoard(default.get_plugins(), assets.get_default_assets_zip_provider())
        tb.configure(argv=[None, '--logdir', dir_path])
        url = tb.launch()
        sys.stdout.write('TensorBoard at %s \n' % url)

# tb_tool = TensorBoardTool(os.path.dirname(os.path.abspath(__file__))+"/train_logs")
# tb_tool.run()

import subprocess
tb = subprocess.check_output("ps -ef | grep tensorboard", shell=True).decode('utf-8')   # need to install ps command in docker
tb = tb.split("\n")
a = tb[0].split()
port = a[1]
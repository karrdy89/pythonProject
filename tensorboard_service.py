import logging
import sys
import os

from tensorboard import program, default

# make tensorboard by given name(log path) and port
# append tensorboard url to reverse proxy
# return url
# set expire an hour
# kill in timer and delete in revers proxy


class TensorBoardTool:

    def __init__(self, dir_path):
        self.dir_path = dir_path

    def run(self):
        # Remove http messages
        log = logging.getLogger('werkzeug').setLevel(logging.ERROR)
        # Start tensorboard server
        tb = program.TensorBoard(default.get_plugins())
        tb.configure(argv=[None, '--logdir', self.dir_path])
        url = tb.launch()
        sys.stdout.write('TensorBoard at %s \n' % url)

tb_tool = TensorBoardTool(os.path.dirname(os.path.abspath(__file__))+"/train_logs")
tb_tool.run()
tb_tool2 = TensorBoardTool(os.path.dirname(os.path.abspath(__file__))+"/train_logs")
tb_tool2.run()
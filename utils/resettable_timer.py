from threading import Timer


class ResettableTimer(object):
    def __init__(self):
        self.interval = None
        self.function = None
        self.timer = None
        self.is_run = False

    def set(self, interval, function):
        self.interval = interval
        self.function = function
        self.timer = Timer(self.interval, self.function)

    def run(self):
        self.timer.start()
        self.is_run = True

    def stop(self):
        self.timer.cancel()
        self.is_run = False

    def reset(self, interval: float):
        self.timer.cancel()
        self.timer = Timer(interval, self.function)
        self.timer.start()

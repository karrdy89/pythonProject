from threading import Timer


class ResettableTimer(object):
    def __init__(self, interval, function):
        self.interval = interval
        self.function = function
        self.timer = Timer(self.interval, self.function)

    def run(self):
        self.timer.start()

    def reset(self, interval: int):
        self.timer.cancel()
        self.timer = Timer(interval, self.function)
        self.timer.start()

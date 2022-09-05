from threading import Timer


class ResettableTimer(object):
    """
    A class that provides resettable thread timer.

    Attributes
    ----------
    interval : actor
        an actor handle of global data store.
    function : TrainResult
        a current result of training.
    timer : str
        a name of pipeline.
    is_run : int
        a batch of each epoch.

    Methods
    -------
    __init__(name: str):
        Constructs all the necessary attributes.
    on_epoch_begin(epoch, logs=None) -> None:
        update training progress to global data store when epoch begin.
    on_epoch_end(epoch, logs=None) -> None:
        update training progress to global data store when epoch end.
    on_batch_end(batch, logs=None) -> None:
        update training progress to global data store when batch end.
    """
    def __init__(self):
        self.interval: float | None = None
        self.function: callable = None
        self.timer: Timer | None = None
        self.is_run: bool = False

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

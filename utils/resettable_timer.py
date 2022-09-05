from threading import Timer


class ResettableTimer(object):
    """
    A class that provides resettable thread timer.

    Attributes
    ----------
    interval : float
        The interval of activation.
    function : callable
        The task to run when activated.
    timer : Timer
        A class of thread timer.
    is_run : bool
        The indicator of timer state.

    Methods
    -------
    __init__(name: str):
        Constructs all the necessary attributes.
    set(self, interval, function) -> None:
        Set thread timer with given interval and function.
    run(self) -> None:
        Start thread timer.
    stop(self) -> None:
        Stop thread timer.
    reset(self, interval: float) -> None:
        Assign new thread timer with given interval and function. old one will remove by GC.
    """
    def __init__(self):
        self.interval: float | None = None
        self.function: callable = None
        self.timer: Timer | None = None
        self.is_run: bool = False

    def set(self, interval, function) -> None:
        self.interval = interval
        self.function = function
        self.timer = Timer(self.interval, self.function)

    def run(self) -> None:
        self.timer.start()
        self.is_run = True

    def stop(self) -> None:
        self.timer.cancel()
        self.is_run = False

    def reset(self, interval: float) -> None:
        self.timer.cancel()
        self.timer = Timer(interval, self.function)
        self.timer.start()

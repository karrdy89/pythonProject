import configparser
import os
import asyncio
import shutil
import uuid
import functools
import logging
import logger
from dataclasses import dataclass, field
from shutil import copytree, rmtree
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle
from typing import Optional

import ray


@ray.remote
class OnnxServingManager:
    def __init__(self):
        self._worker: str = type(self).__name__

    def init(self):
        pass

# manage actor like tf container
# method : deploy
# method : end_deploy
# method : set cycle
# method : fail back
# method : garbage collect
# method : predict
# method : init
# merge with tf serving? issue is max container num, init process, manage point
# only advantage is performance



@dataclass
class ServingActor:
    name: str
    actor: ray.actor
    state: int
    ref_count: int = 0


@dataclass
class ModelDeployState:
    model: tuple[str, int]
    state: int
    cycle_iterator = None
    actors: dict = field(default_factory=dict)


class StateCode:
    ALREADY_EXIST = 1
    IN_PROGRESS = 3
    AVAILABLE = 0
    UN_AVAILABLE = -1
    SHUTDOWN = 4


class ManageType:
    MODEL = 0
    CONTAINER = 1

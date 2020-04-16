
from MTNN.core import Mnist
from MTNN.frameworks.core.agents.base import BaseAgent

class MnistAgent(BaseAgent):

    def __init__(self):
        self.model = Mnist()
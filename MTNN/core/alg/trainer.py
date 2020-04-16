"""
Trainer
"""
import MTNN.core.components.models as modeltype


class Trainer:

    def __init__(self, train_batch_size, optimizer):
        self.train_batch_size = train_batch_size
        self.optimizer = optimizer

    def train(self, model):
        pass




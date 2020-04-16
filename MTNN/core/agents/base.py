"""
The base model class where all other models inherit from.
"""
import torch.nn as nn
import logging


class BaseAgent:
    " The base class contains the base functions to be overloaded by any model you implement."
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Agent")

    def forward(self):
        """
        A single forward pass
        """
        raise NotImplementedError

    def train(self):
        """
        Run main training loop
        :return:
        """
        raise NotImplementedError

    def train_one_epoch(self):
        raise NotImplementedError

    def evaluate(self):
        """
        One cycle of validation
        :return:
        """
        raise NotImplementedError

    def load_checkpoint(self, filename):
        raise NotImplementedError

    def save_checkpoint(self, filename):
        raise NotImplementedError

"""
Example with Mnist
"""
# PyTorch
import torch.nn as nn

# local
import MTNN.core.agents.base as base
import MTNN.core.components.datasets as datasets
import MTNN.core.components.models as models
import MTNN.core.alg.trainer as trainer
import MTNN.core.alg.evaluator as evaluator


class MnistAgent(base.BaseAgent):
    """
    Overwrite BaseAgent class methods
    """
    def __init__(self, config):
        super().__init__()
        self.model = models.MnistModel()
        self.data_loader = datasets.MnistDataLoader()
        self.loss = nn.MSELoss()
        self.trainer = trainer.cascadic(config)
        self.evaluator = evaluator.CategoricalEvaluator()

    def train(self):
        pass
    def evaluate(self):
        pass

# Usage
MnistAgent.train()
MnistAgent.evaluate()

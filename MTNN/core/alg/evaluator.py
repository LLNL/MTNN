# A basic testing/evaluation class.
import torch
from MTNN.utils import deviceloader

class BaseEvaluator:
    """
    Base Evaluator Attributes
    """
    @staticmethod
    def evaluate(model, dataloader):
        raise NotImplementedError


class CategoricalEvaluator(BaseEvaluator):
    """Test for categorical accuracy."""

    @staticmethod
    def evaluate(model, dataloader):
        """"
        Args: 
            model <MTNN.core.components.model> 
            dataloader <torch.utils.data.dataloader> 
        Returns:
            correct <int> Number correct
            total <int> Total test set 
        """
        correct = 0
        total = 0

        with torch.no_grad():
            for data in dataloader:
                input_data, labels = deviceloader.load_data(data, model.device)
                outputs = model(input_data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct, total

# TODO: Remove?
class SimpleRegressionEvaluator(BaseEvaluator):
    @staticmethod
    def evaluate(model, dataloader):
        correct = 0
        total = 0

        print("\n   INPUT  |   TARGET  |   PREDICTION")
        with torch.no_grad():
            for data in dataloader:
                input_data, labels = deviceloader.load_data(data, model.device)
                outputs = model(input_data)
                _, predicted = torch.max(outputs.data, 1)

                printout = "{} | {} | {}".format(input_data, labels.tolist(), predicted.tolist())
                print(printout)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print("Percentage correct {} %".format(correct/total))

        return correct, total

